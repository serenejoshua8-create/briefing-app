import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI } from '@google/genai';
import { createClient } from '@supabase/supabase-js';
import { google } from 'googleapis';
import mammoth from 'mammoth';
import * as cheerio from 'cheerio';
import { PDFParse } from 'pdf-parse';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

/* ============================================================
   Setup
   ============================================================ */
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VERSION = fs.readFileSync(path.join(__dirname, 'VERSION'), 'utf8').trim();

// Errors get appended to a local log file in addition to console output.
// Note: Render's filesystem is ephemeral — this file resets on every
// redeploy/restart, so it's useful for debugging the current running
// instance, not as a durable audit log.
const LOG_FILE = path.join(__dirname, 'error.log');
function logError(context, err) {
  const line = `[${new Date().toISOString()}] ${context}: ${err?.stack || err?.message || String(err)}\n`;
  console.error(line);
  try { fs.appendFileSync(LOG_FILE, line); } catch { /* best-effort only */ }
}

const app = express();
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || '*' }));
// 20mb to leave room for a few PDF uploads sent as base64 (~33% larger than raw)
app.use(express.json({ limit: '20mb' }));

const HAS_ANTHROPIC = !!process.env.ANTHROPIC_API_KEY;
const anthropic = HAS_ANTHROPIC ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
// Dispatch generation always uses Gemini (free), even when an Anthropic key
// is configured. Confirmed via live testing: Claude's web_search-driven
// generation for a 12-topic dispatch realistically costs $60-180/month at
// 2 runs/day (each search round-trip resends the whole accumulated
// conversation as input tokens) — incompatible with a ~$5/month budget.
// Claude still powers the occasional, cheap, search-free weekly digest.
// Flip this back to true if the budget constraint changes.
const USE_CLAUDE_FOR_GENERATION = false;
const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);

const driveAuth = new google.auth.GoogleAuth({
  credentials: JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT_KEY || '{}'),
  scopes: ['https://www.googleapis.com/auth/drive'],
});
const drive = google.drive({ version: 'v3', auth: driveAuth });

const CLAUDE_MODEL = 'claude-sonnet-5';
const GEMINI_MODEL = 'gemini-2.5-flash';
const TOTAL_TOPICS = 12;
const DETAILED_TOPICS = 5;
const MIN_PAYWALLED = 3;
// 9000 was the original value but truncated mid-JSON on a real 12-topic +
// 5-analysis run (confirmed via the "partial" flag) — a full dispatch needs
// more headroom than that.
const DAILY_MAX_TOKENS = 16000;
// Gemini 2.5 Flash's internal "thinking" tokens count against maxOutputTokens,
// so it needs a much larger budget than Claude to finish a 12-topic dispatch
// without getting cut off mid-JSON.
const GEMINI_DAILY_MAX_TOKENS = 32768;

/* ============================================================
   Auth guard — require a shared secret header on mutating routes.
   Cheap but effective for a single-user app with no login system.
   ============================================================ */
function requireSecret(req, res, next) {
  if (!process.env.APP_SHARED_SECRET) return next(); // not configured yet — allow (dev only)
  if (req.headers['x-app-secret'] === process.env.APP_SHARED_SECRET) return next();
  return res.status(401).json({ error: 'Missing or invalid x-app-secret header.' });
}

/* ============================================================
   Sources (same 24 as the prototype)
   ============================================================ */
const SOURCES = [
  { name: 'Reuters', url: 'reuters.com', free: true },
  { name: 'AP News', url: 'apnews.com', free: true },
  { name: 'BBC News', url: 'bbc.co.uk', free: true },
  { name: 'The Guardian', url: 'theguardian.com', free: true },
  { name: 'Politico', url: 'politico.com', free: true },
  { name: 'Politico Europe', url: 'politico.eu', free: true },
  { name: 'Al Jazeera', url: 'aljazeera.com', free: true },
  { name: 'Economic Times', url: 'economictimes.indiatimes.com', free: true },
  { name: 'Hindustan Times', url: 'hindustantimes.com', free: true },
  { name: 'The Hindu', url: 'thehindu.com', free: true },
  { name: 'CFR', url: 'cfr.org', free: true },
  { name: 'White House', url: 'whitehouse.gov', free: true },
  { name: 'U.S. State Dept', url: 'state.gov', free: true },
  { name: 'EU Commission', url: 'ec.europa.eu', free: true },
  { name: 'UK Government', url: 'gov.uk', free: true },
  { name: 'NATO', url: 'nato.int', free: true },
  { name: 'India PIB', url: 'pib.gov.in', free: true },
  { name: 'India MEA', url: 'mea.gov.in', free: true },
  { name: 'Wall Street Journal', url: 'wsj.com', free: false, aliases: ['wsj', 'wall street journal'] },
  { name: 'New York Times', url: 'nytimes.com', free: false, aliases: ['nyt', 'new york times'] },
  { name: 'Washington Post', url: 'washingtonpost.com', free: false, aliases: ['washington post', 'wapo'] },
  { name: 'Financial Times', url: 'ft.com', free: false, aliases: ['financial times'] },
  { name: 'Bloomberg', url: 'bloomberg.com', free: false, aliases: ['bloomberg'] },
  { name: 'The Economist', url: 'economist.com', free: false, aliases: ['economist'] },
];
const PAYWALLED_ALIASES = SOURCES.filter(s => !s.free).flatMap(s => s.aliases || [s.name.toLowerCase()]);
function isPaywalledSource(name) {
  const n = (name || '').toLowerCase();
  return PAYWALLED_ALIASES.some(a => n.includes(a) || a.includes(n));
}

const TOPIC_GROUPS = [
  { group: 'India & U.S.', topics: [
    { id: 'india_us_strategic', label: 'India-U.S. Strategic Partnership' },
    { id: 'us_india_trade', label: 'Trade Relations: India & U.S.' },
    { id: 'india_diplomacy', label: 'India Diplomatic Signals (Modi-Trump, EU, etc.)' },
  ]},
  { group: 'U.S. & Great Powers', topics: [
    { id: 'us_china', label: 'U.S.-China Relations' },
    { id: 'us_indo_pacific', label: 'U.S. Indo-Pacific Engagement' },
    { id: 'tech_trade_supply', label: 'Tech Trade & Supply Chains' },
  ]},
  { group: 'Conflicts & Wars', topics: [
    { id: 'middle_east_war', label: 'Middle East War' },
    { id: 'russia_ukraine_eu', label: 'Russia-Ukraine War & EU Dynamics' },
    { id: 'conflict_updates', label: 'Conflict/War Updates (Russia, Middle East, Taiwan-China)' },
  ]},
  { group: 'Economics, Energy & Policy', topics: [
    { id: 'energy_geo', label: 'Energy & Geo-Economics (oil, supply chains)' },
    { id: 'eu_us_policy', label: 'EU / U.S. Policy Statements (sanctions, trade)' },
    { id: 'official_stmts', label: 'Press Releases & Official Statements (EU, UK Gov, etc.)' },
  ]},
];
const ALL_TOPICS = TOPIC_GROUPS.flatMap(g => g.topics);

/* ============================================================
   JSON extraction (Claude sometimes wraps JSON in prose/fences)
   ============================================================ */
function extractJSON(text) {
  const start = text.indexOf('{');
  if (start === -1) throw new Error('No JSON found in model response.');
  // Scan for the first *balanced* top-level object rather than slicing from the
  // first '{' to the last '}' — some models repeat the full JSON object twice
  // in one response, which the naive slice would concatenate into invalid JSON.
  let depth = 0, inString = false, escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') { inString = true; continue; }
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return JSON.parse(text.slice(start, i + 1));
    }
  }
  throw new Error('No complete JSON object found in model response.');
}

/* ============================================================
   Claude: generate the 12-summary + 5-analysis dispatch
   ============================================================ */
function buildDailyPrompt(config, dateStr, session) {
  const chosen = ALL_TOPICS.filter(t => config.topics.includes(t.id));
  const extras = (config.extraTopics || '').trim();
  const idList = chosen.map(t => t.id).join('|') + (extras ? '|extra' : '');
  const sourceList = SOURCES.map(s => s.url).join(', ');
  const paywalledList = SOURCES.filter(s => !s.free).map(s => `${s.name} (${s.url})`).join(', ');
  const topicLines = chosen.map(t => `  - ${t.label}`).join('\n');
  const extraLine = extras
    ? `\n\nAdditional reader-specified topics (equal priority):\n  - ${extras.split(',').map(s => s.trim()).filter(Boolean).join('\n  - ')}`
    : '';

  return `You are a senior geopolitical and economic intelligence analyst preparing a twice-daily briefing for a single expert reader in Delhi, India.

Today is ${dateStr} (IST). This is the ${session === 'AM' ? 'MORNING (0700 IST)' : 'EVENING (2100 IST)'} briefing.

The reader has selected ONLY these topics:
${topicLines}${extraLine}

Cover ONLY developments within these topics.

=== SOURCING — hard requirement, read carefully ===
You may cite ONLY these exact ${SOURCES.length} domains — no exceptions, no other outlets, no blogs, no aggregators, no think-tank sites, no analyst commentary sites:
${sourceList}

Any point citing a domain outside this exact list will be discarded automatically before publication, so it is worthless to include one. When you search, prefer queries scoped to these domains directly (e.g. "site:reuters.com <topic>", "site:apnews.com <topic>") rather than an open web search, and only fall back to a broader search to discover the story before finding the matching article specifically on one of these ${SOURCES.length} domains.

Each "point" is a separately-sourced quote. Points within one topic may come from different articles/outlets, but every single one must be from the approved list above.
For every point provide: "quote" (10-30 word VERBATIM excerpt copied exactly from the article), "sourceName", and "url".
The "url" MUST be the exact article URL on one of the approved domains — NEVER a homepage or section page (e.g. https://www.wsj.com/ is FORBIDDEN). It must contain a real article path/slug.
If you cannot find a matching article on an approved domain, DROP that point rather than citing an unapproved source or guessing a URL.
Do not reuse one URL for quotes from different articles.

=== STRUCTURE ===
LAYER 1 (all ${TOTAL_TOPICS}): the most significant developments, ranked. Each gets: "thread" (2-4 word tag), "headline" (<12 words), "topicId", "storyDate" (ISO date), "points" (array of exactly 3 sourced quotes as above).
LAYER 2 (top ${DETAILED_TOPICS} of those ${TOTAL_TOPICS}): mark "detailed":true and add "analysis": {whatHappened, rightWrong, delta, forecast, outlook, confidence ("High"|"Medium"|"Low"), confidencePct (0-100), confidenceReason}. The rest get "detailed":false, no analysis.

RECENCY: every story must be from the last 24-48 hours relative to ${dateStr}.
SOURCE DIVERSITY: at least ${MIN_PAYWALLED} of the ${TOTAL_TOPICS} topics must have a point citing a paywalled outlet: ${paywalledList}.

Respond with ONLY valid JSON, no markdown fences, no commentary:
{"topics":[{"thread":"...","headline":"...","topicId":"<one of: ${idList}>","storyDate":"YYYY-MM-DD","detailed":true,"points":[{"quote":"...","sourceName":"...","url":"..."}],"analysis":{...only if detailed:true...}}]}

Rules:
- CRITICAL: never use a straight double-quote (") inside a string value — use single quotes instead. Output must be valid JSON.
- Headline under 12 words, plain, specific.`;
}

async function generateDispatch(config, dateStr, session) {
  const systemPrompt = buildDailyPrompt(config, dateStr, session);

  if (HAS_ANTHROPIC && USE_CLAUDE_FOR_GENERATION) {
    const res = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: DAILY_MAX_TOKENS,
      system: systemPrompt,
      messages: [{ role: 'user', content: `Prepare the ${session} briefing for ${dateStr}.` }],
      // max_uses caps the agentic search loop — each search round-trip resends the
      // whole accumulated conversation as input tokens, so an unbounded loop can
      // balloon into millions of input tokens (observed: 68 searches -> 1.15M
      // input tokens in one request). 20 is enough headroom for ~12 topics x 3
      // quotes without runaway cost.
      tools: [{ type: 'web_search_20250305', name: 'web_search', max_uses: 20 }],
    });
    const text = res.content.filter(b => b.type === 'text').map(b => b.text).join('\n');
    const partial = res.stop_reason === 'max_tokens';
    const parsed = extractJSON(text);
    return { topics: parsed.topics || [], partial };
  }

  // No Anthropic key configured — fall back to Gemini for generation too.
  const res = await genai.models.generateContent({
    model: GEMINI_MODEL,
    contents: `${systemPrompt}\n\nPrepare the ${session} briefing for ${dateStr}.`,
    config: { tools: [{ googleSearch: {} }], maxOutputTokens: GEMINI_DAILY_MAX_TOKENS },
  });
  const text = res.text || '';
  const partial = res.candidates?.[0]?.finishReason === 'MAX_TOKENS';
  const parsed = extractJSON(text);
  return { topics: parsed.topics || [], partial };
}

/* ============================================================
   URL sanity check — reject homepages / non-article links.
   A cheap heuristic (path depth) plus an optional live reachability check.
   ============================================================ */
function looksLikeArticleUrl(url) {
  try {
    const u = new URL(url);
    const path = u.pathname.replace(/\/+$/, '');
    return path.length > 1 && path.split('/').filter(Boolean).length >= 2;
  } catch {
    return false;
  }
}
async function urlIsReachable(url) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(url, { method: 'HEAD', redirect: 'follow', signal: controller.signal });
    clearTimeout(timeout);
    return res.status < 400;
  } catch {
    return false; // network errors / blocks don't necessarily mean the link is fake — treat as "unknown", not fatal
  }
}
// Enforce the "draw only from these sources" instruction in code — models
// (Gemini especially, observed empirically) don't reliably follow it as a
// prompt-only constraint.
const APPROVED_HOSTNAMES = SOURCES.map(s => s.url.toLowerCase());
function isApprovedSourceUrl(url) {
  try {
    const host = new URL(url).hostname.toLowerCase().replace(/^www\./, '');
    return APPROVED_HOSTNAMES.some(h => host === h || host.endsWith('.' + h));
  } catch {
    return false;
  }
}
// Shared by the initial pass and by reconciliation's "fix" path below —
// a corrected topic from a verifier is just as capable of citing a fake or
// off-list URL as the original generation, so it must pass the same checks.
function sanitizeTopicPoints(t, { enforceApprovedDomains = true, validUrls = null } = {}) {
  if (!Array.isArray(t.points)) { t.points = []; return t; }
  t.points = t.points.filter(p => {
    if (!p.quote || !p.sourceName) return false;
    if (!p.url) return !enforceApprovedDomains; // no-URL points (e.g. from an uploaded PDF) only allowed outside the strict autonomous-search path
    // Source-driven mode has an exact set of known-good URLs — require an
    // exact match rather than just a shape check, since a model can garble
    // a character while copying a URL (observed: "laucnh" for "launches").
    if (validUrls) return validUrls.has(p.url);
    if (!looksLikeArticleUrl(p.url)) return false;
    return !enforceApprovedDomains || isApprovedSourceUrl(p.url);
  });
  return t;
}
async function sanitizePoints(topics, opts) {
  return topics.map(t => sanitizeTopicPoints(t, opts)).filter(t => t.points.length > 0);
}

/* ============================================================
   Source-driven generation — the user supplies paywalled article
   URLs/PDFs and government site URLs directly, instead of relying on
   autonomous web search. This removes Claude's web_search tool (and its
   per-search fee + runaway conversation growth) entirely: generation
   becomes a single bounded summarization call over the supplied text.
   ============================================================ */
const MAX_SOURCE_CHARS = 6000; // per-source cap — keeps total token usage predictable regardless of how long a page/PDF is
const MAX_GOV_ITEMS_PER_SITE = 6;
const GOV_RECENCY_HOURS = 72;

function truncateText(text, max) {
  return text.length > max ? text.slice(0, max) + '…' : text;
}

async function fetchHtml(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(url, {
      redirect: 'follow',
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; briefing-app/1.0; +source-fetcher)' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
    return await res.text();
  } finally {
    clearTimeout(timeout);
  }
}

function extractReadableText(html) {
  const $ = cheerio.load(html);
  $('script, style, nav, header, footer, noscript, iframe, svg').remove();
  const title = $('title').first().text().trim();
  const articleText = $('article').text().trim();
  const bodyText = (articleText || $('body').text() || '').replace(/\s+/g, ' ').trim();
  return { title, text: bodyText };
}

async function fetchArticleText(url) {
  try {
    const html = await fetchHtml(url);
    const { title, text } = extractReadableText(html);
    const hostname = new URL(url).hostname.replace(/^www\./, '');
    return { sourceName: title || hostname, url, text: truncateText(text, MAX_SOURCE_CHARS) };
  } catch (e) {
    logError(`fetchArticleText:${url}`, e);
    return null;
  }
}

async function extractPdfText(base64, label) {
  const parser = new PDFParse({ data: Buffer.from(base64, 'base64') });
  try {
    const result = await parser.getText();
    return { sourceName: label || 'Uploaded PDF', url: null, text: truncateText(result.text.replace(/\s+/g, ' ').trim(), MAX_SOURCE_CHARS) };
  } finally {
    await parser.destroy();
  }
}

// Heuristic: scan a government listing page for links whose nearby text
// contains a recognizable date within the last GOV_RECENCY_HOURS, then
// fetch each matching page's text. Government site markup varies widely
// (and some are JS-rendered SPAs this plain fetch can't see), so this is
// best-effort — it won't work equally well on every site.
const DATE_PATTERN = /\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b/i;

async function fetchGovRecentItems(baseUrl, sinceHours = GOV_RECENCY_HOURS) {
  const results = [];
  try {
    const html = await fetchHtml(baseUrl);
    const $ = cheerio.load(html);
    const cutoff = Date.now() - sinceHours * 3600 * 1000;
    const candidates = [];
    $('a[href]').each((_, el) => {
      const href = $(el).attr('href');
      if (!href) return;
      let absoluteUrl;
      try { absoluteUrl = new URL(href, baseUrl).toString(); } catch { return; }
      const context = ($(el).closest('li, article, div').text() || $(el).text() || '').slice(0, 300);
      const match = context.match(DATE_PATTERN);
      if (!match) return;
      const date = new Date(match[0]);
      if (isNaN(date.getTime()) || date.getTime() < cutoff) return;
      candidates.push({ url: absoluteUrl, date });
    });
    const seen = new Set();
    const deduped = candidates
      .sort((a, b) => b.date - a.date)
      .filter(c => (seen.has(c.url) ? false : (seen.add(c.url), true)))
      .slice(0, MAX_GOV_ITEMS_PER_SITE);
    for (const c of deduped) {
      const article = await fetchArticleText(c.url);
      if (article) results.push(article);
    }
  } catch (e) {
    logError(`fetchGovRecentItems:${baseUrl}`, e);
  }
  return results;
}

async function assembleSourceMaterial(sources) {
  const items = [];
  for (const url of sources.urls || []) {
    const article = await fetchArticleText(url);
    if (article) items.push(article);
  }
  for (const govUrl of sources.govUrls || []) {
    items.push(...await fetchGovRecentItems(govUrl));
  }
  for (const pdf of sources.pdfs || []) {
    try {
      const extracted = await extractPdfText(pdf.base64, pdf.sourceName || pdf.filename);
      if (pdf.sourceUrl) extracted.url = pdf.sourceUrl;
      items.push(extracted);
    } catch (e) {
      logError(`extractPdfText:${pdf.filename || 'unknown'}`, e);
    }
  }
  const materialBlock = items
    .map(it => `--- SOURCE: ${it.sourceName}${it.url ? ` (${it.url})` : ''} ---\n${it.text}`)
    .join('\n\n');
  return { items, materialBlock };
}

function buildSourceDrivenPrompt(config, dateStr, session, materialBlock) {
  const chosen = ALL_TOPICS.filter(t => config.topics.includes(t.id));
  const extras = (config.extraTopics || '').trim();
  const idList = chosen.map(t => t.id).join('|') + (extras ? '|extra' : '');
  const topicLines = chosen.map(t => `  - ${t.label}`).join('\n');
  const extraLine = extras
    ? `\n\nAdditional reader-specified topics (equal priority):\n  - ${extras.split(',').map(s => s.trim()).filter(Boolean).join('\n  - ')}`
    : '';

  return `You are a senior geopolitical and economic intelligence analyst preparing a twice-daily briefing for a single expert reader in Delhi, India.

Today is ${dateStr} (IST). This is the ${session === 'AM' ? 'MORNING (0700 IST)' : 'EVENING (2100 IST)'} briefing.

The reader has selected ONLY these topics:
${topicLines}${extraLine}

=== SOURCE MATERIAL — this is your ONLY source of information ===
Do NOT use any outside knowledge or web search. Base every fact and quote strictly on the material below. If the material doesn't support ${TOTAL_TOPICS} distinct topics, return fewer rather than inventing anything.

${materialBlock}

=== SOURCING RULES ===
Each "point" is a quote drawn verbatim (10-30 words) from the material above, tagged with the "sourceName" and "url" it came from — copy the exact url given for that source; if a source has no url listed, omit the "url" field for that point entirely, never invent one.
Do not reuse one quote for multiple points. Do not fabricate a quote not present in the material.

=== STRUCTURE ===
LAYER 1 (up to ${TOTAL_TOPICS}): the most significant developments in the material, ranked. Each gets: "thread" (2-4 word tag), "headline" (<12 words), "topicId", "storyDate" (ISO date, from the material if stated else ${dateStr}), "points" (array of up to 3 sourced quotes as above).
LAYER 2 (top ${DETAILED_TOPICS} of those): mark "detailed":true and add "analysis": {whatHappened, rightWrong, delta, forecast, outlook, confidence ("High"|"Medium"|"Low"), confidencePct (0-100), confidenceReason}. The rest get "detailed":false, no analysis.

Respond with ONLY valid JSON, no markdown fences, no commentary:
{"topics":[{"thread":"...","headline":"...","topicId":"<one of: ${idList}>","storyDate":"YYYY-MM-DD","detailed":true,"points":[{"quote":"...","sourceName":"...","url":"..."}],"analysis":{...only if detailed:true...}}]}

Rules:
- CRITICAL: never use a straight double-quote (") inside a string value — use single quotes instead. Output must be valid JSON.
- Headline under 12 words, plain, specific.`;
}

async function generateDispatchFromSources(config, dateStr, session, materialBlock) {
  const systemPrompt = buildSourceDrivenPrompt(config, dateStr, session, materialBlock);
  const userMsg = `Prepare the ${session} briefing for ${dateStr} from the supplied source material.`;

  // No tools/search in either branch — a plain summarization call over
  // supplied text, not an autonomous search loop. Cost is bounded by input
  // material size, not by an open-ended search budget.
  if (HAS_ANTHROPIC) {
    const res = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: DAILY_MAX_TOKENS,
      system: systemPrompt,
      messages: [{ role: 'user', content: userMsg }],
    });
    const text = res.content.filter(b => b.type === 'text').map(b => b.text).join('\n');
    const partial = res.stop_reason === 'max_tokens';
    const parsed = extractJSON(text);
    return { topics: parsed.topics || [], partial };
  }

  const res = await genai.models.generateContent({
    model: GEMINI_MODEL,
    contents: `${systemPrompt}\n\n${userMsg}`,
    config: { maxOutputTokens: GEMINI_DAILY_MAX_TOKENS },
  });
  const text = res.text || '';
  const partial = res.candidates?.[0]?.finishReason === 'MAX_TOKENS';
  const parsed = extractJSON(text);
  return { topics: parsed.topics || [], partial };
}

/* ============================================================
   Gemini: independent verification pass (separate Google Search grounding)
   ============================================================ */
async function verifyWithGemini(topics, dateStr, session) {
  const prompt = `You are a strict fact-checking editor. Verify this geopolitical briefing before publication.
Briefing date: ${dateStr} (IST). Session: ${session}.

Topics (JSON): ${JSON.stringify(topics.map((t, i) => ({ index: i, thread: t.thread, headline: t.headline, topicId: t.topicId, storyDate: t.storyDate, points: t.points })))}

For each topic, use Google Search to verify:
1. RECENCY — genuinely from the last 24-48 hours of the briefing date.
2. ACCURACY — quotes/names/numbers match current reporting.
3. RELEVANCE — story matches its assigned topicId.

Respond with ONLY valid JSON, no markdown fences:
{"results":[{"index":0,"verdict":"pass"},{"index":1,"verdict":"drop","reason":"stale, from 3 weeks ago"}]}
Use "pass" or "drop" only (no fixes from this pass — Claude's fixes come from a separate reconciliation). Never use straight double-quotes inside string values.`;

  const res = await genai.models.generateContent({
    model: GEMINI_MODEL,
    contents: prompt,
    config: { tools: [{ googleSearch: {} }], maxOutputTokens: GEMINI_DAILY_MAX_TOKENS },
  });
  const text = res.text || '';
  try {
    return extractJSON(text).results || [];
  } catch {
    return []; // if Gemini's response doesn't parse, treat as "no objections" rather than blocking the whole dispatch
  }
}

/* ============================================================
   Gemini: second independent verification pass — can suggest fixes, not
   just pass/drop. Verification always runs on Gemini (free tier) even
   when Claude generates the dispatch: Claude's web_search tool bills per
   search plus the accumulated-conversation input tokens each search
   round-trip resends, so adding a second Claude call here would roughly
   double Claude spend per run for comparatively little verification
   benefit over Gemini's own independent check.
   ============================================================ */
async function verifyWithGeminiFixer(topics, dateStr, session) {
  const prompt = `You are a strict fact-checking editor verifying a geopolitical briefing before publication.
Briefing date: ${dateStr} (IST). Session: ${session}.
Topics (JSON): ${JSON.stringify(topics.map((t, i) => ({ index: i, thread: t.thread, headline: t.headline, topicId: t.topicId, storyDate: t.storyDate, points: t.points })))}

Use Google Search to independently verify recency, accuracy, and topic relevance for each topic.
Respond with ONLY valid JSON, no markdown fences:
{"results":[{"index":0,"verdict":"pass"},{"index":1,"verdict":"fix","corrected":{...full corrected topic, same schema...}},{"index":2,"verdict":"drop","reason":"..."}]}
Never use straight double-quotes inside string values.`;

  const res = await genai.models.generateContent({
    model: GEMINI_MODEL,
    contents: prompt,
    config: { tools: [{ googleSearch: {} }], maxOutputTokens: GEMINI_DAILY_MAX_TOKENS },
  });
  const text = res.text || '';
  try {
    return extractJSON(text).results || [];
  } catch {
    return [];
  }
}

/* ============================================================
   Reconciliation — conservative merge of both verifiers.
   Either verifier saying "drop" wins. A "fix" from the second pass is
   applied. Both passes run on Gemini's free tier regardless of whether
   Claude generated the dispatch — see the comment above verifyWithGemini.
   ============================================================ */
async function verifyDispatch(topics, dateStr, session, sanitizeOpts) {
  const [passA, passB] = await Promise.all([
    verifyWithGemini(topics, dateStr, session),
    verifyWithGeminiFixer(topics, dateStr, session),
  ]);
  let fixed = 0, dropped = 0;
  const next = [];
  topics.forEach((t, i) => {
    const a = passA.find(r => r.index === i);
    const b = passB.find(r => r.index === i);
    if ((a && a.verdict === 'drop') || (b && b.verdict === 'drop')) { dropped++; return; }
    if (b && b.verdict === 'fix' && b.corrected && b.corrected.headline) {
      const corrected = sanitizeTopicPoints(b.corrected, sanitizeOpts);
      if (corrected.points.length > 0) { next.push(corrected); fixed++; }
      else { dropped++; } // the "fix" cited no valid sources — treat as a drop, not a silent pass-through
      return;
    }
    next.push(t);
  });
  return { topics: next, fixed, dropped, clean: fixed === 0 && dropped === 0 };
}

/* ============================================================
   Routes — config
   ============================================================ */
app.get('/api/config', async (req, res) => {
  const { data, error } = await supabase.from('app_config').select('*').eq('id', 1).single();
  if (error) return res.status(500).json({ error: error.message });
  res.json({ topics: data.topics, extraTopics: data.extra_topics });
});

app.post('/api/config', requireSecret, async (req, res) => {
  const { topics, extraTopics } = req.body;
  const { error } = await supabase.from('app_config')
    .update({ topics, extra_topics: extraTopics, updated_at: new Date().toISOString() })
    .eq('id', 1);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true });
});

/* ============================================================
   Routes — briefing generation & retrieval
   ============================================================ */
app.post('/api/briefing/run', requireSecret, async (req, res) => {
  const { session, sources } = req.body;
  if (session !== 'AM' && session !== 'PM') return res.status(400).json({ error: 'session must be AM or PM' });

  try {
    const { data: cfgRow } = await supabase.from('app_config').select('*').eq('id', 1).single();
    const config = { topics: cfgRow?.topics || [], extraTopics: cfgRow?.extra_topics || '' };
    if (!config.topics.length && !config.extraTopics.trim()) {
      return res.status(400).json({ error: 'No topics selected. POST /api/config first.' });
    }

    const dateStr = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });

    const hasSuppliedSources = sources && (
      (sources.urls && sources.urls.length) ||
      (sources.govUrls && sources.govUrls.length) ||
      (sources.pdfs && sources.pdfs.length)
    );

    let rawTopics, partial, sanitized, sanitizeOpts;
    if (hasSuppliedSources) {
      const { items, materialBlock } = await assembleSourceMaterial(sources);
      if (!materialBlock.trim()) return res.status(400).json({ error: 'Could not extract any usable text from the supplied sources.' });
      sanitizeOpts = { enforceApprovedDomains: false, validUrls: new Set(items.filter(it => it.url).map(it => it.url)) };
      ({ topics: rawTopics, partial } = await generateDispatchFromSources(config, dateStr, session, materialBlock));
      if (!rawTopics.length) return res.status(502).json({ error: 'No topics could be drawn from the supplied sources.' });
      sanitized = await sanitizePoints(rawTopics, sanitizeOpts);
    } else {
      ({ topics: rawTopics, partial } = await generateDispatch(config, dateStr, session));
      if (!rawTopics.length) return res.status(502).json({ error: 'Model returned no topics.' });
      sanitized = await sanitizePoints(rawTopics);
    }

    const verified = await verifyDispatch(sanitized, dateStr, session, sanitizeOpts);

    const record = {
      date: dateStr,
      session,
      generated_at: new Date().toISOString(),
      topics: verified.topics,
      partial,
      verify: { fixed: verified.fixed, dropped: verified.dropped, clean: verified.clean },
      meta: { topics: config.topics, extraTopics: config.extraTopics },
    };

    const { error: upsertErr } = await supabase.from('briefings').upsert(record, { onConflict: 'date,session' });
    if (upsertErr) return res.status(500).json({ error: upsertErr.message });

    res.json(record);
  } catch (e) {
    logError('POST /api/briefing/run', e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.get('/api/briefing/:date/:session', async (req, res) => {
  const { date, session } = req.params;
  const { data, error } = await supabase.from('briefings').select('*').eq('date', date).eq('session', session).single();
  if (error) return res.status(404).json({ error: 'Not found' });
  res.json(data);
});

app.get('/api/briefings', async (req, res) => {
  const { data, error } = await supabase.from('briefings').select('*').order('date', { ascending: false });
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

/* ============================================================
   Routes — weekly digest
   ============================================================ */
app.post('/api/weekly/run', requireSecret, async (req, res) => {
  try {
    const now = new Date();
    const day = now.getDay();
    const monday = new Date(now); monday.setDate(now.getDate() + (day === 0 ? -6 : 1 - day));
    const weekStart = monday.toISOString().slice(0, 10);
    const sunday = new Date(monday); sunday.setDate(monday.getDate() + 6);
    const weekEnd = sunday.toISOString().slice(0, 10);

    const { data: weekItems } = await supabase.from('briefings').select('*').gte('date', weekStart).lte('date', weekEnd);
    if (!weekItems || !weekItems.length) return res.status(400).json({ error: 'No dispatches saved this week yet.' });

    const compact = weekItems.map(b => ({
      date: b.date, session: b.session,
      topics: b.topics.map(t => ({ thread: t.thread, headline: t.headline, points: t.points })),
    }));

    const prompt = `Synthesise one week of geopolitical briefings into a digest.
Respond with ONLY valid JSON: {"narrative":"...","topStories":[{"title":"...","summary":"..."}]}
"narrative" under 80 words. Exactly 5 topStories, each summary under 30 words. Never use straight double-quotes inside string values.`;
    const payload = JSON.stringify({ weekStart, weekEnd, briefings: compact });

    let text;
    if (HAS_ANTHROPIC) {
      const resp = await anthropic.messages.create({
        model: CLAUDE_MODEL, max_tokens: 900, system: prompt,
        messages: [{ role: 'user', content: payload }],
      });
      text = resp.content.filter(b => b.type === 'text').map(b => b.text).join('\n');
    } else {
      const resp = await genai.models.generateContent({
        model: GEMINI_MODEL,
        contents: `${prompt}\n\n${payload}`,
        config: { maxOutputTokens: GEMINI_DAILY_MAX_TOKENS },
      });
      text = resp.text || '';
    }
    const parsed = extractJSON(text);

    const record = { week_start: weekStart, week_end: weekEnd, generated_at: new Date().toISOString(), narrative: parsed.narrative, top_stories: parsed.topStories || [] };
    await supabase.from('weekly_digests').upsert(record, { onConflict: 'week_start' });
    res.json(record);
  } catch (e) {
    logError('POST /api/weekly/run', e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

/* ============================================================
   Routes — Google Drive (read-only archive of past notes)
   A service account has no Drive storage quota of its own, so it can
   read files shared with it but cannot create new files in a personal
   Gmail Drive. Generated dispatches are archived in Supabase only.
   ============================================================ */
app.get('/api/drive/archive', async (req, res) => {
  try {
    const folderId = process.env.GOOGLE_DRIVE_FOLDER_ID;
    const list = await drive.files.list({
      q: `'${folderId}' in parents and trashed = false`,
      fields: 'files(id, name, mimeType, modifiedTime, webViewLink)',
      orderBy: 'modifiedTime desc',
      pageSize: 100,
    });
    res.json(list.data.files || []);
  } catch (e) {
    logError('GET /api/drive/archive', e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.get('/api/drive/file/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;
    const meta = await drive.files.get({ fileId, fields: 'name, mimeType' });

    if (meta.data.mimeType === 'application/vnd.google-apps.document') {
      // Native Google Doc — export as plain HTML
      const exported = await drive.files.export({ fileId, mimeType: 'text/html' }, { responseType: 'text' });
      return res.json({ name: meta.data.name, html: exported.data });
    }

    // Uploaded .docx — download raw bytes, convert with mammoth
    const file = await drive.files.get({ fileId, alt: 'media' }, { responseType: 'arraybuffer' });
    const { value: html } = await mammoth.convertToHtml({ buffer: Buffer.from(file.data) });
    res.json({ name: meta.data.name, html });
  } catch (e) {
    logError('GET /api/drive/file/:fileId', e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

/* ============================================================
   Health check, version, logs + start
   ============================================================ */
app.get('/', (req, res) => res.json({ ok: true, service: 'briefing-backend', version: VERSION }));

// Tail the local error log — useful since Render's dashboard log view can be
// awkward to search; this file resets on every redeploy/restart though.
app.get('/api/logs', requireSecret, (req, res) => {
  try {
    const content = fs.existsSync(LOG_FILE) ? fs.readFileSync(LOG_FILE, 'utf8') : '';
    const lines = content.split('\n').filter(Boolean);
    const tail = lines.slice(-200);
    res.json({ lines: tail.length, log: tail.join('\n') });
  } catch (e) {
    res.status(500).json({ error: e.message || String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend v${VERSION} listening on :${PORT}`));
