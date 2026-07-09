import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI } from '@google/genai';
import { createClient } from '@supabase/supabase-js';
import { google } from 'googleapis';
import mammoth from 'mammoth';

/* ============================================================
   Setup
   ============================================================ */
const app = express();
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || '*' }));
app.use(express.json({ limit: '2mb' }));

const HAS_ANTHROPIC = !!process.env.ANTHROPIC_API_KEY;
const anthropic = HAS_ANTHROPIC ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);

const driveAuth = new google.auth.GoogleAuth({
  credentials: JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT_KEY || '{}'),
  scopes: ['https://www.googleapis.com/auth/drive'],
});
const drive = google.drive({ version: 'v3', auth: driveAuth });

const CLAUDE_MODEL = 'claude-sonnet-4-6';
const GEMINI_MODEL = 'gemini-2.5-flash';
const TOTAL_TOPICS = 5;
const DETAILED_TOPICS = 3;
const MIN_PAYWALLED = 1;
const DAILY_MAX_TOKENS = 9000;
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

  if (HAS_ANTHROPIC) {
    const res = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: DAILY_MAX_TOKENS,
      system: systemPrompt,
      messages: [{ role: 'user', content: `Prepare the ${session} briefing for ${dateStr}.` }],
      tools: [{ type: 'web_search_20250305', name: 'web_search' }],
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
async function sanitizePoints(topics) {
  for (const t of topics) {
    if (!Array.isArray(t.points)) { t.points = []; continue; }
    const kept = [];
    for (const p of t.points) {
      if (!p.url || !looksLikeArticleUrl(p.url)) continue; // drop homepage/fabricated-looking links outright
      if (!isApprovedSourceUrl(p.url)) continue; // drop points citing outlets outside the approved source list
      kept.push(p);
    }
    t.points = kept;
  }
  return topics.filter(t => t.points.length > 0);
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
   Claude: self-verification pass (separate call, re-searches independently)
   ============================================================ */
async function verifyWithClaude(topics, dateStr, session) {
  const prompt = `You are a strict fact-checking editor verifying a geopolitical briefing before publication.
Briefing date: ${dateStr} (IST). Session: ${session}.
Topics (JSON): ${JSON.stringify(topics.map((t, i) => ({ index: i, thread: t.thread, headline: t.headline, topicId: t.topicId, storyDate: t.storyDate, points: t.points })))}

Use web_search to independently verify recency, accuracy, and topic relevance for each topic.
Respond with ONLY valid JSON, no markdown fences:
{"results":[{"index":0,"verdict":"pass"},{"index":1,"verdict":"fix","corrected":{...full corrected topic, same schema...}},{"index":2,"verdict":"drop","reason":"..."}]}
Never use straight double-quotes inside string values.`;

  const res = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 6000,
    system: prompt,
    messages: [{ role: 'user', content: 'Verify the topics now.' }],
    tools: [{ type: 'web_search_20250305', name: 'web_search' }],
  });
  const text = res.content.filter(b => b.type === 'text').map(b => b.text).join('\n');
  try {
    return extractJSON(text).results || [];
  } catch {
    return [];
  }
}

/* ============================================================
   Gemini: second independent verification pass, used only when no
   Anthropic key is configured. Mirrors verifyWithClaude's job (can
   suggest fixes, not just pass/drop) but stays within Gemini's free tier.
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
   Either model saying "drop" wins. A "fix" from the second pass is applied.
   With an Anthropic key: Gemini + Claude, two independent providers.
   Without one: two independent Gemini calls (weaker — same provider twice —
   but still catches issues a single pass would miss).
   ============================================================ */
async function verifyDispatch(topics, dateStr, session) {
  const [passA, passB] = await Promise.all([
    verifyWithGemini(topics, dateStr, session),
    HAS_ANTHROPIC ? verifyWithClaude(topics, dateStr, session) : verifyWithGeminiFixer(topics, dateStr, session),
  ]);
  let fixed = 0, dropped = 0;
  const next = [];
  topics.forEach((t, i) => {
    const a = passA.find(r => r.index === i);
    const b = passB.find(r => r.index === i);
    if ((a && a.verdict === 'drop') || (b && b.verdict === 'drop')) { dropped++; return; }
    if (b && b.verdict === 'fix' && b.corrected && b.corrected.headline) { next.push(b.corrected); fixed++; return; }
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
  const { session } = req.body;
  if (session !== 'AM' && session !== 'PM') return res.status(400).json({ error: 'session must be AM or PM' });

  try {
    const { data: cfgRow } = await supabase.from('app_config').select('*').eq('id', 1).single();
    const config = { topics: cfgRow?.topics || [], extraTopics: cfgRow?.extra_topics || '' };
    if (!config.topics.length && !config.extraTopics.trim()) {
      return res.status(400).json({ error: 'No topics selected. POST /api/config first.' });
    }

    const dateStr = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });

    const { topics: rawTopics, partial } = await generateDispatch(config, dateStr, session);
    if (!rawTopics.length) return res.status(502).json({ error: 'Claude returned no topics.' });

    const sanitized = await sanitizePoints(rawTopics);
    const verified = await verifyDispatch(sanitized, dateStr, session);

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
    console.error(e);
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
    console.error(e);
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
    console.error(e);
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
    console.error(e);
    res.status(500).json({ error: e.message || String(e) });
  }
});

/* ============================================================
   Health check + start
   ============================================================ */
app.get('/', (req, res) => res.json({ ok: true, service: 'briefing-backend' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend listening on :${PORT}`));
