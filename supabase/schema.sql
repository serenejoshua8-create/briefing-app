-- Run this once in the Supabase SQL Editor (Project → SQL Editor → New Query → Run)

create extension if not exists "pgcrypto";

-- One row per AM/PM dispatch
create table if not exists briefings (
  id uuid primary key default gen_random_uuid(),
  date date not null,
  session text not null check (session in ('AM','PM')),
  generated_at timestamptz not null default now(),
  topics jsonb not null,           -- array of 12 topic objects (points[], analysis?, detailed, storyDate...)
  partial boolean default false,   -- true if the model response was truncated
  verify jsonb,                    -- {claudeRounds, geminiChecked, fixed, dropped, clean}
  meta jsonb,                      -- {topics: [...topicIds], extraTopics: "..."}
  drive_file_id text,              -- set once saved to Drive, null until then
  unique (date, session)
);

create index if not exists briefings_date_idx on briefings (date desc);

-- One row per generated weekly digest
create table if not exists weekly_digests (
  id uuid primary key default gen_random_uuid(),
  week_start date not null unique,
  week_end date not null,
  generated_at timestamptz not null default now(),
  narrative text,
  top_stories jsonb
);

-- Single-row table holding your topic selection (id is always 1)
create table if not exists app_config (
  id int primary key default 1,
  topics jsonb not null default '[]'::jsonb,
  extra_topics text default '',
  updated_at timestamptz default now()
);
insert into app_config (id, topics, extra_topics)
values (1, '[]'::jsonb, '')
on conflict (id) do nothing;

-- Optional audit log of Drive reads/writes, useful for debugging sync issues
create table if not exists drive_sync_log (
  id uuid primary key default gen_random_uuid(),
  drive_file_id text not null,
  action text not null check (action in ('read','write')),
  synced_at timestamptz default now()
);

-- Stores the refresh token from the one-time Google OAuth consent flow
-- (GET /api/auth/google), so the backend can create real Google Docs in a
-- personal Drive folder -- something the read-only service account can't do.
create table if not exists oauth_tokens (
  provider text primary key,
  refresh_token text not null,
  updated_at timestamptz default now()
);
