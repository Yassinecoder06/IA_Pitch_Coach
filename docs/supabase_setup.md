# Supabase Setup Guide for AI Pitch Coach

This guide adds persistent session history, messages, and speech metrics to the existing FastAPI + WebSocket architecture.

## 1. Create a Supabase Project

1. Sign in at https://supabase.com.
2. Select New project.
3. Choose organization, name the project (for example: ai-pitch-coach), and set a strong database password.
4. Pick a region close to your backend deployment.
5. Wait for project provisioning to complete.

After creation, copy these values from Project Settings > API:
- Project URL
- anon public key
- service_role key

Use the service_role key for backend server-side operations.

## 2. Create Required Tables

Run the SQL below in Supabase SQL Editor.

```sql
-- Required extension for UUID generation
create extension if not exists pgcrypto;

create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  title text not null default 'Untitled Session',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  current_mode text not null default 'pitch_analysis',
  context_md text not null default ''
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  transcript text,
  created_at timestamptz not null default now(),
  audio_url text
);

create table if not exists public.speech_metrics (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid references auth.users(id) on delete cascade,
  words_per_minute double precision not null default 0,
  pause_frequency double precision not null default 0,
  pause_duration double precision not null default 0,
  energy_variation double precision not null default 0,
  rhythm_score double precision not null default 0,
  created_at timestamptz not null default now()
);

alter table public.sessions alter column user_id set default auth.uid();
alter table public.messages alter column user_id set default auth.uid();
alter table public.speech_metrics alter column user_id set default auth.uid();

create index if not exists idx_messages_session_created_at
  on public.messages(session_id, created_at);

create index if not exists idx_speech_metrics_session_created_at
  on public.speech_metrics(session_id, created_at desc);

create index if not exists idx_sessions_updated_at
  on public.sessions(updated_at desc);

create index if not exists idx_sessions_user_updated_at
  on public.sessions(user_id, updated_at desc);

create index if not exists idx_messages_user_session_created_at
  on public.messages(user_id, session_id, created_at);

create index if not exists idx_speech_metrics_user_session_created_at
  on public.speech_metrics(user_id, session_id, created_at desc);

create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_sessions_set_updated_at on public.sessions;
create trigger trg_sessions_set_updated_at
before update on public.sessions
for each row execute function public.set_updated_at();

alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.speech_metrics enable row level security;

drop policy if exists sessions_select_own on public.sessions;
create policy sessions_select_own on public.sessions
  for select using (auth.uid() = user_id);

drop policy if exists sessions_insert_own on public.sessions;
create policy sessions_insert_own on public.sessions
  for insert with check (auth.uid() = user_id);

drop policy if exists sessions_update_own on public.sessions;
create policy sessions_update_own on public.sessions
  for update using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists sessions_delete_own on public.sessions;
create policy sessions_delete_own on public.sessions
  for delete using (auth.uid() = user_id);

drop policy if exists messages_select_own on public.messages;
create policy messages_select_own on public.messages
  for select using (auth.uid() = user_id);

drop policy if exists messages_insert_own on public.messages;
create policy messages_insert_own on public.messages
  for insert with check (auth.uid() = user_id);

drop policy if exists messages_update_own on public.messages;
create policy messages_update_own on public.messages
  for update using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists messages_delete_own on public.messages;
create policy messages_delete_own on public.messages
  for delete using (auth.uid() = user_id);

drop policy if exists metrics_select_own on public.speech_metrics;
create policy metrics_select_own on public.speech_metrics
  for select using (auth.uid() = user_id);

drop policy if exists metrics_insert_own on public.speech_metrics;
create policy metrics_insert_own on public.speech_metrics
  for insert with check (auth.uid() = user_id);

drop policy if exists metrics_update_own on public.speech_metrics;
create policy metrics_update_own on public.speech_metrics
  for update using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists metrics_delete_own on public.speech_metrics;
create policy metrics_delete_own on public.speech_metrics
  for delete using (auth.uid() = user_id);
```

## 3. Environment Variables

Add these to your .env file used by backend/config/settings.py:

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY

# Optional: direct Postgres connection string for migrations/CLI tooling
SUPABASE_DB_URL=postgresql://postgres:[YOUR-PASSWORD]@db.YOUR_PROJECT_REF.supabase.co:5432/postgres

# Bounded LLM context window when restoring session context
SESSION_CONTEXT_WINDOW=8
```

## 4. Connect Supabase to FastAPI Backend

This repo now includes:
- backend/storage/supabase_client.py
- backend/storage/session_manager.py

How it works:
1. backend/storage/supabase_client.py creates a Supabase client from env vars.
2. backend/storage/session_manager.py performs table CRUD and markdown summary updates.
3. backend/main.py wires REST and WebSocket handlers to SessionManager.
4. The main migration includes authenticated RPC helpers for deleting a session and deleting the last turn, so delete/edit actions honor `auth.uid()` under RLS.

If SUPABASE_URL or key is missing, session endpoints return disabled/unavailable states and voice coaching still works.

## 5. Python Client Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

New packages used:
- supabase
- librosa
- pyAudioAnalysis

Verify import quickly:

```bash
python -c "from supabase import create_client; import librosa; import pyAudioAnalysis; print('ok')"
```

## 6. Example Queries (Python)

These are equivalent to SessionManager operations.

```python
from supabase import create_client
import os

supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_ROLE_KEY'])

# Create session
session = supabase.table('sessions').insert({
    'title': 'Startup Pitch Practice',
    'current_mode': 'interactive-coaching'
}).execute().data[0]

# Add user message
supabase.table('messages').insert({
    'session_id': session['id'],
    'role': 'user',
    'content': 'Here is my latest pitch draft...',
    'transcript': 'Here is my latest pitch draft...'
}).execute()

# Add assistant message
supabase.table('messages').insert({
    'session_id': session['id'],
    'role': 'assistant',
    'content': 'Your opening is clear. Slow down in the first 20 seconds.'
}).execute()

# Add speech metrics
supabase.table('speech_metrics').insert({
    'session_id': session['id'],
    'words_per_minute': 152.5,
    'pause_frequency': 12.2,
    'pause_duration': 0.44,
    'energy_variation': 0.006,
    'rhythm_score': 7.9
}).execute()

# Fetch session + latest messages
saved = supabase.table('sessions').select('*').eq('id', session['id']).single().execute().data
recent = supabase.table('messages').select('*').eq('session_id', session['id']).order('created_at').limit(8).execute().data
```

## 7. Run Migrations Locally

Option A: Supabase Dashboard SQL Editor
- Paste SQL from section 2.
- Run and confirm table creation.

Option B: Supabase CLI (recommended for team workflow)

```bash
npm install -g supabase
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase migration new init_pitch_coach_sessions
```

Put the SQL from section 2 in the new migration file, then:

```bash
supabase db push
```

For local containerized Supabase:

```bash
supabase start
supabase db reset
```

## 8. Test Integration End-to-End

1. Start backend:

```bash
python ./backend/main.py
```

2. Verify session API:

```bash
curl http://localhost:8000/api/sessions
curl -X POST http://localhost:8000/api/sessions -H "Content-Type: application/json" -d "{\"title\":\"Test Session\",\"mode\":\"conversation\"}"
```

3. In UI:
- Create a session.
- Record voice or send text.
- Resume the same session.
- Confirm prior messages render and mode remains.

4. Validate DB rows in Supabase Table Editor:
- sessions updated_at changes
- messages appended for both voice and text interactions
- speech_metrics rows inserted for voice turns

5. Validate bounded-context behavior:
- Confirm backend sends markdown summary and last N messages to LLM (not full history).

## 9. Security Notes

- Use SUPABASE_SERVICE_ROLE_KEY only on backend server.
- Never expose service_role in frontend JavaScript.
- For production, add RLS policies and service-to-service auth if multiple backend services write to the same tables.
