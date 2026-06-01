-- Supabase auth-scoped sessions and chat history

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

create or replace function public.delete_session_owned(p_session_id uuid)
returns boolean
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  v_user_id uuid := auth.uid();
  v_owner_id uuid;
begin
  if v_user_id is null then
    raise exception 'Not authenticated' using errcode = '28000';
  end if;

  select user_id
    into v_owner_id
    from public.sessions
   where id = p_session_id;

  if v_owner_id is null then
    raise exception 'Session not found' using errcode = 'P0002';
  end if;

  if v_owner_id <> v_user_id then
    raise exception 'Not authorized' using errcode = '42501';
  end if;

  delete from public.sessions
   where id = p_session_id
     and user_id = v_user_id;

  return true;
end;
$$;

revoke all on function public.delete_session_owned(uuid) from public;
grant execute on function public.delete_session_owned(uuid) to authenticated;

create or replace function public.delete_last_turn_owned(p_session_id uuid)
returns integer
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  v_user_id uuid := auth.uid();
  v_owner_id uuid;
  v_deleted integer := 0;
begin
  if v_user_id is null then
    raise exception 'Not authenticated' using errcode = '28000';
  end if;

  select user_id
    into v_owner_id
    from public.sessions
   where id = p_session_id;

  if v_owner_id is null then
    raise exception 'Session not found' using errcode = 'P0002';
  end if;

  if v_owner_id <> v_user_id then
    raise exception 'Not authorized' using errcode = '42501';
  end if;

  with latest_messages as (
    select id
      from public.messages
     where session_id = p_session_id
       and user_id = v_user_id
     order by created_at desc, id desc
     limit 2
  )
  delete from public.messages m
   using latest_messages lm
   where m.id = lm.id;

  get diagnostics v_deleted = row_count;

  if v_deleted > 0 then
    delete from public.speech_metrics
     where id in (
       select id
         from public.speech_metrics
        where session_id = p_session_id
          and user_id = v_user_id
        order by created_at desc, id desc
        limit 1
     );
  end if;

  return v_deleted;
end;
$$;

revoke all on function public.delete_last_turn_owned(uuid) from public;
grant execute on function public.delete_last_turn_owned(uuid) to authenticated;
