import React, { useCallback, useEffect, useRef, useState } from 'react';
import { SessionSidebar } from './SessionSidebar';
import { PitchWorkspace } from './PitchWorkspace';
import { PitchInsights } from './PitchInsights';
import { api } from '@/lib/api';
import { coachSocket } from '@/lib/ws';
import { PCM_SAMPLE_RATE } from '@/audio/pcmRecorder';
import { toast } from '@/stores/toast';
import type { CoachingMode, Message, Scorecard, Session, SpeechMetrics } from '@/lib/types';

/** Plays base64 WAV clips one after another. */
function useAudioQueue() {
  const queue = useRef<string[]>([]);
  const playing = useRef(false);

  const playNext = useCallback(() => {
    const b64 = queue.current.shift();
    if (!b64) {
      playing.current = false;
      return;
    }
    playing.current = true;
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audio.onended = playNext;
    audio.onerror = playNext;
    audio.play().catch(playNext);
  }, []);

  return useCallback(
    (b64: string) => {
      queue.current.push(b64);
      if (!playing.current) playNext();
    },
    [playNext],
  );
}

export function CoachPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeSessionTitle, setActiveSessionTitle] = useState('New Pitch Session');
  const [currentMode, setCurrentMode] = useState('pitch_analysis');
  const [modes, setModes] = useState<CoachingMode[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [speechMetrics, setSpeechMetrics] = useState<SpeechMetrics | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  // The socket listener is registered once; refs give it the current values.
  const activeSessionRef = useRef<string | null>(null);
  const streamRef = useRef('');
  const playAudio = useAudioQueue();

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  const selectSession = useCallback(async (sessionId: string) => {
    try {
      const res = await api.getSession(sessionId);
      setActiveSessionId(sessionId);
      setActiveSessionTitle(res.session.title);
      setCurrentMode(res.session.current_mode);
      setMessages(res.messages);
      setScorecard([...res.messages].reverse().find(m => m.scores)?.scores ?? null);
      setSpeechMetrics([...res.messages].reverse().find(m => m.metrics)?.metrics ?? null);
      coachSocket.send({ type: 'config', session_id: sessionId, mode: res.session.current_mode });
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to open session', 'error');
    }
  }, []);

  useEffect(() => {
    api.getModes().then(res => setModes(res.modes)).catch(console.error);
    api
      .listSessions()
      .then(res => {
        setSessions(res.sessions);
        if (res.sessions.length > 0) void selectSession(res.sessions[0].id);
      })
      .catch(console.error);
  }, [selectSession]);

  // One socket for the page; handlers read refs so they never go stale.
  useEffect(() => {
    coachSocket.connect();
    const unsubscribe = coachSocket.subscribe(msg => {
      switch (msg.type) {
        case 'transcript':
          setMessages(prev => [
            ...prev,
            {
              id: `temp-${Date.now()}`,
              session_id: activeSessionRef.current || '',
              role: 'user',
              content: msg.text,
              created_at: new Date().toISOString(),
            },
          ]);
          streamRef.current = '';
          setStreamingText('');
          setIsStreaming(true);
          break;
        case 'status':
          setStatus(msg.message);
          break;
        case 'speech_metrics':
          setSpeechMetrics(msg.data);
          break;
        case 'coaching':
          if (msg.streaming) {
            streamRef.current += msg.text;
            setStreamingText(streamRef.current);
          }
          break;
        case 'scorecard':
          setScorecard(msg.data);
          break;
        case 'audio':
          playAudio(msg.data);
          break;
        case 'error':
          toast(msg.message, 'error');
          setStatus(null);
          setIsStreaming(false);
          break;
        case 'complete': {
          const text = streamRef.current;
          streamRef.current = '';
          setIsStreaming(false);
          setStreamingText('');
          setStatus(null);
          if (text.trim()) {
            setMessages(prev => [
              ...prev,
              {
                id: `coach-${Date.now()}`,
                session_id: activeSessionRef.current || '',
                role: 'assistant',
                content: text,
                created_at: new Date().toISOString(),
              },
            ]);
          }
          break;
        }
      }
    });
    return unsubscribe;
  }, [playAudio]);

  async function handleNewSession() {
    try {
      const created = await api.createSession('New Pitch Session', currentMode);
      setSessions(prev => [created, ...prev]);
      await selectSession(created.id);
      toast('Created new pitch session.');
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to create session', 'error');
    }
  }

  async function handleRenameSession(id: string, newTitle: string) {
    try {
      await api.renameSession(id, newTitle);
      setSessions(prev => prev.map(s => (s.id === id ? { ...s, title: newTitle } : s)));
      if (activeSessionId === id) setActiveSessionTitle(newTitle);
      toast('Session renamed.');
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Rename failed', 'error');
    }
  }

  async function handleDeleteSession(id: string) {
    try {
      await api.deleteSession(id);
      const remaining = sessions.filter(s => s.id !== id);
      setSessions(remaining);
      if (activeSessionId === id) {
        if (remaining.length > 0) {
          await selectSession(remaining[0].id);
        } else {
          setActiveSessionId(null);
          setActiveSessionTitle('New Pitch Session');
          setMessages([]);
          setScorecard(null);
          setSpeechMetrics(null);
          coachSocket.send({ type: 'config', session_id: null });
        }
      }
      toast('Session deleted.');
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Delete failed', 'error');
    }
  }

  async function handleSelectMode(modeId: string) {
    setCurrentMode(modeId);
    if (activeSessionId) await api.updateMode(activeSessionId, modeId).catch(console.error);
    coachSocket.send({ type: 'config', mode: modeId });
  }

  function sendMessage(text: string) {
    if (!coachSocket.send({ type: 'text', text })) toast('Not connected to the coach yet. Retrying...', 'error');
  }

  function sendAudio(frames: ArrayBuffer[]) {
    if (!coachSocket.send({ type: 'start', sample_rate: PCM_SAMPLE_RATE })) {
      toast('Not connected to the coach yet. Retrying...', 'error');
      return;
    }
    for (const frame of frames) coachSocket.sendBinary(frame);
    coachSocket.send({ type: 'stop' });
  }

  async function handleEditLastTurn(newText: string) {
    if (!activeSessionId) return;
    try {
      await api.deleteLastTurn(activeSessionId);
      setMessages(prev => prev.slice(0, -2));
      // Reload so the server-side conversation history no longer contains the removed turn.
      coachSocket.send({ type: 'config', session_id: activeSessionId });
      sendMessage(newText);
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Edit failed', 'error');
    }
  }

  async function handleReadAloud(text: string) {
    try {
      const res = await api.readAloud(text);
      if (res.data) playAudio(res.data);
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Read aloud failed', 'error');
    }
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={id => void selectSession(id)}
        onNewSession={handleNewSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
      />
      <PitchWorkspace
        modes={modes}
        currentMode={currentMode}
        onSelectMode={handleSelectMode}
        activeSessionTitle={activeSessionTitle}
        messages={messages}
        isStreaming={isStreaming}
        streamingText={streamingText}
        statusText={status}
        onSendMessage={sendMessage}
        onSendAudio={sendAudio}
        onEditLastTurn={handleEditLastTurn}
        onReadAloud={handleReadAloud}
        activeScorecard={scorecard}
        activeSpeechMetrics={speechMetrics}
      />
      <PitchInsights scorecard={scorecard} speechMetrics={speechMetrics} />
    </div>
  );
}
