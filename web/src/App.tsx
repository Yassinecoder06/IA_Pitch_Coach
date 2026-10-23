import React, { useEffect, useState, useRef } from 'react';
import { OwnerGate } from './components/OwnerGate';
import { SessionSidebar } from './components/SessionSidebar';
import { PitchWorkspace } from './components/PitchWorkspace';
import { PitchInsights } from './components/PitchInsights';
import { AiProvidersPanel } from './components/AiProvidersPanel';
import { api, AUTH_REQUIRED_EVENT } from './lib/api';
import { PCM_SAMPLE_RATE } from './audio/pcmRecorder';
import type { CoachingMode, Message, Scorecard, Session, SpeechMetrics } from './lib/types';

function PitchCoachApp() {
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

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingAudioRef = useRef(false);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // 1. Initial Load: Fetch modes & sessions
  useEffect(() => {
    api.getModes().then(res => setModes(res.modes)).catch(console.error);
    refreshSessions();
  }, []);

  async function refreshSessions() {
    try {
      const res = await api.listSessions();
      setSessions(res.sessions);
      if (res.sessions.length > 0 && !activeSessionId) {
        selectSession(res.sessions[0].id);
      }
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }

  // 2. Select / Load Session
  async function selectSession(sessionId: string) {
    try {
      const res = await api.getSession(sessionId);
      setActiveSessionId(sessionId);
      setActiveSessionTitle(res.session.title);
      setCurrentMode(res.session.current_mode);
      setMessages(res.messages);

      // Find latest scores & metrics in history
      const latestAssistant = [...res.messages].reverse().find(m => m.scores);
      if (latestAssistant?.scores) {
        setScorecard(latestAssistant.scores);
      } else {
        setScorecard(null);
      }

      const latestMetrics = [...res.messages].reverse().find(m => m.metrics);
      if (latestMetrics?.metrics) {
        setSpeechMetrics(latestMetrics.metrics);
      } else {
        setSpeechMetrics(null);
      }

      // Update WebSocket config
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'config',
            session_id: sessionId,
            mode: res.session.current_mode,
          })
        );
      }
    } catch (err) {
      console.error('Failed to get session:', err);
    }
  }

  // 3. New Session
  async function handleNewSession() {
    try {
      const newSess = await api.createSession('New Pitch Session', currentMode);
      setSessions(prev => [newSess, ...prev]);
      selectSession(newSess.id);
      showToast('Created new pitch session.');
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  }

  // 4. Rename Session
  async function handleRenameSession(id: string, newTitle: string) {
    try {
      await api.renameSession(id, newTitle);
      setSessions(prev => prev.map(s => (s.id === id ? { ...s, title: newTitle } : s)));
      if (activeSessionId === id) {
        setActiveSessionTitle(newTitle);
      }
      showToast('Session renamed.');
    } catch (err) {
      console.error('Failed to rename session:', err);
    }
  }

  // 5. Delete Session
  async function handleDeleteSession(id: string) {
    try {
      await api.deleteSession(id);
      const remaining = sessions.filter(s => s.id !== id);
      setSessions(remaining);
      if (activeSessionId === id) {
        if (remaining.length > 0) {
          selectSession(remaining[0].id);
        } else {
          setActiveSessionId(null);
          setActiveSessionTitle('New Pitch Session');
          setMessages([]);
          setScorecard(null);
          setSpeechMetrics(null);
        }
      }
      showToast('Session deleted.');
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  }

  // 6. Mode Change
  async function handleSelectMode(modeId: string) {
    setCurrentMode(modeId);
    if (activeSessionId) {
      await api.updateMode(activeSessionId, modeId).catch(console.error);
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'config', mode: modeId }));
    }
  }

  // 7. WebSocket Connection Management
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onclose = event => {
      // 4401: the server rejected the session cookie
      if (event.code === 4401) window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT));
    };

    ws.onopen = () => {
      if (activeSessionId) {
        ws.send(JSON.stringify({ type: 'config', session_id: activeSessionId, mode: currentMode }));
      }
    };

    ws.onmessage = event => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'transcript') {
          // Add user message to conversation list
          const userMsg: Message = {
            id: `temp-${Date.now()}`,
            session_id: activeSessionId || '',
            role: 'user',
            content: msg.text,
            created_at: new Date().toISOString(),
          };
          setMessages(prev => [...prev, userMsg]);
          setIsStreaming(true);
          setStreamingText('');
        } else if (msg.type === 'speech_metrics') {
          setSpeechMetrics(msg.data);
        } else if (msg.type === 'coaching') {
          if (msg.streaming) {
            setIsStreaming(true);
            setStreamingText(prev => prev + msg.text);
          } else if (msg.complete) {
            setIsStreaming(false);
          }
        } else if (msg.type === 'scorecard') {
          setScorecard(msg.data);
        } else if (msg.type === 'audio') {
          playAudioChunk(msg.data);
        } else if (msg.type === 'complete') {
          setIsStreaming(false);
          setStreamingText(currentStream => {
            if (currentStream.trim()) {
              const assistantMsg: Message = {
                id: `coach-${Date.now()}`,
                session_id: activeSessionId || '',
                role: 'assistant',
                content: currentStream,
                created_at: new Date().toISOString(),
              };
              setMessages(prev => [...prev, assistantMsg]);
            }
            return '';
          });
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    // Heartbeat ping
    const pingTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 25000);

    return () => {
      clearInterval(pingTimer);
      ws.close();
    };
  }, [activeSessionId]);

  // Sequential audio playback helper
  function playAudioChunk(b64Wav: string) {
    audioQueueRef.current.push(b64Wav);
    if (!isPlayingAudioRef.current) {
      playNextAudio();
    }
  }

  function playNextAudio() {
    if (audioQueueRef.current.length === 0) {
      isPlayingAudioRef.current = false;
      return;
    }
    isPlayingAudioRef.current = true;
    const b64 = audioQueueRef.current.shift()!;
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    audio.onended = () => playNextAudio();
    audio.onerror = () => playNextAudio();
    audio.play().catch(() => playNextAudio());
  }

  // 8. Turn Actions
  function sendMessage(text: string) {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'text', text }));
    }
  }

  function sendAudio(frames: ArrayBuffer[]) {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'start', sample_rate: PCM_SAMPLE_RATE }));
      for (const frame of frames) ws.send(frame); // binary PCM16 frames
      ws.send(JSON.stringify({ type: 'stop' }));
    }
  }

  async function handleEditLastTurn(newText: string) {
    if (!activeSessionId) return;
    try {
      await api.deleteLastTurn(activeSessionId);
      // Remove last user and coach messages from state
      setMessages(prev => {
        const lastIdx = prev.length - 1;
        const secondLastIdx = prev.length - 2;
        return prev.filter((_, idx) => idx !== lastIdx && idx !== secondLastIdx);
      });
      sendMessage(newText);
    } catch (err) {
      console.error('Failed to edit last turn:', err);
    }
  }

  async function handleReadAloud(text: string) {
    try {
      const res = await api.readAloud(text);
      if (res.data) {
        playAudioChunk(res.data);
      }
    } catch (err) {
      console.error('Failed read aloud:', err);
    }
  }

  async function handleLockApp() {
    await api.logoutOwner().catch(console.error);
    window.location.reload();
  }

  return (
    <div className="flex h-screen w-screen bg-[#0b0a10] overflow-hidden">
      {/* Session Navigation Rail */}
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={selectSession}
        onNewSession={handleNewSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        onOpenSettings={() => setSettingsOpen(true)}
        onLockApp={handleLockApp}
      />

      {/* Main Pitch Workspace */}
      <PitchWorkspace
        modes={modes}
        currentMode={currentMode}
        onSelectMode={handleSelectMode}
        activeSessionTitle={activeSessionTitle}
        messages={messages}
        isStreaming={isStreaming}
        streamingText={streamingText}
        onSendMessage={sendMessage}
        onSendAudio={sendAudio}
        onEditLastTurn={handleEditLastTurn}
        onReadAloud={handleReadAloud}
        activeScorecard={scorecard}
        activeSpeechMetrics={speechMetrics}
      />

      {/* SOTA Insights Panel */}
      <PitchInsights scorecard={scorecard} speechMetrics={speechMetrics} />

      {/* Settings Modal */}
      {settingsOpen && (
        <AiProvidersPanel
          onClose={() => setSettingsOpen(false)}
          onToast={showToast}
        />
      )}

      {/* Toast Notification Banner */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-2xl bg-neutral-900 border border-purple-500/30 px-4 py-2.5 text-xs text-white shadow-2xl animate-in fade-in slide-in-from-bottom-2">
          <span className="h-2 w-2 rounded-full bg-purple-400" />
          {toastMessage}
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <OwnerGate>
      <PitchCoachApp />
    </OwnerGate>
  );
}
