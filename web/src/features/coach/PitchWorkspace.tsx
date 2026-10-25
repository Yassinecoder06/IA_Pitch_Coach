import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  MicOff,
  Send,
  Volume2,
  Sparkles,
  Edit3,
  Timer as TimerIcon,
  Loader2,
} from 'lucide-react';
import { DrillTimer } from './DrillTimer';
import { startPcmRecorder, type PcmRecorder } from '@/audio/pcmRecorder';
import { segmentFillers } from '@/lib/fillers';
import type { CoachingMode, Message, Scorecard, SpeechMetrics } from '@/lib/types';

interface PitchWorkspaceProps {
  modes: CoachingMode[];
  currentMode: string;
  onSelectMode: (modeId: string) => void;
  activeSessionTitle: string;
  messages: Message[];
  isStreaming: boolean;
  statusText?: string | null;
  streamingText: string;
  onSendMessage: (text: string) => void;
  onSendAudio: (frames: ArrayBuffer[]) => void;
  onEditLastTurn: (newText: string) => void;
  onReadAloud: (text: string) => void;
  activeScorecard: Scorecard | null;
  activeSpeechMetrics: SpeechMetrics | null;
}

export function PitchWorkspace({
  modes,
  currentMode,
  onSelectMode,
  activeSessionTitle,
  messages,
  isStreaming,
  statusText,
  streamingText,
  onSendMessage,
  onSendAudio,
  onEditLastTurn,
  onReadAloud,
}: PitchWorkspaceProps) {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showTimer, setShowTimer] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const recorderRef = useRef<PcmRecorder | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll on new message or stream chunk
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  // Recording timer
  useEffect(() => {
    let timer: any = null;
    if (isRecording) {
      setRecordingSeconds(0);
      timer = setInterval(() => setRecordingSeconds(s => s + 1), 1000);
    } else {
      setRecordingSeconds(0);
    }
    return () => clearInterval(timer);
  }, [isRecording]);

  // Audio Recording & Waveform Visualizer
  async function startRecording() {
    try {
      const recorder = await startPcmRecorder();
      recorderRef.current = recorder;
      analyserRef.current = recorder.analyser;
      setIsRecording(true);
      drawWaveform();
    } catch (err) {
      console.error('Failed to open microphone:', err);
    }
  }

  async function stopRecording() {
    const recorder = recorderRef.current;
    recorderRef.current = null;
    setIsRecording(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (!recorder) return;
    const frames = await recorder.stop();
    if (frames.length > 0) onSendAudio(frames);
  }

  function drawWaveform() {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 1.8;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        ctx.fillStyle = `rgba(168, 85, 247, ${Math.max(0.3, dataArray[i] / 255)})`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    };
    render();
  }

  function handleSend() {
    if (!inputText.trim()) return;
    onSendMessage(inputText.trim());
    setInputText('');
  }

  // Highlight filler words inline in user speech (same rules as the backend counter)
  function renderTranscript(text: string) {
    return segmentFillers(text).map((seg, i) =>
      seg.filler ? (
        <span
          key={i}
          className="rounded bg-amber-500/20 px-1 py-0.5 text-amber-300 font-semibold underline decoration-amber-500/50"
          title={`Filler: ${seg.filler}`}
        >
          {seg.text}
        </span>
      ) : (
        <React.Fragment key={i}>{seg.text}</React.Fragment>
      ),
    );
  }

  const activeModeObj = modes.find(m => m.id === currentMode) || modes[0];

  return (
    <div className="flex flex-1 flex-col h-full overflow-hidden bg-panel">
      {/* Top Bar */}
      <header className="glass-panel flex h-16 items-center justify-between px-6 border-b border-white/5 shrink-0">
        <div>
          <h2 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
            {activeSessionTitle}
          </h2>
          <span className="text-[11px] text-purple-400">Mode: {activeModeObj?.name}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Drill Timer Toggle */}
          <button
            onClick={() => setShowTimer(!showTimer)}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-all ${
              showTimer
                ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'
            }`}
          >
            <TimerIcon className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Pitch Timer</span>
          </button>

          {/* Mode Selector Dropdown */}
          <div className="relative">
            <select
              value={currentMode}
              onChange={e => onSelectMode(e.target.value)}
              className="rounded-xl border border-white/10 bg-neutral-900/80 px-3 py-1.5 text-xs font-medium text-white focus:outline-none focus:border-purple-500"
            >
              {modes.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* Optional Timer Banner */}
      {showTimer && (
        <div className="p-4 border-b border-white/5">
          <DrillTimer onClose={() => setShowTimer(false)} />
        </div>
      )}

      {/* Conversation Turns Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-5">
        {messages.map((m, idx) => {
          const isUser = m.role === 'user';
          const isLastUserTurn = isUser && idx === messages.length - 2;

          return (
            <div
              key={m.id || idx}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-3xl mx-auto`}
            >
              <div className="flex items-center gap-2 mb-1 text-[11px] text-neutral-500 px-1">
                <span>{isUser ? 'You (Speaker)' : 'AI Coach'}</span>
                <span>•</span>
                <span>{new Date(m.created_at || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                {isLastUserTurn && (
                  <button
                    onClick={() => onEditLastTurn(m.content)}
                    className="ml-2 flex items-center gap-1 text-purple-400 hover:text-purple-300"
                    title="Edit and re-coach"
                  >
                    <Edit3 className="h-3 w-3" /> Edit
                  </button>
                )}
              </div>

              <div
                className={`rounded-2xl px-4 py-3.5 text-sm leading-relaxed max-w-2xl ${
                  isUser
                    ? 'bg-purple-600/90 text-white shadow-lg shadow-purple-600/15'
                    : 'glass-panel text-neutral-200 border border-white/10'
                }`}
              >
                {isUser ? renderTranscript(m.content) : (
                  <div className="space-y-2 whitespace-pre-wrap">
                    {m.content.replace(/```json[\s\S]*?```/g, '').trim()}
                  </div>
                )}
              </div>

              {!isUser && (
                <button
                  onClick={() => onReadAloud(m.content)}
                  className="mt-1 flex items-center gap-1 text-[11px] text-neutral-500 hover:text-neutral-300 px-1"
                >
                  <Volume2 className="h-3 w-3" /> Read aloud
                </button>
              )}
            </div>
          );
        })}

        {/* Live Streaming Coach Turn */}
        {(isStreaming || statusText) && (
          <div className="flex flex-col items-start max-w-3xl mx-auto" aria-live="polite">
            <div className="flex items-center gap-1.5 mb-1 text-[11px] text-purple-400 px-1">
              <Loader2 className="h-3 w-3 animate-spin" /> {statusText || 'AI Coach is speaking...'}
            </div>
            {streamingText && (
              <div className="glass-panel rounded-2xl px-4 py-3.5 text-sm leading-relaxed text-neutral-200 border border-purple-500/30 whitespace-pre-wrap max-w-2xl">
                {streamingText}
              </div>
            )}
          </div>
        )}

        {messages.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-center p-8 my-auto">
            <div className="flex h-14 w-14 items-center justify-center rounded-3xl bg-purple-600/10 text-purple-400 mb-4 shadow-xl shadow-purple-950/20">
              <Sparkles className="h-7 w-7" />
            </div>
            <h3 className="text-lg font-bold text-white">What pitch would you like to practice?</h3>
            <p className="mt-1 text-xs text-neutral-400 max-w-sm">
              Tap the record button to speak your startup pitch, elevator intro, or investor question response.
            </p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Composer & Microphone Station */}
      <div className="glass-panel p-4 border-t border-white/5 shrink-0 bg-panel/95">
        <div className="max-w-3xl mx-auto flex flex-col gap-3">
          {/* Live Waveform Canvas (when recording) */}
          {isRecording && (
            <div className="flex items-center justify-between rounded-2xl bg-neutral-900/80 px-4 py-2 border border-purple-500/40">
              <div className="flex items-center gap-2 text-xs font-mono text-purple-300">
                <span className="h-2 w-2 rounded-full bg-rose-500 animate-ping" />
                <span>Recording: {Math.floor(recordingSeconds / 60)}:{(recordingSeconds % 60).toString().padStart(2, '0')}</span>
              </div>
              <canvas ref={canvasRef} width="220" height="28" className="rounded-lg" />
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl shadow-lg transition-all ${
                isRecording
                  ? 'bg-rose-600 text-white shadow-rose-600/30 scale-105 animate-pulse'
                  : 'bg-purple-600 text-white shadow-purple-600/25 hover:bg-purple-500 hover:scale-105'
              }`}
              title={isRecording ? 'Stop recording' : 'Start microphone'}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
            </button>

            <div className="flex flex-1 items-center rounded-2xl border border-white/10 bg-neutral-900/60 px-4 py-2 focus-within:border-purple-500">
              <input
                type="text"
                placeholder={isRecording ? 'Listening to your speech...' : 'Type pitch text or tap Record...'}
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={isRecording}
                aria-label="Pitch text"
                className="w-full bg-transparent text-sm text-white placeholder:text-neutral-600 focus:outline-none"
              />
            </div>

            <button
              onClick={handleSend}
              aria-label="Send"
              disabled={!inputText.trim() || isRecording}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-white/5 border border-white/10 text-white hover:bg-purple-600 hover:border-purple-500 transition-all disabled:opacity-30 disabled:hover:bg-white/5 disabled:hover:border-white/10"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
