import { AUTH_REQUIRED_EVENT, PROFILE_REQUIRED_EVENT } from './api';

export type SocketStatus = 'connecting' | 'open' | 'closed';
type Listener = (msg: any) => void;

/** Close codes sent by the backend (see backend/ws/coach.py). */
export const WS_UNAUTHORIZED = 4401;
export const WS_FORBIDDEN_ORIGIN = 4403;
export const WS_PROFILE_REQUIRED = 4409;

const MAX_BACKOFF_MS = 15000;

/**
 * One long-lived WebSocket with automatic reconnect.
 * JSON messages are sent as text frames, audio as binary frames. The last `config`
 * message is replayed after a reconnect so the server-side state is restored.
 */
export class CoachSocket {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private statusListeners = new Set<(s: SocketStatus) => void>();
  private retries = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private stopped = true;
  private lastConfig: Record<string, unknown> = {};
  status: SocketStatus = 'closed';

  constructor(private readonly path = '/ws') {}

  connect(): void {
    this.stopped = false;
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}${this.path}`);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    this.setStatus('connecting');

    ws.onopen = () => {
      this.retries = 0;
      this.setStatus('open');
      if (Object.keys(this.lastConfig).length) ws.send(JSON.stringify({ type: 'config', ...this.lastConfig }));
      this.pingTimer = setInterval(() => this.send({ type: 'ping' }), 25000);
    };
    ws.onmessage = event => {
      if (typeof event.data !== 'string') return;
      let msg: any;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      this.listeners.forEach(l => l(msg));
    };
    ws.onclose = event => {
      if (this.pingTimer) clearInterval(this.pingTimer);
      this.ws = null;
      this.setStatus('closed');
      if (event.code === WS_UNAUTHORIZED) {
        window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT));
        return;
      }
      if (event.code === WS_PROFILE_REQUIRED) {
        window.dispatchEvent(new CustomEvent(PROFILE_REQUIRED_EVENT));
        return;
      }
      if (event.code === WS_FORBIDDEN_ORIGIN || this.stopped) return;
      const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** this.retries++);
      this.retryTimer = setTimeout(() => this.connect(), delay);
    };
  }

  disconnect(): void {
    this.stopped = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.ws?.close(1000);
    this.ws = null;
  }

  /** Sends a JSON message. Returns false when the socket is not open. */
  send(msg: Record<string, unknown>): boolean {
    if (msg.type === 'config') {
      const { type: _type, ...rest } = msg;
      this.lastConfig = { ...this.lastConfig, ...rest };
    }
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  sendBinary(data: ArrayBuffer): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(data);
    return true;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onStatus(listener: (s: SocketStatus) => void): () => void {
    this.statusListeners.add(listener);
    listener(this.status);
    return () => this.statusListeners.delete(listener);
  }

  private setStatus(s: SocketStatus) {
    this.status = s;
    this.statusListeners.forEach(l => l(s));
  }
}

export const coachSocket = new CoachSocket('/ws');
