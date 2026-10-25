import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoachSocket, WS_PROFILE_REQUIRED, WS_UNAUTHORIZED } from '../ws';
import { AUTH_REQUIRED_EVENT, PROFILE_REQUIRED_EVENT } from '../api';

class FakeWS {
  static CONNECTING = 0;
  static OPEN = 1;
  static instances: FakeWS[] = [];
  readyState = FakeWS.CONNECTING;
  binaryType = 'blob';
  sent: unknown[] = [];
  onopen?: () => void;
  onmessage?: (e: { data: unknown }) => void;
  onclose?: (e: { code: number }) => void;
  constructor(readonly url: string) {
    FakeWS.instances.push(this);
  }
  send(data: unknown) {
    this.sent.push(data);
  }
  close(code = 1000) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
  open() {
    this.readyState = FakeWS.OPEN;
    this.onopen?.();
  }
}

describe('CoachSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWS.instances = [];
    vi.stubGlobal('WebSocket', FakeWS);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('replays the last config after reconnecting', () => {
    const s = new CoachSocket('/ws');
    s.connect();
    FakeWS.instances[0].open();
    s.send({ type: 'config', session_id: 'abc', mode: 'investor_qa' });
    s.send({ type: 'config', mode: 'pitch_analysis' });

    FakeWS.instances[0].close(1006); // network drop
    vi.advanceTimersByTime(600);
    expect(FakeWS.instances).toHaveLength(2);
    FakeWS.instances[1].open();
    expect(JSON.parse(FakeWS.instances[1].sent[0] as string)).toEqual({
      type: 'config',
      session_id: 'abc',
      mode: 'pitch_analysis',
    });
  });

  it('backs off exponentially between reconnects', () => {
    const s = new CoachSocket('/ws');
    s.connect();
    FakeWS.instances[0].close(1006);
    vi.advanceTimersByTime(499);
    expect(FakeWS.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWS.instances).toHaveLength(2);
    FakeWS.instances[1].close(1006);
    vi.advanceTimersByTime(999);
    expect(FakeWS.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeWS.instances).toHaveLength(3);
  });

  it.each([
    [WS_UNAUTHORIZED, AUTH_REQUIRED_EVENT],
    [WS_PROFILE_REQUIRED, PROFILE_REQUIRED_EVENT],
  ])('close code %i raises %s and does not reconnect', (code, eventName) => {
    const handler = vi.fn();
    window.addEventListener(eventName, handler);
    const s = new CoachSocket('/ws');
    s.connect();
    FakeWS.instances[0].close(code);
    vi.advanceTimersByTime(30000);
    expect(handler).toHaveBeenCalledOnce();
    expect(FakeWS.instances).toHaveLength(1);
    window.removeEventListener(eventName, handler);
  });

  it('delivers JSON messages to subscribers and ignores binary', () => {
    const s = new CoachSocket('/ws');
    const got: unknown[] = [];
    s.subscribe(m => got.push(m));
    s.connect();
    FakeWS.instances[0].open();
    FakeWS.instances[0].onmessage?.({ data: '{"type":"pong"}' });
    FakeWS.instances[0].onmessage?.({ data: new ArrayBuffer(4) });
    FakeWS.instances[0].onmessage?.({ data: 'not json' });
    expect(got).toEqual([{ type: 'pong' }]);
  });

  it('does not send while disconnected', () => {
    const s = new CoachSocket('/ws');
    expect(s.send({ type: 'text', text: 'hi' })).toBe(false);
    expect(s.sendBinary(new ArrayBuffer(2))).toBe(false);
  });
});
