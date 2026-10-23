import workletUrl from './pcm-worklet.ts?worker&url';

export const PCM_SAMPLE_RATE = 16000;

export interface PcmRecorder {
  /** Analyser on the raw mic signal, for waveform visualisation. */
  analyser: AnalyserNode;
  /** Stops capture, releases the mic and resolves with every captured PCM16 frame. */
  stop: () => Promise<ArrayBuffer[]>;
}

/** Opens the microphone and captures 16 kHz mono PCM16 through an AudioWorklet. */
export async function startPcmRecorder(onFrame?: (frame: ArrayBuffer) => void): Promise<PcmRecorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule(workletUrl);

  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 64;
  // No outputs: the node is pulled by its input alone, so it never needs to reach the speakers.
  const capture = new AudioWorkletNode(ctx, 'pcm-capture', { numberOfInputs: 1, numberOfOutputs: 0 });

  const frames: ArrayBuffer[] = [];
  capture.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
    frames.push(event.data);
    onFrame?.(event.data);
  };

  source.connect(analyser);
  source.connect(capture);

  return {
    analyser,
    stop: async () => {
      source.disconnect();
      capture.port.onmessage = null;
      stream.getTracks().forEach(track => track.stop());
      if (ctx.state !== 'closed') await ctx.close();
      return frames;
    },
  };
}
