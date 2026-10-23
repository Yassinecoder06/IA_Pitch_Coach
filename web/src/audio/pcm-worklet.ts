/**
 * AudioWorklet that turns the microphone stream into 16 kHz mono PCM16 frames.
 * Runs on the audio rendering thread; posts ~100 ms Int16Array buffers to the main thread.
 */

// Minimal AudioWorkletGlobalScope typings (not part of lib.dom).
declare const sampleRate: number;
declare function registerProcessor(name: string, ctor: unknown): void;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}

const TARGET_RATE = 16000;
const FRAME_SAMPLES = TARGET_RATE / 10; // 100 ms

class PcmCaptureProcessor extends AudioWorkletProcessor {
  private ratio = sampleRate / TARGET_RATE;
  private carry = 0; // fractional read position into the next input block
  private acc = 0;
  private accCount = 0;
  private out = new Int16Array(FRAME_SAMPLES);
  private outLen = 0;

  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    // Box-filter decimation: average every `ratio` input samples into one output sample.
    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.accCount++;
      this.carry += 1;
      if (this.carry >= this.ratio) {
        this.carry -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accCount));
        this.out[this.outLen++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        this.acc = 0;
        this.accCount = 0;
        if (this.outLen === FRAME_SAMPLES) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(FRAME_SAMPLES);
          this.outLen = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
