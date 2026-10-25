import React, { forwardRef, useEffect, useRef } from 'react';
import { useSensors } from '@/stores/sensors';
import { FACE_POINTS, POSE } from '@/lib/sensors/vision/types';

const BONES: [number, number][] = [
  [POSE.leftShoulder, POSE.rightShoulder],
  [POSE.leftShoulder, POSE.leftElbow],
  [POSE.leftElbow, POSE.leftWrist],
  [POSE.rightShoulder, POSE.rightElbow],
  [POSE.rightElbow, POSE.rightWrist],
  [POSE.leftShoulder, POSE.leftHip],
  [POSE.rightShoulder, POSE.rightHip],
  [POSE.leftHip, POSE.rightHip],
];

/** Mirrored self-view with an optional landmark overlay. */
export const CameraView = forwardRef<HTMLVideoElement, { overlay: boolean; active: boolean }>(function CameraView(
  { overlay, active },
  videoRef,
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pose = useSensors(s => s.pose);
  const face = useSensors(s => s.face);
  const gazeState = useSensors(s => s.gaze?.state);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!overlay) return;
    const W = canvas.width;
    const H = canvas.height;
    const lm = pose?.landmarks;
    if (lm) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.9)';
      ctx.lineWidth = 3;
      for (const [a, b] of BONES) {
        if ((lm[a]?.visibility ?? 0) < 0.4 || (lm[b]?.visibility ?? 0) < 0.4) continue;
        ctx.beginPath();
        ctx.moveTo(lm[a].x * W, lm[a].y * H);
        ctx.lineTo(lm[b].x * W, lm[b].y * H);
        ctx.stroke();
      }
    }
    if (face) {
      ctx.fillStyle = gazeState === 'good' ? '#34d399' : gazeState === 'weak' ? '#fbbf24' : '#f87171';
      for (const i of [FACE_POINTS.leftIris, FACE_POINTS.rightIris]) {
        const p = face.points[i];
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p.x * W, p.y * H, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [pose, face, overlay, gazeState]);

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl bg-black">
      <video ref={videoRef} playsInline muted className="h-full w-full -scale-x-100 object-cover" data-testid="camera" />
      <canvas ref={canvasRef} width={640} height={480} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
      {!active && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-neutral-500">Camera off</div>
      )}
    </div>
  );
});
