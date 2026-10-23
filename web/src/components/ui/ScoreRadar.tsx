import React, { useEffect, useRef } from 'react';
import type { Scorecard } from '@/lib/types';

interface ScoreRadarProps {
  scorecard: Scorecard;
  size?: number;
}

export function ScoreRadar({ scorecard, size = 280 }: ScoreRadarProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const dimensions = [
    { label: 'Clarity', val: scorecard.clarity?.score ?? 7 },
    { label: 'Structure', val: scorecard.structure?.score ?? 7 },
    { label: 'Persuasion', val: scorecard.persuasion?.score ?? 7 },
    { label: 'Delivery', val: scorecard.delivery?.score ?? 7 },
    { label: 'Confidence', val: scorecard.confidence?.score ?? 7 },
    { label: 'Relevance', val: scorecard.relevance?.score ?? 7 },
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Retina support
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const center = size / 2;
    const radius = size * 0.36;
    const numAxes = dimensions.length;
    const angleStep = (Math.PI * 2) / numAxes;

    ctx.clearRect(0, 0, size, size);

    // Draw background grid concentric polygons (levels 2, 4, 6, 8, 10)
    const levels = [0.2, 0.4, 0.6, 0.8, 1.0];
    levels.forEach((lvl, idx) => {
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = center + Math.cos(angle) * (radius * lvl);
        const y = center + Math.sin(angle) * (radius * lvl);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = idx === levels.length - 1 ? 'rgba(255, 255, 255, 0.15)' : 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw axis lines from center to outer ring
    for (let i = 0; i < numAxes; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = center + Math.cos(angle) * radius;
      const y = center + Math.sin(angle) * radius;
      ctx.beginPath();
      ctx.moveTo(center, center);
      ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.stroke();
    }

    // Draw data polygon
    ctx.beginPath();
    dimensions.forEach((dim, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const normalized = Math.min(Math.max(dim.val / 10, 0.1), 1);
      const x = center + Math.cos(angle) * (radius * normalized);
      const y = center + Math.sin(angle) * (radius * normalized);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    // Fill with gradient
    const grad = ctx.createRadialGradient(center, center, 10, center, center, radius);
    grad.addColorStop(0, 'rgba(168, 85, 247, 0.45)');
    grad.addColorStop(1, 'rgba(99, 102, 241, 0.15)');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw vertex dots and values
    dimensions.forEach((dim, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const normalized = Math.min(Math.max(dim.val / 10, 0.1), 1);
      const x = center + Math.cos(angle) * (radius * normalized);
      const y = center + Math.sin(angle) * (radius * normalized);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#9333ea';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label positioning
      const labelX = center + Math.cos(angle) * (radius + 24);
      const labelY = center + Math.sin(angle) * (radius + 24);

      ctx.font = '500 11px Plus Jakarta Sans, sans-serif';
      ctx.fillStyle = 'rgba(241, 240, 245, 0.85)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(dim.label, labelX, labelY);
    });
  }, [scorecard, size]);

  return (
    <div className="flex flex-col items-center">
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
        className="block"
      />
    </div>
  );
}
