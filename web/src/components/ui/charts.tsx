import React, { useMemo, useRef, useState } from 'react';

/* ---------- Radar (6-D scorecard) ---------- */

export interface RadarDatum {
  label: string;
  value: number; // 0..max
}

export function RadarChart({ data, max = 10, size = 260, title }: { data: RadarDatum[]; max?: number; size?: number; title: string }) {
  const c = size / 2;
  const r = size / 2 - 44;
  const point = (i: number, v: number) => {
    const a = (Math.PI * 2 * i) / data.length - Math.PI / 2;
    return [c + Math.cos(a) * r * (v / max), c + Math.sin(a) * r * (v / max)] as const;
  };
  const poly = data.map((d, i) => point(i, Math.max(0, Math.min(max, d.value))).join(',')).join(' ');
  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[300px]" role="img" aria-label={title}>
        {[0.25, 0.5, 0.75, 1].map(f => (
          <polygon key={f} points={data.map((_, i) => point(i, max * f).join(',')).join(' ')} fill="none" stroke="var(--viz-grid)" />
        ))}
        {data.map((_, i) => {
          const [x, y] = point(i, max);
          return <line key={i} x1={c} y1={c} x2={x} y2={y} stroke="var(--viz-grid)" />;
        })}
        <polygon points={poly} fill="var(--viz-series-1)" fillOpacity={0.25} stroke="var(--viz-series-1)" strokeWidth={2} strokeLinejoin="round" />
        {data.map((d, i) => {
          const [x, y] = point(i, d.value);
          const [lx, ly] = point(i, max * 1.28);
          return (
            <g key={d.label}>
              <circle cx={x} cy={y} r={4} fill="var(--viz-series-1)" stroke="var(--viz-surface)" strokeWidth={2}>
                <title>{`${d.label}: ${d.value}/${max}`}</title>
              </circle>
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" fontSize={11} fill="var(--viz-text-2)">
                {d.label}
              </text>
              <text x={lx} y={ly + 13} textAnchor="middle" dominantBaseline="middle" fontSize={10} fill="var(--viz-muted)">
                {d.value}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

/* ---------- Horizontal score bars (0-100, higher is better) ---------- */

export function ScoreBars({ items }: { items: { key: string; label: string; value: number | null | undefined; note?: string }[] }) {
  return (
    <ul className="space-y-2" role="list">
      {items.map(it => (
        <li key={it.key} className="grid grid-cols-[110px_minmax(0,1fr)_44px] items-center gap-2 text-sm" title={it.note}>
          <span className="truncate text-[color:var(--viz-text-2)]">{it.label}</span>
          <span data-viz className="h-2.5 overflow-hidden rounded-full" style={{ background: 'var(--viz-grid)' }} aria-hidden>
            {it.value != null && (
              <span data-viz className="block h-full rounded-full" style={{ width: `${Math.max(2, it.value)}%`, background: 'var(--viz-series-1)' }} />
            )}
          </span>
          <span className="text-right font-mono text-xs text-[color:var(--viz-text)]" data-testid={`score-${it.key}`}>
            {it.value == null ? 'n/a' : Math.round(it.value)}
          </span>
        </li>
      ))}
    </ul>
  );
}

/* ---------- Small multiples over time (one y-scale per panel, shared x) ---------- */

export interface TimelinePanel {
  key: string;
  label: string;
  unit: string;
  color: string; // CSS var
  points: { t: number; v: number | null }[];
  domain: [number, number];
  band?: [number, number]; // target band shading
}

export function TimelineChart({
  panels,
  duration,
  marks = [],
  boundaries = [],
  cursor,
  onSeek,
}: {
  panels: TimelinePanel[];
  duration: number;
  marks?: { t: number; label: string }[]; // e.g. fillers
  boundaries?: { t: number; label: string }[]; // slide changes
  cursor?: number | null; // audio playhead
  onSeek?: (t: number) => void;
}) {
  const W = 720;
  const PH = 64; // panel height
  const GAP = 18;
  const LEFT = 92;
  const RIGHT = 12;
  const H = panels.length * (PH + GAP) + 24;
  const x = (t: number) => LEFT + (Math.max(0, Math.min(duration, t)) / Math.max(duration, 1)) * (W - LEFT - RIGHT);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const valueAt = useMemo(
    () => (p: TimelinePanel, t: number) => {
      let best: { t: number; v: number | null } | null = null;
      for (const pt of p.points) if (!best || Math.abs(pt.t - t) < Math.abs(best.t - t)) best = pt;
      return best?.v ?? null;
    },
    [],
  );

  const toTime = (clientX: number) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    return Math.max(0, Math.min(duration, ((px - LEFT) / (W - LEFT - RIGHT)) * duration));
  };

  const ticks = Array.from({ length: Math.min(8, Math.floor(duration / 15) + 1) }, (_, i) => (i * duration) / Math.max(1, Math.min(7, Math.floor(duration / 15))));

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full cursor-crosshair select-none"
        role="img"
        aria-label="Pace, vocal variety and eye contact over the rehearsal"
        onMouseMove={e => setHover(toTime(e.clientX))}
        onMouseLeave={() => setHover(null)}
        onClick={e => onSeek?.(toTime(e.clientX))}
      >
        {boundaries.map((b, i) => (
          <g key={i}>
            <line x1={x(b.t)} x2={x(b.t)} y1={0} y2={H - 20} stroke="var(--viz-grid)" strokeDasharray="3 3" />
            <text x={x(b.t) + 3} y={10} fontSize={9} fill="var(--viz-muted)">
              {b.label}
            </text>
          </g>
        ))}
        {panels.map((p, i) => {
          const top = 16 + i * (PH + GAP);
          const y = (v: number) => top + PH - ((v - p.domain[0]) / (p.domain[1] - p.domain[0])) * PH;
          // Runs of consecutive values become lines; a lone value (gaps on both sides) becomes a dot.
          const segments: string[] = [];
          const dots: [number, number][] = [];
          let run: [number, number][] = [];
          const flush = () => {
            if (run.length === 1) dots.push(run[0]);
            else if (run.length > 1) segments.push(run.map(([px, py], k) => `${k ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(''));
            run = [];
          };
          for (const pt of p.points) {
            if (pt.v == null) {
              flush();
              continue;
            }
            run.push([x(pt.t), y(Math.max(p.domain[0], Math.min(p.domain[1], pt.v)))]);
          }
          flush();
          return (
            <g key={p.key}>
              <text x={0} y={top + 12} fontSize={11} fill="var(--viz-text-2)">
                {p.label}
              </text>
              <text x={0} y={top + 26} fontSize={9} fill="var(--viz-muted)">
                {p.unit}
              </text>
              {p.band && (
                <rect x={LEFT} width={W - LEFT - RIGHT} y={y(p.band[1])} height={y(p.band[0]) - y(p.band[1])} fill="var(--viz-grid)" />
              )}
              <line x1={LEFT} x2={W - RIGHT} y1={top + PH} y2={top + PH} stroke="var(--viz-grid)" />
              {segments.map((d, k) => (
                <path key={k} d={d} fill="none" stroke={p.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              ))}
              {dots.map(([cx, cy], k) => (
                <circle key={k} cx={cx} cy={cy} r={4} fill={p.color} stroke="var(--viz-surface)" strokeWidth={2} />
              ))}
              {p.points.length === 0 && (
                <text x={LEFT + 8} y={top + PH / 2} fontSize={10} fill="var(--viz-muted)">
                  not measured
                </text>
              )}
            </g>
          );
        })}
        {marks.map((m, i) => (
          <line key={i} x1={x(m.t)} x2={x(m.t)} y1={H - 26} y2={H - 18} stroke="var(--viz-series-2)" strokeWidth={2}>
            <title>{`${m.label} at ${m.t.toFixed(1)}s`}</title>
          </line>
        ))}
        {ticks.map((t, i) => (
          <text key={i} x={x(t)} y={H - 4} fontSize={9} fill="var(--viz-muted)" textAnchor="middle">
            {`${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, '0')}`}
          </text>
        ))}
        {cursor != null && <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={H - 20} stroke="var(--viz-text)" strokeWidth={1} />}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H - 20} stroke="var(--viz-text-2)" strokeWidth={1} strokeDasharray="2 2" />}
      </svg>
      {hover != null && (
        <div
          className="pointer-events-none absolute top-0 z-10 rounded-lg border border-white/10 bg-neutral-950/95 px-2 py-1.5 text-[11px] shadow-xl"
          style={{ left: `${(x(hover) / W) * 100}%`, transform: `translateX(${x(hover) > W * 0.7 ? '-105%' : '8px'})` }}
        >
          <div className="font-mono text-[color:var(--viz-muted)]">{`${Math.floor(hover / 60)}:${String(Math.round(hover % 60)).padStart(2, '0')}`}</div>
          {panels.map(p => {
            const v = valueAt(p, hover);
            return (
              <div key={p.key} className="flex items-center gap-1.5 text-[color:var(--viz-text)]">
                <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
                {p.label}: {v == null ? 'n/a' : `${Math.round(v * 10) / 10} ${p.unit}`}
              </div>
            );
          })}
          <div className="text-[color:var(--viz-muted)]">Click to play from here</div>
        </div>
      )}
    </div>
  );
}

/* ---------- Trend across attempts (several 0-100 series on one shared scale) ---------- */

export interface TrendSeries {
  key: string;
  label: string;
  color: string;
  values: (number | null)[]; // one per attempt
}

export function TrendChart({
  labels,
  series,
  onSelect,
  highlight,
}: {
  labels: string[]; // per attempt, e.g. dates
  series: TrendSeries[];
  onSelect?: (index: number) => void;
  highlight?: number | null;
}) {
  const W = 720;
  const H = 220;
  const L = 34;
  const R = 92;
  const T = 12;
  const B = 28;
  const n = labels.length;
  const x = (i: number) => (n <= 1 ? (L + W - R) / 2 : L + (i / (n - 1)) * (W - L - R));
  const y = (v: number) => T + (1 - v / 100) * (H - T - B);
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const nearest = (clientX: number) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    return best;
  };

  if (!n) return <p className="text-sm text-neutral-500">No finished rehearsals yet.</p>;

  return (
    <div className="relative">
      <div className="mb-1 flex flex-wrap gap-3 text-xs text-[color:var(--viz-text-2)]" aria-hidden>
        {series.map(s => (
          <span key={s.key} className="flex items-center gap-1">
            <span data-viz className="h-0.5 w-4 rounded" style={{ background: s.color }} /> {s.label}
          </span>
        ))}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full cursor-pointer"
        role="img"
        aria-label={`Scores across ${n} attempts`}
        onMouseMove={e => setHover(nearest(e.clientX))}
        onMouseLeave={() => setHover(null)}
        onClick={e => onSelect?.(nearest(e.clientX))}
      >
        {[0, 25, 50, 75, 100].map(v => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--viz-grid)" />
            <text x={L - 6} y={y(v)} fontSize={9} textAnchor="end" dominantBaseline="middle" fill="var(--viz-muted)">
              {v}
            </text>
          </g>
        ))}
        {labels.map((l, i) =>
          n <= 10 || i % Math.ceil(n / 10) === 0 ? (
            <text key={i} x={x(i)} y={H - 8} fontSize={9} textAnchor="middle" fill="var(--viz-muted)">
              {l}
            </text>
          ) : null,
        )}
        {highlight != null && <line x1={x(highlight)} x2={x(highlight)} y1={T} y2={H - B} stroke="var(--viz-text-2)" strokeDasharray="2 3" />}
        {series.map(s => {
          const pts = s.values.map((v, i) => (v == null ? null : ([x(i), y(v)] as const)));
          const d = pts.reduce((acc, p, i) => (p ? acc + `${acc && pts[i - 1] ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}` : acc), '');
          const lastIdx = s.values.map((v, i) => (v == null ? -1 : i)).reduce((a, b) => Math.max(a, b), -1);
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {pts.map((p, i) => p && <circle key={i} cx={p[0]} cy={p[1]} r={hover === i ? 5 : 3.5} fill={s.color} stroke="var(--viz-surface)" strokeWidth={2} />)}
              {lastIdx >= 0 && (
                <text x={x(lastIdx) + 8} y={y(s.values[lastIdx]!)} fontSize={10} dominantBaseline="middle" fill="var(--viz-text-2)">
                  {s.label} {Math.round(s.values[lastIdx]!)}
                </text>
              )}
            </g>
          );
        })}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--viz-text-2)" strokeWidth={1} />}
      </svg>
      {hover != null && (
        <div
          className="pointer-events-none absolute top-6 z-10 rounded-lg border border-white/10 bg-neutral-950/95 px-2 py-1.5 text-[11px] shadow-xl"
          style={{ left: `${(x(hover) / W) * 100}%`, transform: `translateX(${x(hover) > W * 0.6 ? '-105%' : '8px'})` }}
        >
          <div className="text-[color:var(--viz-muted)]">
            Attempt {hover + 1} · {labels[hover]}
          </div>
          {series.map(s => (
            <div key={s.key} className="flex items-center gap-1.5 text-[color:var(--viz-text)]">
              <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.label}: {s.values[hover] == null ? 'n/a' : Math.round(s.values[hover]!)}
            </div>
          ))}
          <div className="text-[color:var(--viz-muted)]">Click to open the report</div>
        </div>
      )}
    </div>
  );
}
