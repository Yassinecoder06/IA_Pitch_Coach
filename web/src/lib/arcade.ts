import { request } from './api';
import type { Difficulty, GameId } from '@/features/arcade/games';

export interface ArcadeSummary {
  games: { id: GameId; title: string; metric: string; needs: string[]; bests: Partial<Record<Difficulty, number>> }[];
  recent: { id: string; game: GameId; difficulty: Difficulty; score: number; created_at: string }[];
  streak_days: number;
  played_today: boolean;
  queue: { game: GameId; reason: string }[];
}

export const arcadeApi = {
  summary: () => request<ArcadeSummary>('/arcade'),
  record: (run: { game: GameId; difficulty: Difficulty; score: number; stats: Record<string, number> }) =>
    request<{ personal_best: boolean; previous_best: number | null; score: number }>('/arcade/runs', {
      method: 'POST',
      body: JSON.stringify(run),
    }),
};
