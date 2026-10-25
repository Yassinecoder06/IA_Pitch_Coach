import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { OwnerGate } from '@/features/auth/OwnerGate';
import { ProfileGate } from '@/features/profiles/ProfileGate';
import { CoachPage } from '@/features/coach/CoachPage';
import { SettingsPage } from '@/features/settings/SettingsPage';
import { PracticePage } from '@/features/rehearse/PracticePage';
import { StudioPage } from '@/features/rehearse/StudioPage';
import { DecksPage } from '@/features/decks/DecksPage';
import { DeckEditorPage } from '@/features/decks/DeckEditorPage';
import { ReportPage } from '@/features/report/ReportPage';
import { PrintReportPage } from '@/features/report/PrintReportPage';
import { HistoryPage } from '@/features/history/HistoryPage';
import { ComparePage } from '@/features/history/ComparePage';
import { ArcadePage } from '@/features/arcade/ArcadePage';
import { GamePage } from '@/features/arcade/GamePage';
import { Shell } from './Shell';

export default function App() {
  return (
    <OwnerGate>
      <ProfileGate>
        <Routes>
          <Route path="/report/:rehearsalId/print" element={<PrintReportPage />} />
          <Route element={<Shell />}>
            <Route index element={<Navigate to="/coach" replace />} />
            <Route path="/coach" element={<CoachPage />} />
            <Route path="/practice" element={<PracticePage />} />
            <Route path="/rehearse" element={<StudioPage />} />
            <Route path="/decks" element={<DecksPage />} />
            <Route path="/decks/:deckId" element={<DeckEditorPage />} />
            <Route path="/report/:rehearsalId" element={<ReportPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/history/compare" element={<ComparePage />} />
            <Route path="/arcade" element={<ArcadePage />} />
            <Route path="/arcade/:gameId" element={<GamePage />} />
            <Route path="/settings/*" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/coach" replace />} />
          </Route>
        </Routes>
      </ProfileGate>
    </OwnerGate>
  );
}
