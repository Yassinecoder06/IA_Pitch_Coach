import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { OwnerGate } from '@/features/auth/OwnerGate';
import { ProfileGate } from '@/features/profiles/ProfileGate';
import { CoachPage } from '@/features/coach/CoachPage';
import { SettingsPage } from '@/features/settings/SettingsPage';
import { Shell, ComingSoon } from './Shell';

export default function App() {
  return (
    <OwnerGate>
      <ProfileGate>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<Navigate to="/coach" replace />} />
            <Route path="/coach" element={<CoachPage />} />
            <Route path="/rehearse/*" element={<ComingSoon title="Rehearsal Studio" />} />
            <Route path="/decks/*" element={<ComingSoon title="Decks" />} />
            <Route path="/history/*" element={<ComingSoon title="History" />} />
            <Route path="/arcade/*" element={<ComingSoon title="Arcade" />} />
            <Route path="/settings/*" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/coach" replace />} />
          </Route>
        </Routes>
      </ProfileGate>
    </OwnerGate>
  );
}
