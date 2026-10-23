import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ProfilePicker } from '../ProfileGate';
import { api } from '@/lib/api';
import { useAuth } from '@/stores/auth';

const profiles = [
  { id: 'a', name: 'Yassine', color: '#a855f7', has_pin: false, created_at: null, last_used_at: null },
  { id: 'b', name: 'Cofounder', color: '#10b981', has_pin: true, created_at: null, last_used_at: null },
];

describe('ProfilePicker', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, 'listProfiles').mockResolvedValue({ profiles, active_id: null });
  });

  it('selects a profile without a PIN directly', async () => {
    const switchProfile = vi.fn().mockResolvedValue(profiles[0]);
    useAuth.setState({ switchProfile });
    render(<ProfilePicker />);
    fireEvent.click(await screen.findByText('Yassine'));
    await waitFor(() => expect(switchProfile).toHaveBeenCalledWith('a', undefined));
  });

  it('asks for the PIN of a protected profile', async () => {
    const switchProfile = vi.fn().mockResolvedValue(profiles[1]);
    useAuth.setState({ switchProfile });
    render(<ProfilePicker />);
    fireEvent.click(await screen.findByText('Cofounder'));
    expect(switchProfile).not.toHaveBeenCalled();
    const input = screen.getByLabelText('PIN for Cofounder');
    fireEvent.change(input, { target: { value: '12a34' } });
    expect(input).toHaveValue('1234'); // digits only
    fireEvent.click(screen.getByText('Open'));
    await waitFor(() => expect(switchProfile).toHaveBeenCalledWith('b', '1234'));
  });

  it('shows the server error for a wrong PIN', async () => {
    useAuth.setState({ switchProfile: vi.fn().mockRejectedValue(new Error('Wrong PIN.')) });
    render(<ProfilePicker />);
    fireEvent.click(await screen.findByText('Cofounder'));
    fireEvent.change(screen.getByLabelText('PIN for Cofounder'), { target: { value: '0000' } });
    fireEvent.click(screen.getByText('Open'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Wrong PIN.');
  });
});
