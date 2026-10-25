import { QueryClient } from '@tanstack/react-query';
import { ApiError } from './api';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      // Auth errors are handled by the gates; retrying them only delays the prompt.
      retry: (count, err) => !(err instanceof ApiError && err.status === 401) && count < 2,
    },
  },
});
