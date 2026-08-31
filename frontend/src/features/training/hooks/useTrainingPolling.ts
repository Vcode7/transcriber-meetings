import { useEffect, useRef, useCallback } from 'react';
import { useTrainingStore } from '../store/trainingStore';
import { fetchJob } from '../api/trainingApi';

export function useTrainingPolling() {
  const activeJobId = useTrainingStore((s) => s.activeJobId);
  const setActiveJob = useTrainingStore((s) => s.setActiveJob);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async (jobId: string) => {
    try {
      const job = await fetchJob(jobId);
      setActiveJob(job);
      if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled') {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (err) {
      // Network error — keep polling
      console.warn('[Training Polling] Fetch error:', err);
    }
  }, [setActiveJob]);

  useEffect(() => {
    if (!activeJobId) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    // Immediate first poll
    poll(activeJobId);

    intervalRef.current = setInterval(() => {
      poll(activeJobId);
    }, 2000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [activeJobId, poll]);
}
