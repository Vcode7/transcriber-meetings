/**
 * useGlobalJobPoller — app-level background poller.
 *
 * Mounted once in App.tsx via <GlobalJobTracker />.
 * Polls ALL active jobs in jobsStore every 3 seconds, independent of
 * which page the user is currently viewing.
 *
 * On startup: reconciles localStorage jobs with backend via GET /audio/jobs.
 * On job completion: updates the store and fires a CustomEvent so the
 *   source page (if mounted) can pick up the result without its own poller.
 * On done: shows a Sonner toast with a link to History.
 */
import { useEffect, useRef, useCallback } from 'react'
import { toast } from 'sonner'
import api from '../api/client'
import { useJobsStore, type ActiveJob } from '../store/jobs'
import { useAuthStore } from '../store/auth'

const POLL_INTERVAL_MS = 3000
const RECONCILE_INTERVAL_MS = 30_000  // re-reconcile with backend every 30s

export const JOB_UPDATE_EVENT = 'vs:job-update'
export const JOB_DONE_EVENT = 'vs:job-done'
export const JOB_TRANSCRIPT_READY_EVENT = 'vs:job-transcript-ready'

export interface JobUpdateDetail {
  jobId: string
  status: ActiveJob['status']
  stage: string | null
  result?: unknown
}

function dispatchJobEvent(name: string, detail: JobUpdateDetail) {
  window.dispatchEvent(new CustomEvent(name, { detail }))
}

export function useGlobalJobPoller() {
  const user = useAuthStore((s) => s.user)
  const { jobs, updateJob, reconcile } = useJobsStore()

  // Track which jobs we've already fired the transcript-ready event for
  const transcriptFiredRef = useRef<Set<string>>(new Set())
  const lastReconcileRef = useRef<number>(0)

  // ── Poll a single job ───────────────────────────────────────
  const pollJob = useCallback(async (job: ActiveJob) => {
    try {
      const res = await api.get(`/audio/jobs/${job.jobId}`)
      const data = res.data
      const newStatus = data.status
      const newStage = data.progress || null

      // Update store with latest status/stage
      updateJob(job.jobId, { status: newStatus, stage: newStage })

      if (newStatus === 'transcript_ready' && !transcriptFiredRef.current.has(job.jobId)) {
        transcriptFiredRef.current.add(job.jobId)
        updateJob(job.jobId, { result: data.result })
        dispatchJobEvent(JOB_TRANSCRIPT_READY_EVENT, {
          jobId: job.jobId,
          status: newStatus,
          stage: newStage,
          result: data.result,
        })
      } else if (newStatus === 'done') {
        updateJob(job.jobId, { status: 'done', stage: null, result: data.result })
        dispatchJobEvent(JOB_DONE_EVENT, {
          jobId: job.jobId,
          status: 'done',
          stage: null,
          result: data.result,
        })
        // Show completion toast
        const label = job.filename ? `"${job.filename}"` : 'Recording'
        toast.success(`${label} — Transcript ready!`, {
          description: 'Open History to view the full transcript and AI insights.',
          duration: 8000,
          action: {
            label: 'View',
            onClick: () => { window.location.hash = '#/dashboard/history' },
          },
        })
      } else if (newStatus === 'cancelled') {
        updateJob(job.jobId, { status: 'cancelled', stage: null })
        const label = job.filename ? `"${job.filename}"` : 'Recording'
        toast.info(`${label} — Processing cancelled`, {
          description: 'The operation was cancelled by the user.',
          duration: 6000,
        })
        dispatchJobEvent(JOB_UPDATE_EVENT, {
          jobId: job.jobId,
          status: 'cancelled',
          stage: null,
        })
      } else if (newStatus === 'error') {
        updateJob(job.jobId, { status: 'error', stage: null })
        const label = job.filename ? `"${job.filename}"` : 'Recording'
        toast.error(`${label} — Processing failed`, {
          description: data.error || 'An unexpected error occurred.',
          duration: 10000,
        })
        dispatchJobEvent(JOB_UPDATE_EVENT, {
          jobId: job.jobId,
          status: 'error',
          stage: null,
        })
      } else {
        // Still running — dispatch update for per-page progress bars
        dispatchJobEvent(JOB_UPDATE_EVENT, {
          jobId: job.jobId,
          status: newStatus,
          stage: newStage,
        })
      }
    } catch (err) {
      // Silent retry — network blips shouldn't disrupt the poller
      console.warn(`[GlobalPoller] Poll failed for job ${job.jobId}:`, err)
    }
  }, [updateJob])

  // ── Reconcile with backend (startup + periodic) ──────────────
  const reconcileWithBackend = useCallback(async () => {
    try {
      const res = await api.get('/audio/jobs')
      const backendJobs = (res.data.jobs || []).map((j: {
        job_id: string
        filename: string
        status: string
        progress: string
        created_at: string
      }) => ({
        jobId: j.job_id,
        filename: j.filename,
        status: j.status,
        stage: j.progress || null,
        startedAt: j.created_at,
        source: 'upload' as const,  // source is unknown from backend; will be overridden if local copy exists
      }))
      reconcile(backendJobs)
      lastReconcileRef.current = Date.now()
    } catch (err) {
      console.warn('[GlobalPoller] Reconcile failed (non-fatal):', err)
    }
  }, [reconcile])

  // ── Main polling effect ──────────────────────────────────────
  useEffect(() => {
    if (!user) return  // don't poll when logged out

    // Reconcile on mount
    reconcileWithBackend()

    const intervalId = setInterval(() => {
      const activeJobs = useJobsStore.getState().jobs.filter(
        (j) => j.status !== 'done' && j.status !== 'error' && j.status !== 'cancelled'
      )

      // Poll all active jobs in parallel
      activeJobs.forEach((job) => pollJob(job))

      // Periodic reconcile
      if (Date.now() - lastReconcileRef.current > RECONCILE_INTERVAL_MS) {
        reconcileWithBackend()
      }
    }, POLL_INTERVAL_MS)

    return () => clearInterval(intervalId)
  }, [user, pollJob, reconcileWithBackend])
}
