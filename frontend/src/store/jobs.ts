/**
 * jobs.ts — Global persistent job store
 *
 * Tracks all active recording jobs so the frontend can reconnect to
 * in-flight processing after navigation, refresh, or app restart.
 *
 * Architecture:
 *   - Job IDs + status are persisted in localStorage under 'vs_active_jobs'
 *   - Full results (transcript, summaries) are NOT persisted — re-fetched on reconnect
 *   - One job per source type is the expected case; multiple are handled gracefully
 *   - Jobs in 'done' or 'error' are kept briefly (for toasts) then cleared
 */
import { create } from 'zustand'
import type { ProcessingResult } from '../types/recording'

export type JobSource = 'record' | 'upload' | 'tab-audio'
export type JobStatus = 'pending' | 'processing' | 'transcript_ready' | 'done' | 'error' | 'cancelled'

export interface ActiveJob {
  jobId: string
  source: JobSource
  status: JobStatus
  /** Current pipeline stage: transcribing | diarizing | identifying_speakers | generating_mom */
  stage: string | null
  filename: string
  startedAt: string   // ISO timestamp
  /** Populated when status reaches transcript_ready or done */
  result?: Partial<ProcessingResult>
}

/**
 * Serializable slice that goes into localStorage.
 * We persist `result` for terminal states (transcript_ready, done) so the
 * transcript is immediately available after navigation/refresh without
 * requiring a re-fetch from the backend.
 */
type PersistedJob = Omit<ActiveJob, 'result'> & {
  result?: Partial<ProcessingResult>
}

const STORAGE_KEY = 'vs_active_jobs'

function loadFromStorage(): ActiveJob[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: PersistedJob[] = JSON.parse(raw)
    // On load, filter out error/cancelled jobs only — keep pending, processing,
    // transcript_ready, and done so completed transcripts survive navigation.
    return parsed
      .filter((j) => j.status !== 'error' && j.status !== 'cancelled')
      .map((j) => ({ ...j }))
  } catch {
    return []
  }
}

function saveToStorage(jobs: ActiveJob[]) {
  try {
    // Persist all non-error, non-cancelled jobs.
    // Include result for transcript_ready/done so transcript survives navigation.
    const toSave: PersistedJob[] = jobs
      .filter((j) => j.status !== 'error' && j.status !== 'cancelled')
      .map((j) => {
        const base: PersistedJob = {
          jobId: j.jobId,
          source: j.source,
          status: j.status,
          stage: j.stage,
          filename: j.filename,
          startedAt: j.startedAt,
        }
        // Persist result only for completed states to keep localStorage lean
        if ((j.status === 'transcript_ready' || j.status === 'done') && j.result) {
          base.result = j.result
        }
        return base
      })
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave))
  } catch {
    // Ignore quota errors
  }
}

interface JobsState {
  jobs: ActiveJob[]

  /** Register a new job (called immediately after receiving recording_id from backend) */
  addJob: (job: Omit<ActiveJob, 'status' | 'stage' | 'result'>) => void

  /** Update fields on an existing job (status, stage, result) */
  updateJob: (jobId: string, patch: Partial<ActiveJob>) => void

  /** Remove a job entirely (e.g., after the done-toast has been shown) */
  removeJob: (jobId: string) => void

  /** Get the most-recent active job for a given source */
  getActiveJobBySource: (source: JobSource) => ActiveJob | undefined

  /** Remove all jobs in done/error state (call after toasts are shown) */
  clearTerminal: () => void

  /** Replace jobs list from backend reconciliation on startup */
  reconcile: (backendJobs: PersistedJob[]) => void
}

export const useJobsStore = create<JobsState>((set, get) => ({
  jobs: loadFromStorage(),

  addJob: (job) => {
    set((state) => {
      const newJob: ActiveJob = { ...job, status: 'pending', stage: null }
      // Replace any existing job from same source that's still pending/processing
      const filtered = state.jobs.filter(
        (j) => !(j.source === job.source && (j.status === 'pending' || j.status === 'processing' || j.status === 'transcript_ready'))
      )
      const next = [newJob, ...filtered]
      saveToStorage(next)
      return { jobs: next }
    })
  },

  updateJob: (jobId, patch) => {
    set((state) => {
      const next = state.jobs.map((j) => (j.jobId === jobId ? { ...j, ...patch } : j))
      saveToStorage(next)
      return { jobs: next }
    })
  },

  removeJob: (jobId) => {
    set((state) => {
      const next = state.jobs.filter((j) => j.jobId !== jobId)
      saveToStorage(next)
      return { jobs: next }
    })
  },

  getActiveJobBySource: (source) => {
    // Returns the most-recent non-error, non-cancelled job for a given source.
    // Includes transcript_ready and done so pages can reconnect to completed jobs.
    return get().jobs.find(
      (j) => j.source === source && j.status !== 'error' && j.status !== 'cancelled'
    )
  },

  clearTerminal: () => {
    set((state) => {
      const next = state.jobs.filter((j) => j.status !== 'done' && j.status !== 'error' && j.status !== 'cancelled')
      saveToStorage(next)
      return { jobs: next }
    })
  },

  reconcile: (backendJobs) => {
    set((state) => {
      // Build a map of local jobs by ID
      const localById = new Map(state.jobs.map((j) => [j.jobId, j]))

      // For each backend job: if we don't have it locally, add it (unknown source)
      // If we do have it, update its status/stage from backend
      const reconciled: ActiveJob[] = [...state.jobs]

      for (const bj of backendJobs) {
        if (localById.has(bj.jobId)) {
          // Update status/stage in place
          const idx = reconciled.findIndex((j) => j.jobId === bj.jobId)
          if (idx !== -1) {
            reconciled[idx] = {
              ...reconciled[idx],
              status: bj.status as JobStatus,
              stage: bj.stage,
              filename: bj.filename || reconciled[idx].filename,
            }
          }
        } else {
          // New job from backend not in local store — add it with unknown source
          // We label it 'upload' as a default since we can't know the source from DB alone
          reconciled.push({
            jobId: bj.jobId,
            source: 'upload',
            status: bj.status as JobStatus,
            stage: bj.stage,
            filename: bj.filename,
            startedAt: bj.startedAt,
          })
        }
      }

      // Remove local jobs that are no longer active on the backend
      const backendIds = new Set(backendJobs.map((j) => j.jobId))
      const final = reconciled.filter(
        // Keep backend-known jobs; also keep done jobs that have a result loaded
        // (the backend doesn't return done jobs from /audio/jobs, but we want
        //  to retain them locally until the page explicitly clears them).
        (j) => backendIds.has(j.jobId) ||
               j.status === 'done' ||
               (j.status !== 'pending' && j.status !== 'processing')
      )

      saveToStorage(final)
      return { jobs: final }
    })
  },
}))
