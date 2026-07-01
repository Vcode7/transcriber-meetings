import { useState, useEffect, useRef, useCallback } from 'react'
import api from '../api/client'
import type { ProcessingResult } from '../types/recording'

type JobStatus = 'pending' | 'processing' | 'transcript_ready' | 'done' | 'error'

interface PollResult {
  status: JobStatus
  progress?: string
  ai_generating?: boolean
  result?: Partial<ProcessingResult>
  error?: string
}

interface UseJobPollerOptions {
  onTranscriptReady?: (result: Partial<ProcessingResult>) => void
  onDone?: (result: ProcessingResult) => void
}

export function useJobPoller(
  recordingId: string | null,
  onDoneOrOptions?: ((result: ProcessingResult) => void) | UseJobPollerOptions,
) {
  // Support both legacy call signature (onDone fn) and new options object
  const opts: UseJobPollerOptions =
    typeof onDoneOrOptions === 'function'
      ? { onDone: onDoneOrOptions }
      : (onDoneOrOptions ?? {})

  const { onTranscriptReady, onDone } = opts

  // Ref-ify callbacks so changing them never restarts the polling interval
  const onTranscriptReadyRef = useRef(onTranscriptReady)
  const onDoneRef = useRef(onDone)
  useEffect(() => { onTranscriptReadyRef.current = onTranscriptReady }, [onTranscriptReady])
  useEffect(() => { onDoneRef.current = onDone }, [onDone])

  const [data, setData] = useState<PollResult | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // Track whether we already fired onTranscriptReady to avoid repeated calls
  const transcriptFiredRef = useRef(false)

  const poll = useCallback(async () => {
    if (!recordingId) return
    try {
      const res = await api.get(`/audio/jobs/${recordingId}`)
      console.log('[JobPoller] Status:', res.data.status, 'Progress:', res.data.progress)
      setData(res.data)

      if (res.data.status === 'transcript_ready' && !transcriptFiredRef.current) {
        // Phase 1 complete — show transcript immediately, keep polling for AI
        console.log('[JobPoller] Transcript ready! Showing transcript, waiting for AI...')
        transcriptFiredRef.current = true
        onTranscriptReadyRef.current?.(res.data.result)
      } else if (res.data.status === 'done') {
        console.log('[JobPoller] Job complete! Calling onDone with result')
        if (intervalRef.current) clearInterval(intervalRef.current)
        intervalRef.current = null
        onDoneRef.current?.(res.data.result as ProcessingResult)
      } else if (res.data.status === 'error') {
        console.error('[JobPoller] Job error:', res.data.error)
        if (intervalRef.current) clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    } catch (err) {
      console.warn('[JobPoller] Poll request failed:', err)
      // silent retry
    }
  }, [recordingId]) // callbacks are accessed via refs — no dependency needed

  useEffect(() => {
    if (!recordingId) {
      console.log('[JobPoller] No recordingId, stopping polling')
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = null
      transcriptFiredRef.current = false
      setData(null)
      return
    }
    console.log('[JobPoller] Starting polling for:', recordingId)
    transcriptFiredRef.current = false
    poll()
    intervalRef.current = setInterval(poll, 2500)
    return () => {
      console.log('[JobPoller] Cleanup - stopping polling')
      if (intervalRef.current) clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [recordingId, poll])

  return data
}
