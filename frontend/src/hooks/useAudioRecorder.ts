import { useEffect, useRef, useState, useCallback } from 'react'
import api from '../api/client'

export type RecordingState = 'idle' | 'recording' | 'stopped'

const CHUNK_INTERVAL_SEC = 600  // 10 minutes

export function useAudioRecorder(advancedOpts?: {
  meetingPrompt?: string
  useVocabularyInPrompt?: boolean
}) {
  const [state, setState] = useState<RecordingState>('idle')
  const [duration, setDuration] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const mimeTypeRef = useRef<string>('audio/webm')

  // Header chunk (EBML init segment) — must be prepended to every standalone blob
  const headerChunkRef = useRef<Blob | null>(null)

  // Index of the first blob that has NOT yet been sent as a background chunk
  const nextChunkBlobIndexRef = useRef<number>(0)

  // Submitted chunk IDs returned by the backend
  const chunkIdsRef = useRef<string[]>([])
  const chunkIndexRef = useRef<number>(0)     // 0-based counter of chunks sent
  const chunkStartTimeRef = useRef<number>(0) // seconds since recording start of current window

  // Duration at which the last background chunk was submitted
  const lastChunkSubmittedAtRef = useRef<number>(0)

  // Flag: is a chunk submission in flight? Prevent double-submission
  const submittingChunkRef = useRef<boolean>(false)

  /**
   * latestChunkRef — most recent 1-second blob for cross-talk detection.
   */
  const latestChunkRef = useRef<Blob | null>(null)

  const clearTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current)
  }

  // ── Submit accumulated blobs as a background chunk ──────────
  const submitChunk = useCallback(async (
    blobs: Blob[],
    chunkStartSec: number,
    chunkEndSec: number,
  ) => {
    if (submittingChunkRef.current || blobs.length === 0 || !headerChunkRef.current) return
    submittingChunkRef.current = true
    const currentIndex = chunkIndexRef.current
    chunkIndexRef.current += 1

    try {
      const chunkBlob = new Blob([headerChunkRef.current, ...blobs.slice(1)], { type: mimeTypeRef.current })
      const form = new FormData()
      form.append('file', chunkBlob, `chunk_${currentIndex}.webm`)
      form.append('chunk_index', String(currentIndex))
      form.append('chunk_start_sec', String(chunkStartSec))
      form.append('chunk_end_sec', String(chunkEndSec))
      if (advancedOpts?.meetingPrompt) form.append('meeting_prompt', advancedOpts.meetingPrompt)
      form.append('use_vocabulary', advancedOpts?.useVocabularyInPrompt ? 'true' : 'false')
      const res = await api.post('/audio/chunk', form)
      const { chunk_id } = res.data
      chunkIdsRef.current.push(chunk_id)
      console.log(`[Recorder] Background chunk ${currentIndex} submitted → chunk_id=${chunk_id}`)
    } catch (e) {
      console.error('[Recorder] Background chunk submission failed:', e)
    } finally {
      submittingChunkRef.current = false
    }
  }, [advancedOpts])

  const start = useCallback(async () => {
    setError(null)
    setAudioBlob(null)
    setAudioUrl(null)
    setDuration(0)
    chunksRef.current = []
    latestChunkRef.current = null
    headerChunkRef.current = null
    nextChunkBlobIndexRef.current = 0
    chunkIdsRef.current = []
    chunkIndexRef.current = 0
    chunkStartTimeRef.current = 0
    lastChunkSubmittedAtRef.current = 0

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Waveform analyser
      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      const src = ctx.createMediaStreamSource(stream)
      const ana = ctx.createAnalyser()
      ana.fftSize = 256
      src.connect(ana)
      setAnalyser(ana)

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      mimeTypeRef.current = mimeType

      const mr = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mr

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)

          if (headerChunkRef.current === null) {
            headerChunkRef.current = e.data
            latestChunkRef.current = new Blob([e.data], { type: mimeType })
          } else {
            latestChunkRef.current = new Blob([headerChunkRef.current, e.data], { type: mimeType })
          }
        }
      }

      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const url = URL.createObjectURL(blob)
        setAudioBlob(blob)
        setAudioUrl(url)
        setState('stopped')
      }

      mr.start(1000)  // timeslice = 1 second
      setState('recording')

      timerRef.current = setInterval(async () => {
        setDuration((d) => {
          const newDuration = d + 1

          // Auto-submit chunk every CHUNK_INTERVAL_SEC seconds
          if (
            newDuration >= CHUNK_INTERVAL_SEC &&
            newDuration - lastChunkSubmittedAtRef.current >= CHUNK_INTERVAL_SEC &&
            !submittingChunkRef.current &&
            headerChunkRef.current !== null
          ) {
            const chunkStart = lastChunkSubmittedAtRef.current
            const chunkEnd = newDuration
            lastChunkSubmittedAtRef.current = newDuration

            // Grab blobs accumulated since last chunk submission (index-based slice)
            const blobStart = nextChunkBlobIndexRef.current
            const blobsSnapshot = chunksRef.current.slice(blobStart)
            nextChunkBlobIndexRef.current = chunksRef.current.length

            // Submit asynchronously (fire-and-forget inside the timer)
            submitChunk(blobsSnapshot, chunkStart, chunkEnd).catch(() => {})
          }

          return newDuration
        })
      }, 1000)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Microphone access denied')
    }
  }, [submitChunk])

  const stop = useCallback(() => {
    clearTimer()
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    audioCtxRef.current?.close()
    setAnalyser(null)
  }, [])

  const reset = useCallback(() => {
    clearTimer()
    setState('idle')
    setDuration(0)
    setAudioBlob(null)
    latestChunkRef.current = null
    headerChunkRef.current = null
    nextChunkBlobIndexRef.current = 0
    chunkIdsRef.current = []
    chunkIndexRef.current = 0
    lastChunkSubmittedAtRef.current = 0
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setError(null)
  }, [audioUrl])

  useEffect(() => () => {
    clearTimer()
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const formatDuration = (s: number) => {
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0')
    const sec = (s % 60).toString().padStart(2, '0')
    return h > 0 ? `${h}:${m}:${sec}` : `${m}:${sec}`
  }

  return {
    state,
    duration,
    formattedDuration: formatDuration(duration),
    audioBlob,
    audioUrl,
    analyser,
    error,
    start,
    stop,
    reset,
    latestChunkRef,
    /** Chunk IDs submitted to backend during recording (populated for >10 min recordings) */
    chunkIdsRef,
    /** True if any background chunks have been submitted */
    get isChunked() { return chunkIdsRef.current.length > 0 },
  }
}

