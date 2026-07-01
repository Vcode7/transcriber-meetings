import { useEffect, useRef, useState, useCallback } from 'react'

export type RecordingState = 'idle' | 'recording' | 'stopped'

export function useAudioRecorder() {
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

  /**
   * latestChunkRef holds the most recent 1-second audio blob produced by the
   * MediaRecorder timeslice. Record.tsx reads it every second for overlap
   * detection without spinning up parallel recorders.
   */
  const latestChunkRef = useRef<Blob | null>(null)

  const clearTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current)
  }

  const start = useCallback(async () => {
    setError(null)
    setAudioBlob(null)
    setAudioUrl(null)
    setDuration(0)
    chunksRef.current = []
    latestChunkRef.current = null

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

      const mr = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mr

      // Accumulate all chunks for the final full-audio blob.
      // latestChunkRef holds a self-contained WebM blob for cross-talk detection.
      // WebM timeslice: chunk[0] = EBML header + init segment (mandatory prefix).
      // chunk[1..N] = raw Clusters — invalid without the header prepended.
      let headerChunk: Blob | null = null

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)

          if (headerChunk === null) {
            // First chunk — this IS the header; use it directly.
            headerChunk = e.data
            latestChunkRef.current = new Blob([e.data], { type: mimeType })
          } else {
            // Subsequent chunks — prepend the header so the blob is parseable.
            latestChunkRef.current = new Blob([headerChunk, e.data], { type: mimeType })
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

      // timeslice=1000ms → ondataavailable fires every 1 second
      mr.start(1000)
      setState('recording')

      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Microphone access denied')
    }
  }, [])

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
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setError(null)
  }, [audioUrl])

  useEffect(() => () => {
    clearTimer()
    streamRef.current?.getTracks().forEach((t) => t.stop())
  }, [])

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0')
    const sec = (s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
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
  }
}
