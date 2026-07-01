import { useEffect, useRef, useState, useCallback } from 'react'

export type TabRecordingState = 'idle' | 'recording' | 'stopped'

/**
 * Hook that captures audio from a browser tab using getDisplayMedia.
 * Optionally mixes in the user's microphone for two-sided meeting capture.
 *
 * Same output contract as useAudioRecorder so it slots in identically
 * with Record-style pages (waveform, latestChunkRef for overlap, etc.).
 */
export function useTabAudioRecorder() {
  const [state, setState] = useState<TabRecordingState>('idle')
  const [duration, setDuration] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tabLabel, setTabLabel] = useState<string | null>(null)
  const [includeMic, setIncludeMic] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const displayStreamRef = useRef<MediaStream | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const destRef = useRef<MediaStreamAudioDestinationNode | null>(null)

  /**
   * latestChunkRef — same pattern as useAudioRecorder.
   * Holds the most recent 1-second self-contained WebM blob for overlap detection.
   */
  const latestChunkRef = useRef<Blob | null>(null)

  /** Check browser support for getDisplayMedia with audio */
  const isSupported = typeof navigator !== 'undefined'
    && !!navigator.mediaDevices
    && typeof navigator.mediaDevices.getDisplayMedia === 'function'

  const clearTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current)
  }

  const stop = useCallback(() => {
    clearTimer()
    mediaRecorderRef.current?.stop()
    // Stop all media tracks
    displayStreamRef.current?.getTracks().forEach((t) => t.stop())
    micStreamRef.current?.getTracks().forEach((t) => t.stop())
    audioCtxRef.current?.close()
    setAnalyser(null)
  }, [])

  const start = useCallback(async () => {
    setError(null)
    setAudioBlob(null)
    setAudioUrl(null)
    setDuration(0)
    setTabLabel(null)
    chunksRef.current = []
    latestChunkRef.current = null

    if (!isSupported) {
      setError('Tab audio capture is not supported in this browser. Please use Chrome, Edge, or Brave.')
      return
    }

    try {
      // ── 1. Capture tab/screen audio ─────────────────────────
      // getDisplayMedia requires video:true for the picker to appear,
      // but we only care about the audio track.
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,    // required for the system picker dialog
        audio: true,    // this is what we actually want
      } as DisplayMediaStreamOptions)

      displayStreamRef.current = displayStream

      // Check that we actually got an audio track
      const audioTracks = displayStream.getAudioTracks()
      if (audioTracks.length === 0) {
        // User didn't check "Share tab audio" in the dialog
        displayStream.getTracks().forEach((t) => t.stop())
        setError('No audio track detected. Please check "Share tab audio" in the browser dialog and try again.')
        return
      }

      // Stop the video track immediately — we don't need it
      displayStream.getVideoTracks().forEach((t) => t.stop())

      // Get the tab label from the audio track
      const trackLabel = audioTracks[0].label
      setTabLabel(trackLabel || 'Browser Tab')

      // Listen for the user clicking "Stop sharing" in the browser chrome
      audioTracks[0].onended = () => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stop()
        }
      }

      // ── 2. Set up AudioContext for mixing + analysis ────────
      const ctx = new AudioContext()
      audioCtxRef.current = ctx

      const dest = ctx.createMediaStreamDestination()
      destRef.current = dest

      // Tab audio → destination
      const tabSource = ctx.createMediaStreamSource(
        new MediaStream(audioTracks)
      )
      tabSource.connect(dest)

      // ── 3. Optionally mix in microphone ─────────────────────
      if (includeMic) {
        try {
          const micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
          micStreamRef.current = micStream
          const micSource = ctx.createMediaStreamSource(micStream)
          micSource.connect(dest)
        } catch (micErr) {
          console.warn('[TabAudio] Microphone access denied, continuing with tab audio only:', micErr)
          // Non-fatal — tab audio still works
        }
      }

      // ── 4. Waveform analyser (on the mixed output) ─────────
      const ana = ctx.createAnalyser()
      ana.fftSize = 256
      // Also connect tab source to analyser for visualization
      tabSource.connect(ana)
      setAnalyser(ana)

      // ── 5. Create MediaRecorder on the mixed stream ────────
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const mr = new MediaRecorder(dest.stream, { mimeType })
      mediaRecorderRef.current = mr

      // Same header-prepending pattern as useAudioRecorder
      let headerChunk: Blob | null = null

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)

          if (headerChunk === null) {
            headerChunk = e.data
            latestChunkRef.current = new Blob([e.data], { type: mimeType })
          } else {
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

      // timeslice=1000ms — same as useAudioRecorder for overlap detection compat
      mr.start(1000)
      setState('recording')

      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000)
    } catch (e: unknown) {
      // User cancelled the dialog or permission was denied
      if (e instanceof DOMException && e.name === 'NotAllowedError') {
        setError('Tab sharing was cancelled. Click "Share Tab Audio" to try again.')
      } else {
        setError(e instanceof Error ? e.message : 'Failed to capture tab audio.')
      }
    }
  }, [includeMic, isSupported, stop])

  const reset = useCallback(() => {
    clearTimer()
    setState('idle')
    setDuration(0)
    setAudioBlob(null)
    setTabLabel(null)
    latestChunkRef.current = null
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setError(null)
  }, [audioUrl])

  // Cleanup on unmount
  useEffect(() => () => {
    clearTimer()
    displayStreamRef.current?.getTracks().forEach((t) => t.stop())
    micStreamRef.current?.getTracks().forEach((t) => t.stop())
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
    tabLabel,
    includeMic,
    setIncludeMic,
    isSupported,
    start,
    stop,
    reset,
    latestChunkRef,
  }
}
