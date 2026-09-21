import { useMemo, useRef } from 'react'
import { useRecordingStore } from '../store/recording'
import { recordingService } from '../services/recordingService'

export type RecordingState = 'idle' | 'recording' | 'paused' | 'stopped'

export function useAudioRecorder(advancedOpts?: {
  meetingPrompt?: string
  useVocabularyInPrompt?: boolean
}) {
  const store = useRecordingStore()

  // Mutable refs for latestChunk and chunkIds for cross-talk detection & chunk compatibility
  const latestChunkRef = useRef<Blob | null>(null)
  latestChunkRef.current = store.latestChunk

  const chunkIdsRef = useRef<string[]>([])
  chunkIdsRef.current = store.chunkIds

  const formatDuration = (s: number) => {
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0')
    const sec = (s % 60).toString().padStart(2, '0')
    return h > 0 ? `${h}:${m}:${sec}` : `${m}:${sec}`
  }

  return {
    state: store.recordingType === 'mic' || store.state === 'idle' || store.state === 'stopped' ? store.state : 'idle',
    duration: store.duration,
    formattedDuration: formatDuration(store.duration),
    audioBlob: store.audioBlob,
    audioUrl: store.audioUrl,
    analyser: store.analyser,
    error: store.error,
    start: () => recordingService.startMic(advancedOpts),
    pause: () => recordingService.pause(),
    resume: () => recordingService.resume(),
    stop: () => recordingService.stop(),
    reset: () => recordingService.reset(),
    latestChunkRef,
    chunkIdsRef,
    get isChunked() {
      return store.isChunked
    },
  }
}
