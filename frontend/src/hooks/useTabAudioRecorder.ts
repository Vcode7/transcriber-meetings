import { useRef } from 'react'
import { useRecordingStore } from '../store/recording'
import { recordingService } from '../services/recordingService'

export type TabRecordingState = 'idle' | 'recording' | 'paused' | 'stopped'

export function useTabAudioRecorder() {
  const store = useRecordingStore()

  const latestChunkRef = useRef<Blob | null>(null)
  latestChunkRef.current = store.latestChunk

  const isSupported = typeof navigator !== 'undefined'
    && !!navigator.mediaDevices
    && typeof navigator.mediaDevices.getDisplayMedia === 'function'

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0')
    const sec = (s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
  }

  return {
    state: store.recordingType === 'tab' || store.state === 'idle' || store.state === 'stopped' ? store.state : 'idle',
    duration: store.duration,
    formattedDuration: formatDuration(store.duration),
    audioBlob: store.audioBlob,
    audioUrl: store.audioUrl,
    analyser: store.analyser,
    error: store.error,
    tabLabel: store.tabLabel,
    includeMic: store.includeMic,
    setIncludeMic: (val: boolean) => store.setIncludeMic(val),
    isSupported,
    start: () => recordingService.startTab(store.includeMic),
    pause: () => recordingService.pause(),
    resume: () => recordingService.resume(),
    stop: () => recordingService.stop(),
    reset: () => recordingService.reset(),
    latestChunkRef,
  }
}
