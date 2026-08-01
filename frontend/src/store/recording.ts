import { create } from 'zustand'
import {
  recordingService,
  type RecordingState,
  type RecordingType,
  type RecordingServiceSnapshot,
} from '../services/recordingService'

export interface RecordingStoreState extends RecordingServiceSnapshot {
  startMic: (opts?: { meetingPrompt?: string; useVocabularyInPrompt?: boolean }) => Promise<void>
  startTab: (includeMic?: boolean) => Promise<void>
  pause: () => void
  resume: () => void
  stop: () => void
  reset: () => void
  cancel: () => void
  setIncludeMic: (val: boolean) => void
  clearCrashRecovery: () => void
}

export const useRecordingStore = create<RecordingStoreState>((set) => {
  // Subscribe to recordingService changes
  recordingService.subscribe((snapshot) => {
    set(snapshot)
  })

  return {
    ...recordingService.getSnapshot(),

    startMic: (opts) => recordingService.startMic(opts),
    startTab: (includeMic) => recordingService.startTab(includeMic),
    pause: () => recordingService.pause(),
    resume: () => recordingService.resume(),
    stop: () => recordingService.stop(),
    reset: () => recordingService.reset(),
    cancel: () => recordingService.cancel(),
    setIncludeMic: (val) => recordingService.setIncludeMic(val),
    clearCrashRecovery: () => recordingService.clearCrashRecovery(),
  }
})
