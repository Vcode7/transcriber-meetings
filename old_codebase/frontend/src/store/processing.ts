import { create } from 'zustand'

export type ProcessingSource = 'record' | 'upload' | 'tab-audio'

export type ProcessingStage =
  | 'uploading'
  | 'queued'
  | 'transcribing'
  | 'diarizing'
  | 'identifying_speakers'
  | 'generating_insights'
  | 'generating_mom'
  | 'done'
  | null

interface ProcessingState {
  isProcessing: boolean
  source: ProcessingSource | null
  stage: ProcessingStage
  startedAt: number | null

  setProcessing: (source: ProcessingSource, stage?: ProcessingStage, startedAt?: number) => void
  updateStage: (stage: ProcessingStage) => void
  clearProcessing: () => void
}

export const useProcessingStore = create<ProcessingState>((set) => ({
  isProcessing: false,
  source: null,
  stage: null,
  startedAt: null,

  setProcessing: (source, stage = 'uploading', startedAt) =>
    set({ isProcessing: true, source, stage, startedAt: startedAt ?? Date.now() }),

  updateStage: (stage) => set({ stage }),

  clearProcessing: () =>
    set({ isProcessing: false, source: null, stage: null, startedAt: null }),
}))
