import { create } from 'zustand';
import type {
  TrainingStage, TrainingMethod, StageConfig, OllamaModel,
  TrainingJob, TrainingDataset, MeetingListItem, MeetingDataSummary,
  ValidationResult, HistoryEntry, TrainingArtifact,
} from '../types/training';
import { defaultStageConfig } from '../types/training';

export type TrainingTab = 'dataset' | 'stages' | 'method' | 'model' | 'config' | 'progress' | 'evaluation' | 'history' | 'edit_training';

interface TrainingState {
  // Tab
  activeTab: TrainingTab;
  setActiveTab: (tab: TrainingTab) => void;

  // Dataset source
  sourceType: 'meeting' | 'upload';
  setSourceType: (t: 'meeting' | 'upload') => void;
  inputSourceType: 'context' | 'meeting';
  setInputSourceType: (t: 'context' | 'meeting') => void;
  useEditedStage2: boolean;
  setUseEditedStage2: (val: boolean) => void;

  selectedMeetingId: string | null;
  setSelectedMeetingId: (id: string | null) => void;
  manualMom: string;
  setManualMom: (mom: string) => void;
  manualMomFilename: string;
  setManualMomFilename: (fn: string) => void;
  extractedMomPoints: string[];
  setExtractedMomPoints: (pts: string[]) => void;
  isExtractingMom: boolean;
  setIsExtractingMom: (v: boolean) => void;

  meetings: MeetingListItem[];
  setMeetings: (m: MeetingListItem[]) => void;
  meetingDataSummary: MeetingDataSummary | null;
  setMeetingDataSummary: (d: MeetingDataSummary | null) => void;

  // Upload
  uploadedTranscript: string;
  setUploadedTranscript: (t: string) => void;
  uploadedContext: string;
  setUploadedContext: (c: string) => void;
  uploadedAgenda: string;
  setUploadedAgenda: (a: string) => void;

  // Stage selection
  selectedStages: TrainingStage[];
  toggleStage: (stage: TrainingStage) => void;

  // Per-stage configs
  stageConfigs: Record<TrainingStage, StageConfig>;
  updateStageConfig: (stage: TrainingStage, update: Partial<StageConfig>) => void;

  // Ollama models
  ollamaModels: OllamaModel[];
  setOllamaModels: (models: OllamaModel[]) => void;
  modelsLoading: boolean;
  setModelsLoading: (v: boolean) => void;

  // Validation
  validationResults: Record<TrainingStage, ValidationResult | null>;
  setValidationResult: (stage: TrainingStage, result: ValidationResult | null) => void;

  // Dataset
  builtDatasetId: string | null;
  setBuiltDatasetId: (id: string | null) => void;
  datasetPreview: unknown[];
  setDatasetPreview: (samples: unknown[]) => void;
  datasetSamplesByStage: Record<string, number>;
  setDatasetSamplesByStage: (s: Record<string, number>) => void;
  datasetBuilding: boolean;
  setDatasetBuilding: (v: boolean) => void;

  // Active job
  activeJobId: string | null;
  setActiveJobId: (id: string | null) => void;
  activeJob: TrainingJob | null;
  setActiveJob: (job: TrainingJob | null) => void;

  // Jobs list
  jobs: TrainingJob[];
  setJobs: (jobs: TrainingJob[]) => void;

  // History
  history: HistoryEntry[];
  setHistory: (h: HistoryEntry[]) => void;
  historyArtifacts: TrainingArtifact[];
  setHistoryArtifacts: (a: TrainingArtifact[]) => void;

  // Reset for new run
  resetForNewRun: () => void;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  activeTab: 'dataset',
  setActiveTab: (tab) => set({ activeTab: tab }),

  sourceType: 'meeting',
  setSourceType: (t) => set({ sourceType: t }),
  inputSourceType: 'meeting',
  setInputSourceType: (t) => set({ inputSourceType: t, sourceType: t === 'meeting' ? 'meeting' : 'upload' }),
  useEditedStage2: false,
  setUseEditedStage2: (val) => set({ useEditedStage2: val }),

  selectedMeetingId: null,
  setSelectedMeetingId: (id) => set({ selectedMeetingId: id, meetingDataSummary: null }),
  manualMom: '',
  setManualMom: (mom) => set({ manualMom: mom }),
  manualMomFilename: '',
  setManualMomFilename: (fn) => set({ manualMomFilename: fn }),
  extractedMomPoints: [],
  setExtractedMomPoints: (pts) => set({ extractedMomPoints: pts }),
  isExtractingMom: false,
  setIsExtractingMom: (v) => set({ isExtractingMom: v }),

  meetings: [],
  setMeetings: (m) => set({ meetings: m }),
  meetingDataSummary: null,
  setMeetingDataSummary: (d) => set({ meetingDataSummary: d }),

  uploadedTranscript: '',
  setUploadedTranscript: (t) => set({ uploadedTranscript: t }),
  uploadedContext: '',
  setUploadedContext: (c) => set({ uploadedContext: c }),
  uploadedAgenda: '',
  setUploadedAgenda: (a) => set({ uploadedAgenda: a }),

  selectedStages: ['stage_1'],
  toggleStage: (stage) => set((s) => ({
    selectedStages: s.selectedStages.includes(stage)
      ? s.selectedStages.filter((x) => x !== stage)
      : [...s.selectedStages, stage],
  })),

  stageConfigs: {
    stage_1: defaultStageConfig('stage_1'),
    stage_2: defaultStageConfig('stage_2'),
    stage_3: defaultStageConfig('stage_3'),
  },
  updateStageConfig: (stage, update) => set((s) => ({
    stageConfigs: {
      ...s.stageConfigs,
      [stage]: { ...s.stageConfigs[stage], ...update },
    },
  })),

  ollamaModels: [],
  setOllamaModels: (models) => set({ ollamaModels: models }),
  modelsLoading: false,
  setModelsLoading: (v) => set({ modelsLoading: v }),

  validationResults: { stage_1: null, stage_2: null, stage_3: null },
  setValidationResult: (stage, result) => set((s) => ({
    validationResults: { ...s.validationResults, [stage]: result },
  })),

  builtDatasetId: null,
  setBuiltDatasetId: (id) => set({ builtDatasetId: id }),
  datasetPreview: [],
  setDatasetPreview: (samples) => set({ datasetPreview: samples }),
  datasetSamplesByStage: {},
  setDatasetSamplesByStage: (s) => set({ datasetSamplesByStage: s }),
  datasetBuilding: false,
  setDatasetBuilding: (v) => set({ datasetBuilding: v }),

  activeJobId: null,
  setActiveJobId: (id) => set({ activeJobId: id }),
  activeJob: null,
  setActiveJob: (job) => set({ activeJob: job }),

  jobs: [],
  setJobs: (jobs) => set({ jobs }),

  history: [],
  setHistory: (h) => set({ history: h }),
  historyArtifacts: [],
  setHistoryArtifacts: (a) => set({ historyArtifacts: a }),

  resetForNewRun: () => set({
    builtDatasetId: null,
    datasetPreview: [],
    datasetSamplesByStage: {},
    activeJobId: null,
    activeJob: null,
    validationResults: { stage_1: null, stage_2: null, stage_3: null },
  }),
}));
