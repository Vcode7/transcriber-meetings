import { create } from 'zustand';
import type {
  MeetingDataSummary,
  Stage3Variant,
  Stage3Settings,
  Stage3HistoryEntry,
  Stage3Batch,
  Stage3Agenda,
  Stage3ValidationOutput,
} from '../types/training';

export type Stage3Tab = 'dataset' | 'training_loop' | 'variants' | 'history';

interface Stage3State {
  // Navigation
  activeTab: Stage3Tab;
  setActiveTab: (tab: Stage3Tab) => void;

  // Meetings
  meetings: MeetingDataSummary[];
  setMeetings: (m: MeetingDataSummary[]) => void;

  // Dataset selection & inputs
  datasetMeetingId: string | null;
  setDatasetMeetingId: (id: string | null) => void;
  datasetMeetingFilename: string;
  setDatasetMeetingFilename: (name: string) => void;
  stage2Points: any[];
  setStage2Points: (pts: any[]) => void;
  agendas: Stage3Agenda[];
  setAgendas: (ag: Stage3Agenda[]) => void;
  hasStage2: boolean;
  setHasStage2: (v: boolean) => void;
  hasAgendas: boolean;
  setHasAgendas: (v: boolean) => void;
  groundTruthMappings: Record<string, string>;
  setGroundTruthMappings: (m: Record<string, string>) => void;
  agendaUploading: boolean;
  setAgendaUploading: (v: boolean) => void;

  // Batching
  pointsPerBatch: number;
  setPointsPerBatch: (n: number) => void;
  batches: Stage3Batch[];
  setBatches: (b: Stage3Batch[]) => void;
  batchesLoading: boolean;
  setBatchesLoading: (v: boolean) => void;

  // Settings
  settings: Stage3Settings;
  setSettings: (s: Stage3Settings) => void;
  updateSettings: (partial: Partial<Stage3Settings>) => void;

  // Training loop
  currentRunId: string | null;
  setCurrentRunId: (id: string | null) => void;
  currentRunStatus: {
    run_id: string;
    status: string;
    progress: number;
    message: string;
    variant_id?: string | null;
    error?: string | null;
  } | null;
  setCurrentRunStatus: (s: any) => void;
  trainingLogs: string[];
  appendTrainingLog: (msg: string) => void;
  clearTrainingLogs: () => void;

  // Validation workspace
  valMeetingId: string | null;
  setValMeetingId: (id: string | null) => void;
  valBatchIndex: number;
  setValBatchIndex: (idx: number) => void;
  valVariantId: string;
  setValVariantId: (id: string) => void;
  validationOutput: Stage3ValidationOutput | null;
  setValidationOutput: (o: Stage3ValidationOutput | null) => void;
  validationLoading: boolean;
  setValidationLoading: (v: boolean) => void;

  // Feedback
  reassigningPointId: string | null;
  setReassigningPointId: (id: string | null) => void;
  feedbackSubmitting: boolean;
  setFeedbackSubmitting: (v: boolean) => void;
  feedbackSubmitted: boolean;
  setFeedbackSubmitted: (v: boolean) => void;

  // Variants
  variants: Stage3Variant[];
  setVariants: (v: Stage3Variant[]) => void;
  variantsLoading: boolean;
  setVariantsLoading: (v: boolean) => void;

  // History
  history: Stage3HistoryEntry[];
  setHistory: (h: Stage3HistoryEntry[]) => void;
  historyLoading: boolean;
  setHistoryLoading: (v: boolean) => void;
}

const DEFAULT_SETTINGS: Stage3Settings = {
  model_name: '',
  optimizer: 'BootstrapFewShot',
  num_trials: 10,
  eval_split: 0.2,
  bootstrap_examples: 3,
  max_demonstrations: 4,
  temperature: 0.0,
  max_tokens: 2048,
  points_per_batch: 15,
};

export const useStage3Store = create<Stage3State>((set) => ({
  activeTab: 'dataset',
  setActiveTab: (tab) => set({ activeTab: tab }),

  meetings: [],
  setMeetings: (m) => set({ meetings: m }),

  datasetMeetingId: null,
  setDatasetMeetingId: (id) => set({ datasetMeetingId: id }),
  datasetMeetingFilename: '',
  setDatasetMeetingFilename: (name) => set({ datasetMeetingFilename: name }),
  stage2Points: [],
  setStage2Points: (pts) => set({ stage2Points: pts }),
  agendas: [],
  setAgendas: (ag) => set({ agendas: ag }),
  hasStage2: false,
  setHasStage2: (v) => set({ hasStage2: v }),
  hasAgendas: false,
  setHasAgendas: (v) => set({ hasAgendas: v }),
  groundTruthMappings: {},
  setGroundTruthMappings: (m) => set({ groundTruthMappings: m }),
  agendaUploading: false,
  setAgendaUploading: (v) => set({ agendaUploading: v }),

  pointsPerBatch: 15,
  setPointsPerBatch: (n) => set({ pointsPerBatch: n }),
  batches: [],
  setBatches: (b) => set({ batches: b }),
  batchesLoading: false,
  setBatchesLoading: (v) => set({ batchesLoading: v }),

  settings: DEFAULT_SETTINGS,
  setSettings: (s) => set({ settings: s }),
  updateSettings: (partial) => set((state) => ({ settings: { ...state.settings, ...partial } })),

  currentRunId: null,
  setCurrentRunId: (id) => set({ currentRunId: id }),
  currentRunStatus: null,
  setCurrentRunStatus: (s) => set({ currentRunStatus: s }),
  trainingLogs: [],
  appendTrainingLog: (msg) => set((s) => ({ trainingLogs: [...s.trainingLogs, `[${new Date().toLocaleTimeString()}] ${msg}`] })),
  clearTrainingLogs: () => set({ trainingLogs: [] }),

  valMeetingId: null,
  setValMeetingId: (id) => set({ valMeetingId: id }),
  valBatchIndex: 0,
  setValBatchIndex: (idx) => set({ valBatchIndex: idx }),
  valVariantId: 'default',
  setValVariantId: (id) => set({ valVariantId: id }),
  validationOutput: null,
  setValidationOutput: (o) => set({ validationOutput: o }),
  validationLoading: false,
  setValidationLoading: (v) => set({ validationLoading: v }),

  reassigningPointId: null,
  setReassigningPointId: (id) => set({ reassigningPointId: id }),
  feedbackSubmitting: false,
  setFeedbackSubmitting: (v) => set({ feedbackSubmitting: v }),
  feedbackSubmitted: false,
  setFeedbackSubmitted: (v) => set({ feedbackSubmitted: v }),

  variants: [],
  setVariants: (v) => set({ variants: v }),
  variantsLoading: false,
  setVariantsLoading: (v) => set({ variantsLoading: v }),

  history: [],
  setHistory: (h) => set({ history: h }),
  historyLoading: false,
  setHistoryLoading: (v) => set({ historyLoading: v }),
}));
