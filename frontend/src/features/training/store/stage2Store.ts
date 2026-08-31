import { create } from 'zustand';
import type {
  Stage2Variant,
  Stage2ValidationOutput,
  Stage2FeedbackCategory,
  Stage2HistoryEntry,
  Stage2Settings,
  Stage2OptimizationRunStatus,
  Stage2Group,
  ReferenceSource,
} from '../types/stage2Types';
import { DEFAULT_STAGE2_SETTINGS } from '../types/stage2Types';
import type { MeetingListItem } from '../types/training';

export type Stage2Tab = 'dataset' | 'training_loop' | 'variants' | 'history';

interface Stage2State {
  // Sub-tab navigation
  activeTab: Stage2Tab;
  setActiveTab: (tab: Stage2Tab) => void;

  // Settings
  settings: Stage2Settings;
  setSettings: (s: Stage2Settings) => void;
  settingsLoading: boolean;
  setSettingsLoading: (v: boolean) => void;
  settingsSaved: boolean;
  setSettingsSaved: (v: boolean) => void;

  // Meetings list
  meetings: MeetingListItem[];
  setMeetings: (m: MeetingListItem[]) => void;

  // Dataset tab state
  datasetMeetingId: string | null;
  setDatasetMeetingId: (id: string | null) => void;
  datasetMeetingFilename: string;
  setDatasetMeetingFilename: (f: string) => void;

  stage1Points: any[];
  setStage1Points: (pts: any[]) => void;
  stage1PointsLoading: boolean;
  setStage1PointsLoading: (v: boolean) => void;

  pointsPerGroup: number;
  setPointsPerGroup: (v: number) => void;
  contextRetrieval: boolean;
  setContextRetrieval: (v: boolean) => void;

  groups: Stage2Group[];
  setGroups: (g: Stage2Group[]) => void;
  groupsLoading: boolean;
  setGroupsLoading: (v: boolean) => void;

  hasStage2Edits: boolean;
  setHasStage2Edits: (v: boolean) => void;
  referenceSource: ReferenceSource;
  setReferenceSource: (s: ReferenceSource) => void;
  referencePoints: any[];
  setReferencePoints: (pts: any[]) => void;

  momFilename: string;
  setMomFilename: (f: string) => void;
  momExtractedText: string;
  setMomExtractedText: (t: string) => void;
  momPoints: any[];
  setMomPoints: (pts: any[]) => void;
  momUploading: boolean;
  setMomUploading: (v: boolean) => void;

  // Training loop state
  currentRunId: string | null;
  setCurrentRunId: (id: string | null) => void;
  currentRunStatus: Stage2OptimizationRunStatus | null;
  setCurrentRunStatus: (s: Stage2OptimizationRunStatus | null) => void;
  trainingLogs: string[];
  appendTrainingLog: (msg: string) => void;
  clearTrainingLogs: () => void;

  // Variants
  variants: Stage2Variant[];
  setVariants: (v: Stage2Variant[]) => void;
  variantsLoading: boolean;
  setVariantsLoading: (v: boolean) => void;

  // Model validation state
  hfModelPathValidation: { valid: boolean; reason: string; warnings: string[]; details?: any } | null;
  setHfModelPathValidation: (v: { valid: boolean; reason: string; warnings: string[]; details?: any } | null) => void;
  hfModelPathValidating: boolean;
  setHfModelPathValidating: (v: boolean) => void;

  // Active Variant
  activeVariantId: string;
  setActiveVariantId: (id: string) => void;

  // Validation workspace
  valMeetingId: string | null;
  setValMeetingId: (id: string | null) => void;
  valGroupIndex: number;
  setValGroupIndex: (idx: number) => void;
  valGroups: Stage2Group[];
  setValGroups: (g: Stage2Group[]) => void;
  valVariantId: string;
  setValVariantId: (id: string) => void;
  validationOutput: Stage2ValidationOutput | null;
  setValidationOutput: (o: Stage2ValidationOutput | null) => void;
  validationLoading: boolean;
  setValidationLoading: (v: boolean) => void;

  // Feedback form
  feedbackCategories: Stage2FeedbackCategory[];
  toggleFeedbackCategory: (cat: Stage2FeedbackCategory) => void;
  feedbackComment: string;
  setFeedbackComment: (c: string) => void;
  feedbackSubmitting: boolean;
  setFeedbackSubmitting: (v: boolean) => void;
  feedbackSubmitted: boolean;
  setFeedbackSubmitted: (v: boolean) => void;
  clearFeedback: () => void;

  // History
  history: Stage2HistoryEntry[];
  setHistory: (h: Stage2HistoryEntry[]) => void;
  historyLoading: boolean;
  setHistoryLoading: (v: boolean) => void;
}

export const useStage2Store = create<Stage2State>((set) => ({
  activeTab: 'dataset',
  setActiveTab: (tab) => set({ activeTab: tab }),

  settings: DEFAULT_STAGE2_SETTINGS,
  setSettings: (s) => set({ settings: s }),
  settingsLoading: false,
  setSettingsLoading: (v) => set({ settingsLoading: v }),
  settingsSaved: false,
  setSettingsSaved: (v) => set({ settingsSaved: v }),

  meetings: [],
  setMeetings: (m) => set({ meetings: m }),

  datasetMeetingId: null,
  setDatasetMeetingId: (id) =>
    set({
      datasetMeetingId: id,
      stage1Points: [],
      groups: [],
      referencePoints: [],
      referenceSource: 'none',
      momPoints: [],
      momFilename: '',
      momExtractedText: '',
    }),
  datasetMeetingFilename: '',
  setDatasetMeetingFilename: (f) => set({ datasetMeetingFilename: f }),

  stage1Points: [],
  setStage1Points: (pts) => set({ stage1Points: pts }),
  stage1PointsLoading: false,
  setStage1PointsLoading: (v) => set({ stage1PointsLoading: v }),

  pointsPerGroup: 5,
  setPointsPerGroup: (v) => set({ pointsPerGroup: v }),
  contextRetrieval: true,
  setContextRetrieval: (v) => set({ contextRetrieval: v }),

  groups: [],
  setGroups: (g) => set({ groups: g }),
  groupsLoading: false,
  setGroupsLoading: (v) => set({ groupsLoading: v }),

  hasStage2Edits: false,
  setHasStage2Edits: (v) => set({ hasStage2Edits: v }),
  referenceSource: 'none',
  setReferenceSource: (s) => set({ referenceSource: s }),
  referencePoints: [],
  setReferencePoints: (pts) => set({ referencePoints: pts }),

  momFilename: '',
  setMomFilename: (f) => set({ momFilename: f }),
  momExtractedText: '',
  setMomExtractedText: (t) => set({ momExtractedText: t }),
  momPoints: [],
  setMomPoints: (pts) => set({ momPoints: pts }),
  momUploading: false,
  setMomUploading: (v) => set({ momUploading: v }),

  currentRunId: null,
  setCurrentRunId: (id) => set({ currentRunId: id }),
  currentRunStatus: null,
  setCurrentRunStatus: (s) => set({ currentRunStatus: s }),
  trainingLogs: [],
  appendTrainingLog: (msg) =>
    set((s) => ({ trainingLogs: [...s.trainingLogs.slice(-99), msg] })),
  clearTrainingLogs: () => set({ trainingLogs: [] }),

  variants: [],
  setVariants: (v) => set({ variants: v }),
  variantsLoading: false,
  setVariantsLoading: (v) => set({ variantsLoading: v }),

  hfModelPathValidation: null,
  setHfModelPathValidation: (v) => set({ hfModelPathValidation: v }),
  hfModelPathValidating: false,
  setHfModelPathValidating: (v) => set({ hfModelPathValidating: v }),

  activeVariantId: 'default',
  setActiveVariantId: (id) => set({ activeVariantId: id }),

  valMeetingId: null,
  setValMeetingId: (id) => set({ valMeetingId: id, valGroups: [], valGroupIndex: 0 }),
  valGroupIndex: 0,
  setValGroupIndex: (idx) => set({ valGroupIndex: idx }),
  valGroups: [],
  setValGroups: (g) => set({ valGroups: g }),
  valVariantId: 'default',
  setValVariantId: (id) => set({ valVariantId: id }),
  validationOutput: null,
  setValidationOutput: (o) => set({ validationOutput: o }),
  validationLoading: false,
  setValidationLoading: (v) => set({ validationLoading: v }),

  feedbackCategories: [],
  toggleFeedbackCategory: (cat) =>
    set((s) => ({
      feedbackCategories: s.feedbackCategories.includes(cat)
        ? s.feedbackCategories.filter((c) => c !== cat)
        : [...s.feedbackCategories, cat],
    })),
  feedbackComment: '',
  setFeedbackComment: (c) => set({ feedbackComment: c }),
  feedbackSubmitting: false,
  setFeedbackSubmitting: (v) => set({ feedbackSubmitting: v }),
  feedbackSubmitted: false,
  setFeedbackSubmitted: (v) => set({ feedbackSubmitted: v }),
  clearFeedback: () =>
    set({ feedbackCategories: [], feedbackComment: '', feedbackSubmitted: false }),

  history: [],
  setHistory: (h) => set({ history: h }),
  historyLoading: false,
  setHistoryLoading: (v) => set({ historyLoading: v }),
}));
