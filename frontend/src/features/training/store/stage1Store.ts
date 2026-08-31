import { create } from 'zustand';
import type {
  TranscriptWindow,
  Stage1Variant,
  Stage1ValidationOutput,
  FeedbackCategory,
  Stage1HistoryEntry,
  Stage1Settings,
  OptimizationRunStatus,
} from '../types/stage1Types';
import { DEFAULT_STAGE1_SETTINGS } from '../types/stage1Types';
import type { MeetingListItem } from '../types/training';

export type Stage1Tab = 'dataset' | 'training_loop' | 'variants' | 'history';

interface Stage1State {
  // Sub-tab navigation
  activeTab: Stage1Tab;
  setActiveTab: (tab: Stage1Tab) => void;

  // Settings (shared with SettingsTab)
  settings: Stage1Settings;
  setSettings: (s: Stage1Settings) => void;
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
  windowSizeMinutes: number;
  setWindowSizeMinutes: (v: number) => void;
  windowOverlapSeconds: number;
  setWindowOverlapSeconds: (v: number) => void;
  generatedWindows: TranscriptWindow[];
  setGeneratedWindows: (w: TranscriptWindow[]) => void;
  windowsLoading: boolean;
  setWindowsLoading: (v: boolean) => void;
  toggleWindowSelected: (windowId: string) => void;
  setWindowRole: (windowId: string, role: 'train' | 'val' | undefined) => void;
  selectAllWindows: (selected: boolean) => void;
  datasetMeetingFilename: string;
  setDatasetMeetingFilename: (f: string) => void;

  // Training loop state
  currentRunId: string | null;
  setCurrentRunId: (id: string | null) => void;
  currentRunStatus: OptimizationRunStatus | null;
  setCurrentRunStatus: (s: OptimizationRunStatus | null) => void;
  trainingLogs: string[];
  appendTrainingLog: (msg: string) => void;
  clearTrainingLogs: () => void;

  // Variants
  variants: Stage1Variant[];
  setVariants: (v: Stage1Variant[]) => void;
  variantsLoading: boolean;
  setVariantsLoading: (v: boolean) => void;

  // Validation workspace
  valMeetingId: string | null;
  setValMeetingId: (id: string | null) => void;
  valWindowId: string | null;
  setValWindowId: (id: string | null) => void;
  valWindows: TranscriptWindow[];
  setValWindows: (w: TranscriptWindow[]) => void;
  valVariantId: string;
  setValVariantId: (id: string) => void;
  validationOutput: Stage1ValidationOutput | null;
  setValidationOutput: (o: Stage1ValidationOutput | null) => void;
  validationLoading: boolean;
  setValidationLoading: (v: boolean) => void;

  // Feedback form
  feedbackCategories: FeedbackCategory[];
  toggleFeedbackCategory: (cat: FeedbackCategory) => void;
  feedbackComment: string;
  setFeedbackComment: (c: string) => void;
  feedbackSubmitting: boolean;
  setFeedbackSubmitting: (v: boolean) => void;
  feedbackSubmitted: boolean;
  setFeedbackSubmitted: (v: boolean) => void;
  clearFeedback: () => void;

  // History
  history: Stage1HistoryEntry[];
  setHistory: (h: Stage1HistoryEntry[]) => void;
  historyLoading: boolean;
  setHistoryLoading: (v: boolean) => void;
}

export const useStage1Store = create<Stage1State>((set) => ({
  activeTab: 'dataset',
  setActiveTab: (tab) => set({ activeTab: tab }),

  settings: DEFAULT_STAGE1_SETTINGS,
  setSettings: (s) => set({ settings: s }),
  settingsLoading: false,
  setSettingsLoading: (v) => set({ settingsLoading: v }),
  settingsSaved: false,
  setSettingsSaved: (v) => set({ settingsSaved: v }),

  meetings: [],
  setMeetings: (m) => set({ meetings: m }),

  datasetMeetingId: null,
  setDatasetMeetingId: (id) => set({ datasetMeetingId: id, generatedWindows: [] }),
  windowSizeMinutes: 2,
  setWindowSizeMinutes: (v) => set({ windowSizeMinutes: v }),
  windowOverlapSeconds: 30,
  setWindowOverlapSeconds: (v) => set({ windowOverlapSeconds: v }),
  generatedWindows: [],
  setGeneratedWindows: (w) => set({ generatedWindows: w }),
  windowsLoading: false,
  setWindowsLoading: (v) => set({ windowsLoading: v }),
  datasetMeetingFilename: '',
  setDatasetMeetingFilename: (f) => set({ datasetMeetingFilename: f }),
  toggleWindowSelected: (windowId) =>
    set((s) => ({
      generatedWindows: s.generatedWindows.map((w) =>
        w.window_id === windowId ? { ...w, selected: !w.selected } : w
      ),
    })),
  setWindowRole: (windowId, role) =>
    set((s) => ({
      generatedWindows: s.generatedWindows.map((w) =>
        w.window_id === windowId ? { ...w, role } : w
      ),
    })),
  selectAllWindows: (selected) =>
    set((s) => ({
      generatedWindows: s.generatedWindows.map((w) => ({ ...w, selected })),
    })),

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

  valMeetingId: null,
  setValMeetingId: (id) => set({ valMeetingId: id, valWindows: [], valWindowId: null }),
  valWindowId: null,
  setValWindowId: (id) => set({ valWindowId: id }),
  valWindows: [],
  setValWindows: (w) => set({ valWindows: w }),
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
  clearFeedback: () => set({ feedbackCategories: [], feedbackComment: '', feedbackSubmitted: false }),

  history: [],
  setHistory: (h) => set({ history: h }),
  historyLoading: false,
  setHistoryLoading: (v) => set({ historyLoading: v }),
}));
