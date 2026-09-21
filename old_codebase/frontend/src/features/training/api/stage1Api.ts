import api from '../../../api/client';
import type {
  TranscriptWindow,
  Stage1Variant,
  Stage1ValidationOutput,
  FeedbackEntry,
  Stage1HistoryEntry,
  Stage1Settings,
  OptimizationRunStatus,
} from '../types/stage1Types';

const BASE = '/api/training/stage1';

// ── Windows ───────────────────────────────────────────────────────────────────

export interface PrepareWindowsResponse {
  meeting_id: string;
  filename: string;
  windows: TranscriptWindow[];
  total_windows: number;
  window_size_seconds: number;
  overlap_seconds: number;
}

export const prepareTranscriptWindows = async (
  meetingId: string,
  windowSizeSeconds: number,
  overlapSeconds: number,
): Promise<PrepareWindowsResponse> => {
  const res = await api.post(`${BASE}/windows`, {
    meeting_id: meetingId,
    window_size_seconds: windowSizeSeconds,
    overlap_seconds: overlapSeconds,
  });
  return res.data;
};

// ── Optimization ──────────────────────────────────────────────────────────────

export const startStage1Optimization = async (params: {
  meeting_id: string;
  windows: TranscriptWindow[];
  settings?: Partial<Stage1Settings>;
  parent_variant_id?: string;
  feedback_examples?: unknown[];
}): Promise<{ run_id: string; status: string }> => {
  const res = await api.post(`${BASE}/optimize`, params);
  return res.data;
};

export const getOptimizationRunStatus = async (
  runId: string,
): Promise<OptimizationRunStatus> => {
  const res = await api.get(`${BASE}/runs/${runId}`);
  return res.data;
};

// ── Validation ────────────────────────────────────────────────────────────────

export const runStage1Validation = async (params: {
  variant_id: string;
  meeting_id: string;
  window_id?: string;
  transcript_text: string;
  context_summary?: string;
  agenda_summary?: string;
}): Promise<Stage1ValidationOutput> => {
  const res = await api.post(`${BASE}/validate`, params);
  return res.data;
};

// ── Feedback ──────────────────────────────────────────────────────────────────

export const submitFeedback = async (
  feedback: FeedbackEntry,
): Promise<{ feedback_id: string; saved: boolean }> => {
  const res = await api.post(`${BASE}/feedback`, feedback);
  return res.data;
};

export const submitFeedbackAndRetrain = async (params: {
  feedback: FeedbackEntry;
  parent_variant_id: string;
  meeting_id: string;
  windows: TranscriptWindow[];
  settings?: Partial<Stage1Settings>;
}): Promise<{ run_id: string; feedback_id: string; status: string }> => {
  const res = await api.post(`${BASE}/retrain`, params);
  return res.data;
};

// ── Variants ──────────────────────────────────────────────────────────────────

export const listStage1Variants = async (): Promise<Stage1Variant[]> => {
  const res = await api.get(`${BASE}/variants`);
  return res.data.variants;
};

export const getStage1Variant = async (variantId: string): Promise<Stage1Variant> => {
  const res = await api.get(`${BASE}/variants/${variantId}`);
  return res.data;
};

// ── History ───────────────────────────────────────────────────────────────────

export const listStage1History = async (): Promise<Stage1HistoryEntry[]> => {
  const res = await api.get(`${BASE}/history`);
  return res.data.history;
};

// ── Settings ──────────────────────────────────────────────────────────────────

export const loadStage1Settings = async (): Promise<Stage1Settings> => {
  const res = await api.get(`${BASE}/settings`);
  return res.data;
};

export const saveStage1Settings = async (
  settings: Stage1Settings,
): Promise<{ saved: boolean }> => {
  const res = await api.post(`${BASE}/settings`, { settings });
  return res.data;
};
