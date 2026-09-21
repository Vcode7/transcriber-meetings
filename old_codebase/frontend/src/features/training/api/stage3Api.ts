import api from '../../../api/client';
import type {
  Stage3Variant,
  Stage3Settings,
  Stage3HistoryEntry,
  Stage3Batch,
  Stage3Agenda,
  Stage3ValidationOutput,
} from '../types/training';

const BASE = '/api/training/stage3';


// ─── Meeting Data & Agenda ───────────────────────────────────────────────────

export const fetchStage3MeetingData = async (
  meetingId: string
): Promise<{
  meeting_id: string;
  filename: string;
  stage2_points: any[];
  agendas: Stage3Agenda[];
  has_stage2: boolean;
  has_agendas: boolean;
  ground_truth_mappings: Record<string, string>;
}> => {
  const res = await api.get(`${BASE}/meetings/${meetingId}/data`);
  return res.data;
};

export const extractStage3File = async (
  file: File
): Promise<{ filename: string; extracted_text: string; char_count: number }> => {
  const form = new FormData();
  form.append('file', file);
  const res = await api.post(`${BASE}/extract-file`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const generateStage3Agendas = async (params: {
  meeting_id: string;
  agenda_text: string;
  previous_mom_texts?: string[];
  use_global_context?: boolean;
  global_context_top_k?: number;
  meeting_context_top_k?: number;
}): Promise<{
  meeting_id: string;
  agendas: Stage3Agenda[];
  expanded_agendas: any[];
  count: number;
}> => {
  const res = await api.post(`${BASE}/generate-agendas`, params);
  return res.data;
};

export const uploadStage3Agenda = async (
  meetingId: string,
  file: File
): Promise<{
  meeting_id: string;
  agendas: Stage3Agenda[];
  agenda_text: string;
  count: number;
}> => {
  const form = new FormData();
  form.append('meeting_id', meetingId);
  form.append('file', file);
  const res = await api.post(`${BASE}/upload-agenda`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

// ─── Batch Preview ───────────────────────────────────────────────────────────

export const previewStage3Batches = async (params: {
  meeting_id: string;
  batch_size: number;
}): Promise<{
  meeting_id: string;
  batch_size: number;
  total_points: number;
  total_batches: number;
  agendas: Stage3Agenda[];
  batches: Stage3Batch[];
}> => {
  const res = await api.post(`${BASE}/preview-batches`, params);
  return res.data;
};

// ─── Optimization ────────────────────────────────────────────────────────────

export const startStage3Optimization = async (params: {
  meeting_id: string;
  batches: Stage3Batch[];
  agendas: Stage3Agenda[];
  settings?: Partial<Stage3Settings>;
  parent_variant_id?: string;
}): Promise<{ run_id: string; status: string }> => {
  const res = await api.post(`${BASE}/optimize`, params);
  return res.data;
};

export const getStage3RunStatus = async (
  runId: string
): Promise<{
  run_id: string;
  status: string;
  progress: number;
  message: string;
  variant_id?: string | null;
  error?: string | null;
}> => {
  const res = await api.get(`${BASE}/runs/${runId}`);
  return res.data;
};

// ─── Validation ───────────────────────────────────────────────────────────────

export const runStage3Validation = async (params: {
  variant_id: string;
  meeting_id: string;
  batch_index: number;
  batch_points: any[];
  agendas: Stage3Agenda[];
}): Promise<Stage3ValidationOutput> => {
  const res = await api.post(`${BASE}/validate`, params);
  return res.data;
};

// ─── Feedback ─────────────────────────────────────────────────────────────────

export const submitStage3Feedback = async (params: {
  meeting_id: string;
  point_id: string;
  point_text: string;
  original_agenda_id: string;
  correct_agenda_id: string;
  reason?: string;
}): Promise<{ feedback_id: string; saved: boolean }> => {
  const res = await api.post(`${BASE}/feedback`, params);
  return res.data;
};

export const submitStage3FeedbackAndRetrain = async (params: {
  feedback: any;
  parent_variant_id: string;
  meeting_id: string;
  batches: Stage3Batch[];
  agendas: Stage3Agenda[];
  settings?: Partial<Stage3Settings>;
}): Promise<{ run_id: string; feedback_id: string; status: string }> => {
  const res = await api.post(`${BASE}/retrain`, params);
  return res.data;
};

// ─── Variants ─────────────────────────────────────────────────────────────────

export const listStage3Variants = async (): Promise<{
  variants: Stage3Variant[];
  count: number;
}> => {
  const res = await api.get(`${BASE}/variants`);
  return res.data;
};

export const getStage3Variant = async (variantId: string): Promise<Stage3Variant> => {
  const res = await api.get(`${BASE}/variants/${variantId}`);
  return res.data;
};

// ─── History ──────────────────────────────────────────────────────────────────

export const listStage3History = async (): Promise<{
  history: Stage3HistoryEntry[];
  count: number;
}> => {
  const res = await api.get(`${BASE}/history`);
  return res.data;
};

// ─── Settings ─────────────────────────────────────────────────────────────────

export const loadStage3Settings = async (): Promise<Stage3Settings> => {
  const res = await api.get(`${BASE}/settings`);
  return res.data;
};

export const saveStage3Settings = async (
  settings: Stage3Settings
): Promise<{ saved: boolean; settings: Stage3Settings }> => {
  const res = await api.post(`${BASE}/settings`, { settings });
  return res.data;
};
