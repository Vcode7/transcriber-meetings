/**
 * Stage 2 Training API client.
 * All calls go to /api/training/stage2/... via the central axios client
 * which attaches authentication tokens and handles silent refresh.
 */

import api from '../../../api/client';
import type {
  Stage2Variant,
  Stage2ValidationOutput,
  Stage2HistoryEntry,
  Stage2Settings,
  Stage2OptimizationRunStatus,
} from '../types/stage2Types';

const BASE = '/api/training/stage2';

// ─── Example Points (style-guide-only) ───────────────────────────────────────

export const loadExamplePoints = async (): Promise<{ points: string[]; count: number }> => {
  const res = await api.get(`${BASE}/example-points`);
  return res.data;
};

export const saveExamplePoints = async (
  points: string[],
): Promise<{ saved: boolean; count: number }> => {
  const res = await api.post(`${BASE}/example-points`, { points });
  return res.data;
};

// ─── Meetings & Stage 1 points ────────────────────────────────────────────────────

export const fetchStage1PointsForMeeting = async (meetingId: string): Promise<{
  meeting_id: string;
  stage1_points: any[];
  stage1_point_count: number;
  has_stage2: boolean;
  has_stage2_edits: boolean;
  reference_source: string;
  reference_points: any[];
}> => {
  const res = await api.get(`${BASE}/meetings/${meetingId}/stage1-points`);
  return res.data;
};

export const previewStage2Groups = async (params: {
  meeting_id: string;
  points_per_group: number;
  context_retrieval: boolean;
}): Promise<{
  meeting_id: string;
  total_points: number;
  total_groups: number;
  points_per_group: number;
  context_retrieval: boolean;
  groups: any[];
}> => {
  const res = await api.post(`${BASE}/preview-groups`, params);
  return res.data;
};

// ─── MoM extraction ───────────────────────────────────────────────────────────

export const uploadMomForStage2 = async (file: File): Promise<{
  filename: string;
  extracted_text: string;
  points: any[];
  point_count: number;
}> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post(`${BASE}/extract-mom`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

// ─── Optimization ─────────────────────────────────────────────────────────────

export const startStage2Optimization = async (params: {
  meeting_id: string;
  groups: any[];
  reference_points: any[];
  settings?: Partial<Stage2Settings>;
  parent_variant_id?: string;
  feedback_examples?: any[];
  example_points?: string[];
}): Promise<{ run_id: string; status: string }> => {
  const res = await api.post(`${BASE}/optimize`, params);
  return res.data;
};

export const getStage2RunStatus = async (
  runId: string,
): Promise<Stage2OptimizationRunStatus> => {
  const res = await api.get(`${BASE}/runs/${runId}`);
  return res.data;
};

// ─── Validation ───────────────────────────────────────────────────────────────

export const runStage2Validation = async (params: {
  variant_id: string;
  meeting_id: string;
  group_index?: number;
  stage1_points: any[];
  global_context?: string;
  meeting_context?: string;
  reference_points?: any[];
  example_points?: string[];
}): Promise<Stage2ValidationOutput> => {
  const res = await api.post(`${BASE}/validate`, params);
  return res.data;
};

// ─── Feedback ─────────────────────────────────────────────────────────────────

export const submitStage2Feedback = async (params: {
  meeting_id: string;
  group_index: number;
  variant_id: string;
  stage1_points_input: string;
  model_output: string;
  corrected_output?: string;
  categories: string[];
  comment?: string;
  global_context?: string;
  meeting_context?: string;
}): Promise<{ feedback_id: string; saved: boolean }> => {
  const res = await api.post(`${BASE}/feedback`, params);
  return res.data;
};

export const submitStage2FeedbackAndRetrain = async (params: {
  feedback: any;
  parent_variant_id: string;
  meeting_id: string;
  groups: any[];
  reference_points: any[];
  settings?: Partial<Stage2Settings>;
}): Promise<{ run_id: string; feedback_id: string; status: string }> => {
  const res = await api.post(`${BASE}/retrain`, params);
  return res.data;
};

// ─── Model Path Validation ───────────────────────────────────────────────────

export const validateStage2ModelPath = async (params: {
  hf_model_path: string;
  method?: string;
}): Promise<{
  valid: boolean;
  reason: string;
  warnings: string[];
  details?: {
    path?: string;
    model_type?: string;
    weights_count?: number;
    has_tokenizer?: boolean;
  };
}> => {
  const res = await api.post(`${BASE}/validate-model`, params);
  return res.data;
};

// ─── Variants ─────────────────────────────────────────────────────────────────

export const listStage2Variants = async (): Promise<{ variants: Stage2Variant[]; count: number }> => {
  const res = await api.get(`${BASE}/variants`);
  return res.data;
};

export const getStage2Variant = async (variantId: string): Promise<Stage2Variant> => {
  const res = await api.get(`${BASE}/variants/${variantId}`);
  return res.data;
};

export const activateStage2Variant = async (variantId: string): Promise<{ activated: boolean; variant_id: string }> => {
  const res = await api.post(`${BASE}/variants/${variantId}/activate`);
  return res.data;
};

export const getActiveStage2Variant = async (): Promise<{ active_variant: Stage2Variant }> => {
  const res = await api.get(`${BASE}/variants/active`);
  return res.data;
};

// ─── History ──────────────────────────────────────────────────────────────────

export const listStage2History = async (): Promise<{ history: Stage2HistoryEntry[]; count: number }> => {
  const res = await api.get(`${BASE}/history`);
  return res.data;
};

// ─── Settings ─────────────────────────────────────────────────────────────────

export const loadStage2Settings = async (): Promise<Stage2Settings> => {
  const res = await api.get(`${BASE}/settings`);
  return res.data;
};

export const saveStage2Settings = async (
  settings: Stage2Settings,
): Promise<{ saved: boolean; settings: Stage2Settings }> => {
  const res = await api.post(`${BASE}/settings`, { settings });
  return res.data;
};
