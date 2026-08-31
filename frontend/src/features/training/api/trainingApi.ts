import api from '../../../api/client';
import type {
  OllamaModel, TrainingStage, TrainingMethod, StageConfig,
  TrainingJob, TrainingDataset, TrainingArtifact, ValidationResult,
  MeetingListItem, MeetingDataSummary, HistoryEntry,
} from '../types/training';

const BASE = '/api/training';

// Models
export const fetchOllamaModels = async (): Promise<OllamaModel[]> => {
  const res = await api.get(`${BASE}/models`);
  return res.data.models;
};

// Validation
export const validateConfig = async (params: {
  model_name: string;
  method: TrainingMethod;
  stage: TrainingStage;
  hf_model_path?: string;
}): Promise<ValidationResult> => {
  const res = await api.post(`${BASE}/validate`, params);
  return res.data;
};

// Meetings
export const fetchMeetingsForTraining = async (): Promise<MeetingListItem[]> => {
  const res = await api.get(`${BASE}/meetings`);
  return res.data.meetings;
};

export const fetchMeetingData = async (recordingId: string): Promise<MeetingDataSummary> => {
  const res = await api.get(`${BASE}/meetings/${recordingId}/data`);
  return res.data;
};

// File & Point Extraction
export const extractFileForTraining = async (file: File): Promise<{ filename: string; extracted_text: string; char_count: number }> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post(`${BASE}/extract-file`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const extractMomPoints = async (raw_mom_text: string, context_text?: string): Promise<{ points: string[]; count: number }> => {
  const res = await api.post(`${BASE}/extract-points`, { raw_mom_text, context_text });
  return res.data;
};

// Datasets
export const buildDataset = async (params: {
  source_type: 'meeting' | 'upload';
  meeting_id?: string;
  stages: TrainingStage[];
  manual_mom?: string;
  use_edited_stage2?: boolean;
  transcript_text?: string;
  context_text?: string;
  agenda_text?: string;
}): Promise<Partial<TrainingDataset> & { samples_preview: unknown[] }> => {
  const res = await api.post(`${BASE}/datasets/build`, params);
  return res.data;
};

export const uploadTrainingFiles = async (formData: FormData): Promise<Partial<TrainingDataset> & { samples_preview: unknown[] }> => {
  const res = await api.post(`${BASE}/datasets/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const fetchDataset = async (datasetId: string): Promise<TrainingDataset> => {
  const res = await api.get(`${BASE}/datasets/${datasetId}`);
  return res.data;
};

export const listDatasets = async () => {
  const res = await api.get(`${BASE}/datasets`);
  return res.data.datasets;
};

// Jobs
export const startTrainingJob = async (params: {
  dataset_id: string;
  stage_configs: StageConfig[];
  description?: string;
}): Promise<{ job_id: string; status: string }> => {
  const res = await api.post(`${BASE}/jobs`, params);
  return res.data;
};

export const fetchJob = async (jobId: string): Promise<TrainingJob> => {
  const res = await api.get(`${BASE}/jobs/${jobId}`);
  return res.data;
};

export const fetchJobs = async (): Promise<TrainingJob[]> => {
  const res = await api.get(`${BASE}/jobs`);
  return res.data.jobs;
};

export const cancelJob = async (jobId: string): Promise<void> => {
  await api.post(`${BASE}/jobs/${jobId}/cancel`);
};

export const fetchJobResults = async (jobId: string) => {
  const res = await api.get(`${BASE}/jobs/${jobId}/results`);
  return res.data;
};

export const activateArtifact = async (jobId: string, params: {
  artifact_id: string;
  stage: TrainingStage;
}): Promise<void> => {
  await api.post(`${BASE}/jobs/${jobId}/activate`, params);
};

// Artifacts
export const fetchArtifacts = async (stage?: TrainingStage, method?: TrainingMethod): Promise<TrainingArtifact[]> => {
  const params: Record<string, string> = {};
  if (stage) params.stage = stage;
  if (method) params.method = method;
  const res = await api.get(`${BASE}/artifacts`, { params });
  return res.data.artifacts;
};

// History
export const fetchTrainingHistory = async (): Promise<{ history: HistoryEntry[]; artifacts: TrainingArtifact[]; total_jobs: number; total_artifacts: number }> => {
  const res = await api.get(`${BASE}/history`);
  return res.data;
};

// ── Stage 2 Edit Training ────────────────────────────────────
export const fetchEditTrainingData = async (recordingId: string): Promise<{ edits: any[] }> => {
  const res = await api.get(`/rom/${recordingId}/stage2/training-edits`);
  return res.data;
};

export const generateTrainingFromEdits = async (
  recordingId: string,
  changeIds: string[]
): Promise<{ dataset_id: string; sample_count: number; samples_preview: any[] }> => {
  const res = await api.post(`/rom/${recordingId}/stage2/generate-training-from-edits`, {
    selected_change_ids: changeIds,
  });
  return res.data;
};

export const fetchMeetingsWithEditHistory = async (): Promise<{ meetings: any[] }> => {
  const res = await api.get(`${BASE}/meetings`);
  return res.data;
};
