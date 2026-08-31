export type TrainingStage = 'stage_1' | 'stage_2' | 'stage_3';
export type TrainingMethod = 'dspy' | 'lora' | 'qlora';
export type JobStatus = 'pending' | 'building_dataset' | 'training' | 'evaluating' | 'done' | 'error' | 'cancelled';

export interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
  details?: Record<string, unknown>;
}

export interface StageConfig {
  stage: TrainingStage;
  method: TrainingMethod;
  model_name: string;
  // DSPy
  dspy_optimizer: 'BootstrapFewShot' | 'MIPROv2';
  dspy_max_bootstrapped_demos: number;
  dspy_max_labeled_demos: number;
  dspy_num_trials: number;
  // LoRA/QLoRA
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  num_train_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  max_seq_length: number;
  hf_model_path?: string;
  eval_split: number;
  eval_metric: string;
}

export const defaultStageConfig = (stage: TrainingStage): StageConfig => ({
  stage,
  method: 'dspy',
  model_name: '',
  dspy_optimizer: 'BootstrapFewShot',
  dspy_max_bootstrapped_demos: 3,
  dspy_max_labeled_demos: 4,
  dspy_num_trials: 10,
  lora_r: 8,
  lora_alpha: 32,
  lora_dropout: 0.05,
  num_train_epochs: 3,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 4,
  learning_rate: 0.0002,
  max_seq_length: 2048,
  eval_split: 0.2,
  eval_metric: 'rouge',
});

export interface ValidationResult {
  valid: boolean;
  reason: string;
  warnings: string[];
}

export interface TrainingJobProgress {
  step: number;
  total_steps: number;
  epoch: number;
  total_epochs: number;
  loss?: number;
  message: string;
  percent: number;
}

export interface TrainingArtifact {
  artifact_id: string;
  stage: TrainingStage;
  method: TrainingMethod;
  base_model: string;
  dataset_ids: string[];
  training_date: string;
  config: StageConfig;
  status: string;
  eval_metrics?: Record<string, unknown>;
  artifact_path: string;
  version: number;
  is_active: boolean;
  job_id?: string;
  notes?: string;
}

export interface TrainingJob {
  job_id: string;
  user_id: string;
  description?: string;
  status: JobStatus;
  stage_configs: StageConfig[];
  dataset_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  logs: string[];
  progress?: TrainingJobProgress;
  artifacts: TrainingArtifact[];
  eval_results?: Record<string, EvaluationResult>;
}

export interface EvaluationResult {
  job_id: string;
  stage: TrainingStage;
  method: TrainingMethod;
  score_before: number;
  score_after: number;
  metrics: Record<string, unknown>;
  sample_comparisons: Array<Record<string, unknown>>;
  error?: string;
}

export interface DatasetSample {
  sample_id: string;
  stage: TrainingStage;
  source_meeting_id?: string;
  inputs: Record<string, unknown>;
  target: string;
  metadata?: Record<string, unknown>;
}

export interface TrainingDataset {
  dataset_id: string;
  stages: TrainingStage[];
  total_samples: number;
  samples_by_stage: Record<string, number>;
  samples: DatasetSample[];
  created_at: string;
  source_type: 'meeting' | 'upload';
  meeting_id?: string;
  manual_mom?: string;
}

export interface MeetingListItem {
  id: string;
  filename: string;
  duration: number;
  created_at: string;
  processed_at?: string;
  stages_available: TrainingStage[];
  has_transcript: boolean;
  has_rom_data: boolean;
  has_stage2_edits?: boolean;
  stage2_edit_count?: number;
}

export interface MeetingDataSummary {
  recording_id: string;
  filename: string;
  transcript_segments: number;
  has_stage1: boolean;
  has_stage2: boolean;
  has_stage3: boolean;
  has_mom: boolean;
  has_agenda: boolean;
  stage1_points: number;
  stage2_points: number;
  stage3_agendas: number;
  has_stage2_edits?: boolean;
  stage2_edit_count?: number;
  original_stage2_points?: any[];
  edited_stage2_points?: any[];
}

export interface HistoryEntry {
  job_id: string;
  created_at: string;
  status: JobStatus;
  description?: string;
  stages: TrainingStage[];
  methods: TrainingMethod[];
  models: string[];
  artifacts: TrainingArtifact[];
  eval_results: Record<string, EvaluationResult>;
  error?: string;
}

// ─── Stage 3: Point -> Agenda Assignment Types ────────────────────────────────

export interface Stage3Agenda {
  agenda_id: string;
  title: string;
  description?: string;
  keywords?: string[];
  points?: any[];
}

export interface Stage3CandidateAgenda {
  agenda_id: string;
  agenda_title: string;
  score: number;
}

export interface Stage3PointItem {
  point_id: string;
  enhanced_point: string;
  speakers?: string[];
  timeline_start?: number;
  timeline_end?: number;
  candidate_agendas: Stage3CandidateAgenda[];
  ground_truth_agenda_id?: string;
}

export interface Stage3Batch {
  batch_index: number;
  point_count: number;
  points: Stage3PointItem[];
}

export interface Stage3Assignment {
  point_id: string;
  assigned_agenda_id: string;
  confidence: string;
  reason: string;
  enhanced_point?: string;
  speakers?: string[];
  candidate_agendas?: Stage3CandidateAgenda[];
  agenda_title?: string;
  ground_truth_agenda_id?: string;
  is_match_ground_truth?: boolean;
}

export interface Stage3AgendaGroup {
  agenda_id: string;
  title: string;
  description?: string;
  points: Stage3Assignment[];
}

export interface Stage3ValidationScores {
  overall: number;
  accuracy: number;
  candidate_validity: number;
  json_validity: number;
  schema_validity: number;
}

export interface Stage3ValidationOutput {
  variant_id: string;
  meeting_id: string;
  batch_index: number;
  scores: Stage3ValidationScores;
  raw_output: string;
  assignments: Stage3Assignment[];
  grouped_by_agenda: Stage3AgendaGroup[];
}

export interface Stage3Variant {
  variant_id: string;
  label: string;
  is_default: boolean;
  parent_variant_id: string | null;
  created_at: string;
  optimizer: string;
  model: string;
  training_batch_count: number;
  validation_batch_count: number;
  feedback_count: number;
  instructions?: string;
  artifact_path: string | null;
  status: string;
  scores: Stage3ValidationScores;
  improvement_over_baseline: number | null;
  is_best?: boolean;
}

export interface Stage3HistoryEntry {
  run_id: string;
  timestamp: string;
  variant_id?: string;
  status: string;
  scores?: Stage3ValidationScores;
  feedback_count?: number;
  batch_count?: number;
  error?: string;
}

export interface Stage3Settings {
  model_name: string;
  optimizer: string;
  num_trials: number;
  eval_split: number;
  bootstrap_examples: number;
  max_demonstrations: number;
  temperature: number;
  max_tokens: number;
  points_per_batch: number;
}

export interface Stage3FeedbackEntry {
  feedback_id?: string;
  meeting_id: string;
  point_id: string;
  point_text: string;
  original_agenda_id: string;
  correct_agenda_id: string;
  reason?: string;
}

