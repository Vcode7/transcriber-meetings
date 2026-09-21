// Stage 1 Training — TypeScript types
// These are exclusively used by the new Stage 1 training workflow.

export type FeedbackCategory =
  | 'missing_point'
  | 'lost_info'
  | 'wrong_speaker'
  | 'wrong_grouping'
  | 'unrelated_merged'
  | 'unnecessary_split'
  | 'missing_decision'
  | 'missing_question'
  | 'missing_date'
  | 'missing_number'
  | 'missing_technical_term'
  | 'hallucination'
  | 'incorrect_fact'
  | 'duplicate_point'
  | 'invalid_json'
  | 'other';

export const FEEDBACK_CATEGORY_LABELS: Record<FeedbackCategory, string> = {
  missing_point: 'Missing discussion point',
  lost_info: 'Lost important information',
  wrong_speaker: 'Wrong speaker',
  wrong_grouping: 'Wrong grouping',
  unrelated_merged: 'Unrelated points merged',
  unnecessary_split: 'Topic unnecessarily split',
  missing_decision: 'Missing decision',
  missing_question: 'Missing question',
  missing_date: 'Missing date',
  missing_number: 'Missing number',
  missing_technical_term: 'Missing technical term',
  hallucination: 'Hallucination',
  incorrect_fact: 'Incorrect fact',
  duplicate_point: 'Duplicate point',
  invalid_json: 'Invalid JSON',
  other: 'Other',
};

export interface TranscriptSegment {
  speaker_label: string;
  start: number;
  end: number;
  text: string;
}

export interface TranscriptWindow {
  window_id: string;
  window_index: number;
  start_time: number;
  end_time: number;
  segments: TranscriptSegment[];
  transcript_text: string;
  speakers: string[];
  segment_count: number;
  selected: boolean;
  role?: 'train' | 'val'; // undefined = auto-split
  context_summary?: string;
  agenda_summary?: string;
  meeting_id?: string;
}

export interface Stage1EvalScores {
  overall: number;
  json_validity: number;
  schema_validity: number;
  required_fields: number;
  action_items_empty: number;
  transcript_coverage: number;
  speaker_preservation: number;
  date_preservation: number;
  number_preservation: number;
  technical_term_preservation: number;
  factual_preservation: number;
  duplicate_detection: number;
  missing_content: number;
  hallucination_detection: number;
}

export interface Stage1Variant {
  variant_id: string;
  label: string;
  is_default: boolean;
  parent_variant_id: string | null;
  created_at: string;
  optimizer: string;
  model: string;
  training_window_count: number;
  validation_window_count: number;
  feedback_count: number;
  artifact_path: string | null;
  status: 'done' | 'training' | 'error';
  scores: Stage1EvalScores;
  improvement_over_baseline: number | null;
  is_best: boolean;
  config_snapshot?: Record<string, unknown>;
  run_id?: string;
  instructions?: string;
}

export interface Stage1ValidationOutput {
  raw_output: string;
  parsed_output: Record<string, unknown> | null;
  eval_scores: Stage1EvalScores;
  variant_id: string;
  variant_label: string;
  error?: string;
}

export interface FeedbackEntry {
  feedback_id?: string;
  meeting_id: string;
  window_id: string;
  variant_id: string;
  transcript_window: string;
  model_output: string;
  corrected_output?: string;
  categories: FeedbackCategory[];
  comment: string;
  context_summary?: string;
  agenda_summary?: string;
  timestamp?: string;
}

export interface Stage1HistoryEntry {
  run_id: string;
  user_id?: string;
  variant_id: string;
  variant_label: string;
  meeting_id: string;
  window_count: number;
  training_window_count: number;
  validation_window_count: number;
  optimizer: string;
  model: string;
  baseline_score: number;
  final_score: number;
  feedback_rounds: number;
  created_at: string;
  status: 'done' | 'error' | 'running';
  parent_variant_id: string | null;
}

export interface Stage1Settings {
  model_name: string;
  optimizer: 'BootstrapFewShot' | 'MIPROv2';
  num_trials: number;
  eval_split: number;
  bootstrap_examples: number;
  max_demonstrations: number;
  temperature: number;
  max_tokens: number;
  default_window_size_minutes: number;
  default_window_overlap_seconds: number;
  min_coverage_threshold: number;
  min_json_validity_threshold: number;
}

export const DEFAULT_STAGE1_SETTINGS: Stage1Settings = {
  model_name: '',
  optimizer: 'BootstrapFewShot',
  num_trials: 10,
  eval_split: 0.2,
  bootstrap_examples: 3,
  max_demonstrations: 4,
  temperature: 0.0,
  max_tokens: 2048,
  default_window_size_minutes: 2,
  default_window_overlap_seconds: 30,
  min_coverage_threshold: 0.5,
  min_json_validity_threshold: 1.0,
};

export interface OptimizationRunStatus {
  run_id: string;
  status: 'starting' | 'running' | 'done' | 'error';
  progress: number; // 0–100
  message: string;
  variant_id: string | null;
  error: string | null;
}

export const METRIC_LABELS: Record<keyof Stage1EvalScores, string> = {
  overall: 'Overall Score',
  json_validity: 'JSON Validity',
  schema_validity: 'Schema Validity',
  required_fields: 'Required Fields',
  action_items_empty: 'Action Items Empty',
  transcript_coverage: 'Transcript Coverage',
  speaker_preservation: 'Speaker Preservation',
  date_preservation: 'Date Preservation',
  number_preservation: 'Number Preservation',
  technical_term_preservation: 'Technical Terms',
  factual_preservation: 'Factual Accuracy',
  duplicate_detection: 'No Duplicates',
  missing_content: 'Missing Content',
  hallucination_detection: 'No Hallucinations',
};
