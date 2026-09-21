// Stage 2 Training — TypeScript types
// These are exclusively used by the Stage 2 training workflow.

export type Stage2FeedbackCategory =
  | 'missing_point'
  | 'incorrect_merge'
  | 'incorrect_split'
  | 'lost_discussion_detail'
  | 'lost_decision'
  | 'lost_question'
  | 'lost_action_info'
  | 'wrong_speaker'
  | 'wrong_action_owner'
  | 'hallucinated_info'
  | 'incorrect_context'
  | 'duplicate_point'
  | 'other';

export const STAGE2_FEEDBACK_LABELS: Record<Stage2FeedbackCategory, string> = {
  missing_point: 'Missing point',
  incorrect_merge: 'Incorrect merge',
  incorrect_split: 'Incorrect split',
  lost_discussion_detail: 'Lost discussion detail',
  lost_decision: 'Lost decision',
  lost_question: 'Lost question',
  lost_action_info: 'Lost action information',
  wrong_speaker: 'Wrong speaker',
  wrong_action_owner: 'Wrong action owner',
  hallucinated_info: 'Hallucinated information',
  incorrect_context: 'Incorrect context usage',
  duplicate_point: 'Duplicate point',
  other: 'Other',
};

export interface Stage2EvalScores {
  overall: number;
  json_validity: number;
  schema_validity: number;
  point_coverage: number;
  reference_point_recall: number;
  semantic_similarity: number;
  info_preservation: number;
  speaker_preservation: number;
  action_owner_preservation: number;
  duplicate_detection: number;
  hallucination_detection: number;
}

export interface Stage2Point {
  point: string;
  speaker: string[];
  action_owner: string[];
}

export interface Stage2OutputParsed {
  points: Stage2Point[];
}

export interface Stage2Group {
  group_index: number;
  points: any[];         // Stage 1 discussion point objects
  global_context: string;
  meeting_context: string;
  point_count: number;
}

export interface Stage2Variant {
  variant_id: string;
  label: string;
  is_default: boolean;
  parent_variant_id: string | null;
  created_at: string;
  method?: 'dspy' | 'lora' | 'qlora';
  optimizer: string;
  model: string;
  base_model?: string;
  training_group_count: number;
  validation_group_count: number;
  feedback_count: number;
  artifact_path: string | null;
  status: 'done' | 'training' | 'error';
  scores: Stage2EvalScores;
  improvement_over_baseline: number | null;
  is_best: boolean;
  is_active?: boolean;
  config_snapshot?: {
    training_mode?: 'dspy' | 'lora' | 'qlora';
    hf_model_path?: string;
    lora_r?: number;
    lora_alpha?: number;
    lora_dropout?: number;
    num_train_epochs?: number;
    learning_rate?: number;
    points_per_group: number;
    context_retrieval: boolean;
  };
  run_id?: string;
  meeting_id?: string;
  instructions?: string;
}

export interface Stage2ValidationOutput {
  raw_output: string;
  parsed_output: Stage2OutputParsed | null;
  eval_scores: Stage2EvalScores;
  variant_id: string;
  variant_label: string;
  error?: string;
}

export interface Stage2FeedbackEntry {
  feedback_id?: string;
  meeting_id: string;
  group_index: number;
  variant_id: string;
  stage1_points_input: string;   // JSON string
  model_output: string;
  corrected_output?: string;
  categories: Stage2FeedbackCategory[];
  comment: string;
  global_context?: string;
  meeting_context?: string;
  timestamp?: string;
}

export interface Stage2HistoryEntry {
  run_id: string;
  user_id?: string;
  variant_id: string;
  variant_label: string;
  meeting_id: string;
  training_group_count: number;
  validation_group_count: number;
  points_per_group: number;
  context_retrieval: boolean;
  optimizer: string;
  model: string;
  baseline_score: number;
  final_score: number;
  feedback_rounds: number;
  created_at: string;
  status: 'done' | 'error' | 'running';
  parent_variant_id: string | null;
}

export interface Stage2ValidateModelResult {
  valid: boolean;
  reason: string;
  warnings: string[];
  details?: {
    path?: string;
    model_type?: string;
    weights_count?: number;
    has_tokenizer?: boolean;
  };
}

export interface Stage2Settings {
  training_mode: 'dspy' | 'lora' | 'qlora';
  model_name: string;
  optimizer: 'BootstrapFewShot' | 'MIPROv2';
  num_trials: number;
  eval_split: number;
  bootstrap_examples: number;
  max_demonstrations: number;
  temperature: number;
  max_tokens: number;
  points_per_group: number;
  context_retrieval: boolean;
  // LoRA / QLoRA settings
  hf_model_path: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_target_modules: string;
  learning_rate: number;
  num_train_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
}

export const DEFAULT_STAGE2_SETTINGS: Stage2Settings = {
  training_mode: 'dspy',
  model_name: '',
  optimizer: 'BootstrapFewShot',
  num_trials: 10,
  eval_split: 0.2,
  bootstrap_examples: 3,
  max_demonstrations: 4,
  temperature: 0.0,
  max_tokens: 2048,
  points_per_group: 5,
  context_retrieval: true,
  // LoRA / QLoRA defaults
  hf_model_path: '',
  lora_r: 8,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: '',
  learning_rate: 0.0002,
  num_train_epochs: 3,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 4,
  max_seq_length: 2048,
};

export interface Stage2OptimizationRunStatus {
  run_id: string;
  status: 'starting' | 'running' | 'done' | 'error';
  progress: number;
  message: string;
  variant_id: string | null;
  error: string | null;
}

export type ReferenceSource = 'edited_stage2' | 'existing_stage2' | 'manual_mom' | 'none';

export const STAGE2_METRIC_LABELS: Record<keyof Stage2EvalScores, string> = {
  overall: 'Overall Score',
  json_validity: 'JSON Validity',
  schema_validity: 'Schema Validity',
  point_coverage: 'Point Coverage',
  reference_point_recall: 'Reference Recall',
  semantic_similarity: 'Semantic Similarity',
  info_preservation: 'Info Preservation',
  speaker_preservation: 'Speaker Preservation',
  action_owner_preservation: 'Action Owner Preservation',
  duplicate_detection: 'No Duplicates',
  hallucination_detection: 'No Hallucinations',
};

export const STAGE2_METRIC_ORDER: (keyof Stage2EvalScores)[] = [
  'overall',
  'json_validity', 'schema_validity',
  'point_coverage', 'reference_point_recall', 'semantic_similarity',
  'info_preservation', 'speaker_preservation', 'action_owner_preservation',
  'duplicate_detection', 'hallucination_detection',
];
