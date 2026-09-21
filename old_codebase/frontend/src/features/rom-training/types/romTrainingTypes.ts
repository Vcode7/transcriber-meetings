export interface Meeting {
  id: string;
  title: string;
  created_at: string;
  has_rom: boolean;
  has_long_rom: boolean;
}

export interface LongRomAgenda {
  agenda_id: string;
  agenda_title: string;
  discussion_points: any[];
}

export interface MomAgenda {
  agenda_title: string;
  points: string[];
}

export interface AgendaMatch {
  mom_agenda_title: string;
  mom_agenda_idx: number;
  meeting_agenda_id: string | null;
  meeting_agenda_title: string | null;
  confidence: number;
  manual: boolean;
}

export interface TrainingDataEntry {
  id: string;
  recording_id: string;
  recording_title: string;
  agenda_id: string;
  agenda_title: string;
  long_rom_points: any[];
  manual_mom_points: any[];
  created_at: string;
}

export interface RomVariant {
  id: string;
  name: string;
  description: string | null;
  base_model: string;
  training_type: 'new' | 'upgrade';
  parent_variant_id: string | null;
  adapter_path: string;
  status: 'pending' | 'training' | 'done' | 'error';
  config: Record<string, any>;
  dataset_size: number;
  test_size: number;
  test_results: TestResult[];
  accuracy: string | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface TestResult {
  agenda_title: string;
  recording_title: string;
  before: string;
  after: string;
  manual: string;
}

export interface LoRAConfig {
  method: 'lora' | 'qlora';
  r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string;
  learning_rate: number;
  num_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  use_4bit: boolean;
  bnb_4bit_compute_dtype: string;
  max_seq_length: number;
}
