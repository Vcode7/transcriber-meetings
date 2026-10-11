export type OcrSettingValue = number | string | boolean
export type OcrSettings = Record<string, OcrSettingValue>

export interface OcrFieldOption {
  value: string
  label: string
}

export interface OcrFieldSchema {
  name: string
  type: 'bool' | 'int' | 'float' | 'enum'
  group: string
  label: string
  help: string
  default: OcrSettingValue
  min?: number
  max?: number
  step?: number
  unit?: string
  options?: OcrFieldOption[]
  depends_on?: { field: string; values: OcrSettingValue[] }
  warning?: string
}

export interface OcrGroupSchema {
  id: string
  title: string
  description: string
}

export interface OcrSchema {
  version: number
  groups: OcrGroupSchema[]
  fields: OcrFieldSchema[]
}

export interface OcrPreset {
  id: string
  label: string
  description: string
  settings: OcrSettings
}

export interface OcrSettingsResponse {
  settings: OcrSettings
  defaults: OcrSettings
  presets: OcrPreset[]
  schema: OcrSchema
  updated_at: string | null
}

export interface OcrSaveResponse {
  settings: OcrSettings
  updated_at: string | null
  fingerprint: string
}

export interface OcrStatus {
  model: string
  model_dir: string
  files: Record<string, number | null>
  models_present: boolean
  loaded: boolean
  active_provider: string | null
  available_providers: string[]
  gpu_available: boolean
  settings_fingerprint: string
  settings_updated_at: string | null
}

export interface OcrTestLine {
  text: string
  score: number
}

export interface OcrTestPage {
  page: number
  source: 'ocr' | 'text_layer' | 'document'
  text: string
  char_count: number
  word_count: number
  avg_confidence?: number
  line_count?: number
  box_count?: number
  rotation?: number
  skew_angle?: number
  applied_steps?: string[]
  image_size?: [number, number]
  det_input_size?: [number, number]
  timings_ms?: Record<string, number>
  lines?: OcrTestLine[]
  preview?: string | null
  elapsed_ms?: number
  render_dpi?: number
  text_layer_chars?: number
  embedded_images?: number
  embedded_images_with_text?: number
}

export interface OcrTestResult {
  filename: string
  kind: 'image' | 'pdf' | 'document'
  page_count: number
  pages_processed: number
  pages: OcrTestPage[]
  text: string
  char_count: number
  word_count: number
  avg_confidence: number | null
  elapsed_ms: number
  settings_fingerprint: string
  provider: string | null
  used_unsaved_settings: boolean
}

export interface VlmModel {
  name: string
  size?: number
  parameter_size?: string
  family?: string
  vision: boolean | null
  thinking: boolean | null
  running: boolean
  is_chandra: boolean
}

export interface VlmModelsResponse {
  success: boolean
  server_url: string
  models: VlmModel[]
  busy?: boolean
  error?: string
}

export type VlmPromptMode = 'ocr_layout' | 'ocr'

export interface VlmResult {
  html: string
  raw: string
  model: string
  prompt_mode: string
  elapsed_ms: number
  prompt_tokens?: number
  output_tokens?: number
  done_reason?: string
  truncated: boolean
  num_ctx: number
  vision_capable: boolean | null
  server_url: string
  filename: string
  page: number
  page_count: number
  image_size: [number, number]
  source_preview: string
}
