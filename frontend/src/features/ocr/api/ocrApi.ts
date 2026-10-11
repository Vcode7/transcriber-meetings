import api from '../../../api/client'
import type {
  OcrSaveResponse,
  OcrSettings,
  OcrSettingsResponse,
  OcrStatus,
  OcrTestResult,
  VlmModelsResponse,
  VlmPromptMode,
  VlmResult,
} from '../types'

export async function fetchOcrSettings(): Promise<OcrSettingsResponse> {
  const res = await api.get<OcrSettingsResponse>('/ocr/settings')
  return res.data
}

export async function saveOcrSettings(settings: OcrSettings): Promise<OcrSaveResponse> {
  const res = await api.put<OcrSaveResponse>('/ocr/settings', settings)
  return res.data
}

export async function resetOcrSettings(): Promise<OcrSaveResponse> {
  const res = await api.post<OcrSaveResponse>('/ocr/settings/reset')
  return res.data
}

export async function fetchOcrStatus(): Promise<OcrStatus> {
  const res = await api.get<OcrStatus>('/ocr/status')
  return res.data
}

export async function runOcrTest(
  file: File,
  opts: { settings?: OcrSettings; maxPages: number; showBoxes: boolean },
  signal?: AbortSignal,
): Promise<OcrTestResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('max_pages', String(opts.maxPages))
  form.append('show_boxes', String(opts.showBoxes))
  if (opts.settings) form.append('settings', JSON.stringify(opts.settings))
  const res = await api.post<OcrTestResult>('/ocr/test', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    signal,
  })
  return res.data
}

export async function fetchVlmModels(): Promise<VlmModelsResponse> {
  const res = await api.get<VlmModelsResponse>('/ocr/vlm/models')
  return res.data
}

export async function runVlmOcr(
  file: File,
  opts: { model: string; page: number; promptMode: VlmPromptMode; maxTokens: number; customPrompt?: string },
  signal?: AbortSignal,
): Promise<VlmResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('model', opts.model)
  form.append('page', String(opts.page))
  form.append('prompt_mode', opts.promptMode)
  form.append('max_tokens', String(opts.maxTokens))
  if (opts.customPrompt?.trim()) form.append('custom_prompt', opts.customPrompt)
  const res = await api.post<VlmResult>('/ocr/vlm/test', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    signal,
  })
  return res.data
}
