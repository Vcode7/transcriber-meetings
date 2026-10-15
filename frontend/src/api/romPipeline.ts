import api from './client'
import type { AgendaItem } from '../components/rom/AgendaItemsEditor'
import type { PreviousMeetingOption } from '../components/rom/RomStageOptions'

const BASE = '/rom-pipeline'

/** off = no automatic run · mom = Stage 1 → 2 → MoM · rom = Stage 1 → 2 → Agenda Matching → Final ROM */
export type RomOutputType = 'off' | 'mom' | 'rom'

export interface StagedFileRef {
  staged_id: string
  filename: string
}

/** Mirrors backend/models/rom_pipeline.py RomPipelineSettings. */
export interface RomPipelineSettings {
  output_type: RomOutputType
  transcript_window_minutes: number
  meeting_top_k: number
  global_top_k: number
  discussion_window_size: number
  min_similarity_threshold: number | null
  process_all_together: boolean | null
  previous_meeting_mode: 'auto' | 'select' | 'off'
  previous_meeting_id: string | null
  previous_meeting_top_k: number
  /** rom only: generate action points for every agenda of the Final ROM */
  include_action_points: boolean
  /** rom only: agenda items come from the agenda document's tables (whole document when it has none) */
  agenda_from_tables: boolean
  context_files: StagedFileRef[]
  agenda_doc: StagedFileRef | null
  agendas: AgendaItem[] | null
  agenda_text: string | null
}

export const DEFAULT_ROM_SETTINGS: RomPipelineSettings = {
  output_type: 'off',
  transcript_window_minutes: 2,
  meeting_top_k: 5,
  global_top_k: 3,
  discussion_window_size: 5,
  min_similarity_threshold: 0.6,
  process_all_together: false,
  previous_meeting_mode: 'auto',
  previous_meeting_id: null,
  previous_meeting_top_k: 3,
  include_action_points: true,
  agenda_from_tables: true,
  context_files: [],
  agenda_doc: null,
  agendas: null,
  agenda_text: null,
}

export type RomPipelineState = 'waiting' | 'queued' | 'running' | 'done' | 'error' | 'interrupted' | 'off'

export interface RomPipelineStatus {
  state: RomPipelineState
  output_type?: RomOutputType
  stage?: string | null
  stage_label?: string
  message?: string
  error?: string | null
  started_at?: string
  finished_at?: string | null
}

export interface RomPipelineInfo {
  settings: RomPipelineSettings | null
  status: RomPipelineStatus | null
  recording_status?: string
  stage1_progress?: {
    windows_completed: number
    windows_total: number
    concurrency: number
    eta_seconds: number | null
    elapsed_seconds: number
    status: string
  } | null
}

/** How agenda items were read: from the document's tables, or from the whole document. */
export interface AgendaExtraction {
  agendas: AgendaItem[]
  extraction_mode: 'tables' | 'document'
  table_count: number
}

export interface ExtractedAgendaDocument extends StagedFileRef, AgendaExtraction {
  agenda_text: string
  slide_count: number
}

/** Toast text for an agenda extraction ("from 2 tables", or why the whole document was used). */
export function describeAgendaExtraction(res: AgendaExtraction, fromTables: boolean): string {
  const items = `${res.agendas.length} agenda item(s)`
  if (res.extraction_mode === 'tables') return `${items} from ${res.table_count} table(s)`
  if (!fromTables) return `${items} from the whole document`
  return res.table_count
    ? `${items} from the whole document (its tables list no agenda items)`
    : `${items} from the whole document (no tables found)`
}

export async function getRomDefaults(): Promise<{ defaults: Partial<RomPipelineSettings>; saved: boolean }> {
  const res = await api.get(`${BASE}/defaults`)
  return res.data
}

export async function saveRomDefaults(settings: RomPipelineSettings): Promise<void> {
  await api.put(`${BASE}/defaults`, settings)
}

export async function stageContextFile(file: File): Promise<StagedFileRef> {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post(`${BASE}/context-files`, form)
  return res.data
}

export async function extractAgendaDocument(file: File, fromTables: boolean): Promise<ExtractedAgendaDocument> {
  const form = new FormData()
  form.append('file', file)
  form.append('from_tables', String(fromTables))
  const res = await api.post(`${BASE}/agenda-document`, form, { timeout: 0 })
  return res.data
}

/** Extract the agenda items of an already staged agenda document again (no re-upload / OCR). */
export async function reextractAgendaItems(stagedId: string, fromTables: boolean): Promise<AgendaExtraction> {
  const res = await api.post(`${BASE}/agenda-document/${stagedId}/agendas`, { from_tables: fromTables }, { timeout: 0 })
  return res.data
}

export async function deleteStagedFile(stagedId: string): Promise<void> {
  await api.delete(`${BASE}/staged/${stagedId}`)
}

export async function listPreviousMeetings(): Promise<PreviousMeetingOption[]> {
  const res = await api.get(`${BASE}/previous-meetings`)
  return Array.isArray(res.data?.meetings) ? res.data.meetings : []
}

export async function getRomPipeline(recordingId: string): Promise<RomPipelineInfo> {
  const res = await api.get(`${BASE}/${recordingId}`)
  return res.data
}

export async function startRomPipeline(recordingId: string): Promise<void> {
  await api.post(`${BASE}/${recordingId}/start`)
}

/** Adds the settings to a capture form (Upload / Record / Tab Recording) and saves them as the
 * user's defaults when asked. */
export async function appendRomSettings(form: FormData, settings: RomPipelineSettings, saveAsDefault: boolean) {
  const payload: RomPipelineSettings = settings.output_type === 'rom'
    ? settings
    : { ...settings, agenda_doc: null, agendas: null, agenda_text: null }
  form.append('rom_settings', JSON.stringify(payload))
  if (saveAsDefault) {
    try {
      await saveRomDefaults(settings)
    } catch {
      // Saving defaults must never block the capture itself.
    }
  }
}
