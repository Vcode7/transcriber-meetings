export interface TranscriptWord {
  word: string
  start: number
  end: number
  probability: number
}

export interface OverlapRegion {
  start: number
  end: number
  speakers: string[]   // resolved names after speaker identification
}

export interface TranscriptSegment {
  seg_id?: string
  speaker?: string
  speaker_label: string
  start: number
  end: number
  text: string
  words: TranscriptWord[]
  is_overlap: boolean
  overlap_regions?: OverlapRegion[]
}

export interface SpeakerSummaryData {
  summary: string
  key_points: string[]
  action_items: string[]
}

export interface VideoTranscriptBlock {
  start: number
  end: number
  text: string
}

export interface ProcessingResult {
  filename?: string
  transcript?: TranscriptSegment[]
  summary?: string
  short_summary?: string
  detailed_summary?: string
  key_points?: string[]
  action_items?: string[]
  speakers_detected?: string[]
  speaker_summary?: Record<string, SpeakerSummaryData> | null
  source_type?: string
  video_transcript?: VideoTranscriptBlock[] | null
  [key: string]: unknown
}

export interface RecordingDetail extends ProcessingResult {
  id?: string
  filename: string
  file_path?: string
  duration: number
  status: string
  created_at: string
  source_type?: string
  video_transcript?: VideoTranscriptBlock[] | null
}

export interface Collection {
  id: string
  name: string
  description: string
  meeting_count: number
  created_at: string
  updated_at: string
}

export interface CollectionDetail extends Collection {
  meetings: CollectionMeeting[]
}

export interface CollectionMeeting {
  id: string
  filename: string
  duration: number
  status: string
  speakers_detected: string[]
  has_summary: boolean
  created_at: string
  display_order: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  message_type: 'chat' | 'comparison' | 'topic_growth'
  metadata: {
    cited_meetings?: string[]
    meeting_names?: Record<string, string>
  }
  created_at: string
}

// ── AI Chat Tab Types ───────────────────────────
export interface AIChatMeeting {
  id: string
  name: string
  date: string
  has_stage2?: boolean
  has_final_rom?: boolean
  is_manual_mom?: boolean
  file_type?: string
  file_size?: number
}

export interface ManualMomItem {
  id: string
  filename: string
  title: string
  meeting_date: string
  file_size: number
  file_type: string
  speakers_count: number
  agendas_count: number
  action_items_count: number
  decisions_count: number
  chunks_count: number
  created_at: string
  updated_at: string
}

export interface ManualMomAgenda {
  agenda_id?: string
  title: string
  discussion_points?: string[]
  content?: string
}

export interface ManualMomActionItem {
  task: string
  owner?: string
  due_date?: string
  status?: string
}

export interface ManualMomChunk {
  chunk_index: number
  agenda_title?: string
  speakers?: string[]
  text: string
}

export interface ManualMomDetail {
  id: string
  filename: string
  title: string
  meeting_date: string
  file_path: string
  file_size: number
  file_type: string
  raw_text: string
  summary: string
  speakers: string[]
  agendas: ManualMomAgenda[]
  action_items: ManualMomActionItem[]
  decisions: string[]
  metadata_json: Record<string, any>
  chunks: ManualMomChunk[]
  created_at: string
  updated_at: string
}

export interface AIChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  metadata: {
    meeting_sources?: AIChatMeeting[]
    query_type?: 'direct' | 'retrieval'
  }
  created_at: string
}

export interface AIChatSession {
  id: string
  title: string
  selected_meeting_id?: string
  messages: AIChatMessage[]
  currentSources?: AIChatMeeting[]
  currentQueryType?: string
  created_at: string
  updated_at: string
}

export interface SyncableMeeting {
  id: string
  name: string
  date: string
  stage2_points_count: number
  indexed_stage2_count?: number
  stage2_sync_status?: 'synced' | 'partial' | 'not_indexed' | 'not_generated'
  final_rom_points_count: number
  indexed_final_rom_count?: number
  final_rom_sync_status?: 'synced' | 'partial' | 'not_indexed' | 'not_generated'
  has_final_rom: boolean
  has_stage2: boolean
  indexed_points_count: number
  needs_sync: boolean
  sync_status: 'synced' | 'partial' | 'not_indexed'
}

export interface ResyncResult {
  status: 'success' | 'error'
  recording_id: string
  meeting_name: string
  total_found: number
  already_indexed: number
  newly_indexed: number
  skipped_failed: number
  final_rom_result?: {
    total_found: number
    already_indexed: number
    newly_indexed: number
    skipped_failed: number
  }
  stage2_result?: {
    total_found: number
    already_indexed: number
    newly_indexed: number
    skipped_failed: number
  }
  message?: string
}


