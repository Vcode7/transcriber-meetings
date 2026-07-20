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
  [key: string]: unknown
}

export interface RecordingDetail extends ProcessingResult {
  id?: string
  filename: string
  file_path?: string
  duration: number
  status: string
  created_at: string
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
