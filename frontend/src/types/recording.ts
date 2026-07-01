export interface TranscriptWord {
  word: string
  start: number
  end: number
  probability: number
}

export interface TranscriptSegment {
  speaker_label: string
  start: number
  end: number
  text: string
  words: TranscriptWord[]
  is_overlap: boolean
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

