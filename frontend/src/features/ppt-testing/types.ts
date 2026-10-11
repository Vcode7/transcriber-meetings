export type InsightStatus = 'completed' | 'ongoing' | 'delayed' | 'at_risk' | 'planned' | 'unknown'
export type InsightType = 'delay' | 'risk' | 'decision' | 'dependency' | 'action' | 'milestone' | 'progress' | 'status'
export type InsightTemporal = 'current' | 'upcoming_soon' | 'upcoming' | 'recent' | 'undated'
export type PptFocus = 'narrow' | 'standard' | 'wide'

export interface SlideInsight {
  id: string
  ref: string
  project: string
  slides: number[]
  slide_title?: string
  type: InsightType
  status: InsightStatus
  headline: string
  summary: string
  attention: string
  dates: string[]
  owner?: string
  temporal: InsightTemporal
  priority: number
}

export interface ProjectBrief {
  name: string
  summary: string
  overall_status: InsightStatus | 'mixed'
  needs_attention: string[]
  slides: number[]
  priority: number
  insight_ids: string[]
}

export interface TimelineEntry {
  label: string
  start: string
  end: string
  slide: number
  dates: string[]
  status_hint: string
}

export interface SlideRecord {
  number: number
  title: string
  kind: 'slide' | 'page' | 'section'
  chars: number
  relevance?: 'dated' | 'undated' | 'out_of_window'
  dates?: string[]
  status?: 'analyzed' | 'skipped_empty' | 'skipped_out_of_window' | 'error' | 'pending'
  insight_count?: number
  project?: string
  reason?: string
}

export interface PptAnalysis {
  version: number
  meeting_date: string
  focus: PptFocus
  window: { from: string; to: string }
  filename: string
  deck_title: string
  overview: string
  timeline: {
    current_phase: TimelineEntry[]
    upcoming: TimelineEntry[]
    recent: TimelineEntry[]
    possibly_overdue: TimelineEntry[]
    phase_text: string
  }
  cutoff: { slide: number; title: string; kind: string; message: string } | null
  excluded_slides: SlideRecord[]
  slides: SlideRecord[]
  slides_total: number
  slides_analyzed: number
  dropped_out_of_window: number
  projects: ProjectBrief[]
  insights: SlideInsight[]
  elapsed_ms: number
}

export interface PptProgress {
  running: boolean
  task?: 'generate' | 'compare'
  stage?: string
  current?: number
  total?: number
  message?: string
}

export interface UnmatchedInsight {
  id: string
  ref: string
  text: string
  slides: number[]
  project: string
  status: InsightStatus
  priority: number
}
