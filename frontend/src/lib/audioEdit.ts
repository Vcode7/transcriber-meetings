/**
 * Audio trim / middle-cut selection — the pure logic behind AudioTrimmer and
 * the Upload / Record / Video upload pages.
 *
 * The backend (routers/audio.py `build_edit_request`) treats each trim
 * boundary independently: an omitted start means "from the beginning" and an
 * omitted end means "to the true end of the decoded audio". So only the
 * boundaries the user actually moved are sent. The browser's duration is only
 * an estimate for some formats (MediaRecorder WebM, VBR MP3), and sending it
 * as the end point would silently clip the tail of the recording.
 */

export interface AudioEditSelection {
  trimStartSec?: number
  trimEndSec?: number
  cutStartSec?: number
  cutEndSec?: number
  /** Duration the browser reported; lets the server rescale if it was an estimate. */
  sourceDurationSec?: number
}

export interface TrimState {
  duration: number
  trimStart: number
  trimEnd: number
  cutEnabled: boolean
  cutStart: number
  cutEnd: number
}

export type TrimHandle = 'start' | 'end' | 'cut-start' | 'cut-end'

export const MIN_KEEP_SEC = 0.5    // shortest trim window
export const MIN_CUT_SEC = 0.3     // shortest removable section
export const CUT_MARGIN_SEC = 0.1  // cut handles stay this far inside the trim window
export const EDGE_EPS_SEC = 0.05   // within this of a file edge counts as "at the edge"

const clamp = (t: number, lo: number, hi: number) => Math.min(Math.max(t, lo), Math.max(lo, hi))

export const roundTenth = (sec: number) => Math.round(sec * 10) / 10

/** Whole-second clock for durations: 75 → "1:15", 3725 → "1:02:05". */
export function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

/** Tenth-of-a-second clock for edit points: 83.44 → "1:23.4". Round-trips through parseTime. */
export function fmtTimePrecise(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00.0'
  const tenths = Math.round(sec * 10)
  const h = Math.floor(tenths / 36000)
  const m = Math.floor((tenths % 36000) / 600)
  const s = (tenths % 600) / 10
  const ss = s.toFixed(1).padStart(4, '0')
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${ss}`
  return `${m}:${ss}`
}

/** "83.4", "1:23.4" or "1:02:03.5" → seconds; null for anything else. */
export function parseTime(v: string): number | null {
  if (typeof v !== 'string') return null
  const parts = v.trim().split(':')
  if (parts.length === 0 || parts.length > 3 || parts.some((p) => p.trim() === '')) return null
  const nums = parts.map(Number)
  if (nums.some((n) => !Number.isFinite(n) || n < 0)) return null
  return nums.reduce((acc, n) => acc * 60 + n, 0)
}

export function initialTrimState(duration: number): TrimState {
  const d = Number.isFinite(duration) && duration > 0 ? duration : 0
  return { duration: d, trimStart: 0, trimEnd: d, cutEnabled: false, cutStart: d * 0.35, cutEnd: d * 0.65 }
}

/**
 * The browser refined its duration estimate (e.g. a WebM stream finished
 * probing). An end handle the user left at the end follows the new duration.
 */
export function withDuration(prev: TrimState, duration: number): TrimState {
  if (!(Number.isFinite(duration) && duration > 0)) return prev
  if (prev.duration <= 0) return initialTrimState(duration)
  const endAtEdge = prev.trimEnd >= prev.duration - EDGE_EPS_SEC
  const trimEnd = endAtEdge ? duration : Math.min(prev.trimEnd, duration)
  const trimStart = Math.min(prev.trimStart, Math.max(0, trimEnd - MIN_KEEP_SEC))
  return { ...prev, duration, trimStart, trimEnd }
}

/** Where a handle may go, given the other handles: the same rules for dragging and typing. */
export function clampHandle(handle: TrimHandle, t: number, s: TrimState): number {
  const v = roundTenth(t)
  switch (handle) {
    case 'start': {
      const max = s.cutEnabled ? Math.min(s.trimEnd - MIN_KEEP_SEC, s.cutStart - CUT_MARGIN_SEC) : s.trimEnd - MIN_KEEP_SEC
      return clamp(v, 0, max)
    }
    case 'end': {
      const min = s.cutEnabled ? Math.max(s.trimStart + MIN_KEEP_SEC, s.cutEnd + CUT_MARGIN_SEC) : s.trimStart + MIN_KEEP_SEC
      return clamp(v, min, s.duration)
    }
    case 'cut-start':
      return clamp(v, s.trimStart + CUT_MARGIN_SEC, s.cutEnd - MIN_CUT_SEC)
    case 'cut-end':
      return clamp(v, s.cutStart + MIN_CUT_SEC, s.trimEnd - CUT_MARGIN_SEC)
  }
}

export function moveHandle(s: TrimState, handle: TrimHandle, t: number): TrimState {
  const v = clampHandle(handle, t, s)
  switch (handle) {
    case 'start': return { ...s, trimStart: v }
    case 'end': return { ...s, trimEnd: v }
    case 'cut-start': return { ...s, cutStart: v }
    case 'cut-end': return { ...s, cutEnd: v }
  }
}

/** Turn the middle cut on/off; a fresh cut sits in the middle 40% of the trim window. */
export function toggleCut(s: TrimState): TrimState {
  if (s.cutEnabled) return { ...s, cutEnabled: false }
  const inside = s.cutStart >= s.trimStart + CUT_MARGIN_SEC
    && s.cutEnd <= s.trimEnd - CUT_MARGIN_SEC
    && s.cutEnd - s.cutStart >= MIN_CUT_SEC
  if (inside) return { ...s, cutEnabled: true }
  const span = s.trimEnd - s.trimStart
  return {
    ...s,
    cutEnabled: true,
    cutStart: roundTenth(s.trimStart + span * 0.3),
    cutEnd: roundTenth(s.trimStart + span * 0.7),
  }
}

/** The part of the cut that actually falls inside the trim window, or null. */
export function effectiveCut(s: TrimState): [number, number] | null {
  if (!s.cutEnabled) return null
  const a = Math.max(s.cutStart, s.trimStart)
  const b = Math.min(s.cutEnd, s.trimEnd)
  return b - a > 0.01 ? [a, b] : null
}

export function keptDuration(s: TrimState): number {
  const cut = effectiveCut(s)
  return Math.max(0, s.trimEnd - s.trimStart - (cut ? cut[1] - cut[0] : 0))
}

/** What to send to the server, or null when the selection is the whole file. */
export function buildEditSelection(s: TrimState): AudioEditSelection | null {
  if (!(s.duration > 0)) return null
  const sel: AudioEditSelection = {}
  if (s.trimStart > EDGE_EPS_SEC) sel.trimStartSec = s.trimStart
  if (s.trimEnd < s.duration - EDGE_EPS_SEC) sel.trimEndSec = s.trimEnd
  const cut = effectiveCut(s)
  if (cut) {
    sel.cutStartSec = cut[0]
    sel.cutEndSec = cut[1]
  }
  if (sel.trimStartSec === undefined && sel.trimEndSec === undefined && !cut) return null
  sel.sourceDurationSec = s.duration
  return sel
}

/** Add the selection's fields to an upload form (no-op for "whole file"). */
export function appendAudioEdit(form: FormData, edit?: AudioEditSelection | null): void {
  if (!edit) return
  const hasCut = edit.cutStartSec !== undefined && edit.cutEndSec !== undefined && edit.cutEndSec > edit.cutStartSec
  if (edit.trimStartSec === undefined && edit.trimEndSec === undefined && !hasCut) return
  const put = (key: string, v?: number) => {
    if (v !== undefined && Number.isFinite(v) && v >= 0) form.append(key, v.toFixed(3))
  }
  put('trim_start_sec', edit.trimStartSec)
  put('trim_end_sec', edit.trimEndSec)
  if (hasCut) {
    put('cut_start_sec', edit.cutStartSec)
    put('cut_end_sec', edit.cutEndSec)
  }
  put('source_duration_sec', edit.sourceDurationSec)
}
