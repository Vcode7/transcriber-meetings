/**
 * Global Background Recording Service.
 *
 * Manages MediaRecorder, MediaStream, Web Audio AudioContext, AnalyserNode,
 * and timer intervals globally so recording continues uninterrupted when
 * navigating between pages.
 *
 * Auto-persists audio chunks to IndexedDB in real-time for crash recovery.
 */

import api from '../api/client'
import {
  saveSessionMeta,
  saveRecordingChunk,
  clearRecordingSession,
  getRecoverableSession,
  type SessionMeta,
} from './recordingDb'

export type RecordingState = 'idle' | 'recording' | 'paused' | 'stopped'
export type RecordingType = 'mic' | 'tab'

const CHUNK_INTERVAL_SEC = 600 // 10 minutes

export interface RecordingServiceSnapshot {
  state: RecordingState
  recordingType: RecordingType | null
  duration: number
  audioBlob: Blob | null
  audioUrl: string | null
  analyser: AnalyserNode | null
  error: string | null
  tabLabel: string | null
  includeMic: boolean
  latestChunk: Blob | null
  chunkIds: string[]
  isChunked: boolean
  recoveredSession: {
    sessionId: string
    type: 'mic' | 'tab'
    duration: number
    blob: Blob
    url: string
  } | null
}

type Listener = (snapshot: RecordingServiceSnapshot) => void

class RecordingService {
  private state: RecordingState = 'idle'
  private recordingType: RecordingType | null = null
  private duration: number = 0
  private startTime: number = 0
  private pauseStartTime: number | null = null
  private totalPausedMs: number = 0

  private mediaRecorder: MediaRecorder | null = null
  private stream: MediaStream | null = null
  private micStream: MediaStream | null = null
  private audioCtx: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private dest: MediaStreamAudioDestinationNode | null = null

  private chunks: Blob[] = []
  private headerChunk: Blob | null = null
  private latestChunk: Blob | null = null
  private audioBlob: Blob | null = null
  private audioUrl: string | null = null
  private error: string | null = null
  private tabLabel: string | null = null
  private includeMic: boolean = false
  private mimeType: string = 'audio/webm'
  private sessionId: string | null = null

  private advancedOpts: { meetingPrompt?: string; useVocabularyInPrompt?: boolean } | null = null

  private timerId: ReturnType<typeof setInterval> | null = null

  // Background 10-minute chunking refs
  private nextChunkBlobIndex: number = 0
  private chunkIds: string[] = []
  private chunkIndex: number = 0
  private lastChunkSubmittedAt: number = 0
  private submittingChunk: boolean = false

  // Crash recovery state
  private recoveredSession: {
    sessionId: string
    type: 'mic' | 'tab'
    duration: number
    blob: Blob
    url: string
  } | null = null

  private listeners: Set<Listener> = new Set()

  constructor() {
    this.setupBeforeUnload()
  }

  public subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    listener(this.getSnapshot())
    return () => this.listeners.delete(listener)
  }

  private notify() {
    const snapshot = this.getSnapshot()
    this.listeners.forEach((fn) => fn(snapshot))
  }

  public getSnapshot(): RecordingServiceSnapshot {
    return {
      state: this.state,
      recordingType: this.recordingType,
      duration: this.duration,
      audioBlob: this.audioBlob,
      audioUrl: this.audioUrl,
      analyser: this.analyser,
      error: this.error,
      tabLabel: this.tabLabel,
      includeMic: this.includeMic,
      latestChunk: this.latestChunk,
      chunkIds: [...this.chunkIds],
      isChunked: this.chunkIds.length > 0,
      recoveredSession: this.recoveredSession,
    }
  }

  public setIncludeMic(val: boolean) {
    this.includeMic = val
    this.notify()
  }

  private setupBeforeUnload() {
    if (typeof window === 'undefined') return
    window.addEventListener('beforeunload', (e) => {
      if (this.state === 'recording' || this.state === 'paused') {
        e.preventDefault()
        e.returnValue = 'Recording is in progress. Are you sure you want to leave?'
      }
    })
  }

  private clearTimer() {
    if (this.timerId) {
      clearInterval(this.timerId)
      this.timerId = null
    }
  }

  // â”€â”€ 1. Start Microphone Recording â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  public async startMic(opts?: { meetingPrompt?: string; useVocabularyInPrompt?: boolean }) {
    if (this.state === 'recording' || this.state === 'paused') return

    this.resetState()
    this.recordingType = 'mic'
    this.advancedOpts = opts || null
    this.sessionId = 'mic_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7)
    this.startTime = Date.now()
    this.totalPausedMs = 0

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.stream = stream

      const ctx = new AudioContext()
      this.audioCtx = ctx
      const src = ctx.createMediaStreamSource(stream)
      const ana = ctx.createAnalyser()
      ana.fftSize = 256
      src.connect(ana)
      this.analyser = ana

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      this.mimeType = mimeType

      const mr = new MediaRecorder(stream, { mimeType })
      this.mediaRecorder = mr

      this.setupRecorderHandlers(mr, mimeType)

      mr.start(1000) // 1-second timeslice
      this.state = 'recording'
      this.startTimer()

      await saveSessionMeta({
        sessionId: this.sessionId,
        type: 'mic',
        startTime: this.startTime,
        duration: 0,
        state: 'recording',
        advancedOpts: this.advancedOpts,
        mimeType,
        updatedAt: Date.now(),
      })

      this.notify()
    } catch (e: unknown) {
      this.error = e instanceof Error ? e.message : 'Microphone access denied'
      this.state = 'idle'
      this.notify()
    }
  }

  // â”€â”€ 2. Start Tab Audio Recording â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  public async startTab(includeMicOption: boolean = false) {
    if (this.state === 'recording' || this.state === 'paused') return

    this.resetState()
    this.recordingType = 'tab'
    this.includeMic = includeMicOption
    this.sessionId = 'tab_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7)
    this.startTime = Date.now()
    this.totalPausedMs = 0

    if (!navigator.mediaDevices?.getDisplayMedia) {
      this.error = 'Tab audio capture is not supported in this browser.'
      this.notify()
      return
    }

    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
      } as DisplayMediaStreamOptions)

      this.stream = displayStream

      const audioTracks = displayStream.getAudioTracks()
      if (audioTracks.length === 0) {
        displayStream.getTracks().forEach((t) => t.stop())
        this.error = 'No audio track detected. Please check "Share tab audio" in the browser dialog and try again.'
        this.notify()
        return
      }

      displayStream.getVideoTracks().forEach((t) => t.stop())

      const trackLabel = audioTracks[0].label
      this.tabLabel = trackLabel || 'Browser Tab'

      audioTracks[0].onended = () => {
        if (this.state === 'recording' || this.state === 'paused') {
          this.stop()
        }
      }

      const ctx = new AudioContext()
      this.audioCtx = ctx
      const dest = ctx.createMediaStreamDestination()
      this.dest = dest

      const tabSource = ctx.createMediaStreamSource(new MediaStream(audioTracks))
      tabSource.connect(dest)

      if (this.includeMic) {
        try {
          const micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
          this.micStream = micStream
          const micSource = ctx.createMediaStreamSource(micStream)
          micSource.connect(dest)
        } catch (micErr) {
          console.warn('[RecordingService] Mic access denied for tab mix:', micErr)
        }
      }

      const ana = ctx.createAnalyser()
      ana.fftSize = 256
      tabSource.connect(ana)
      this.analyser = ana

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      this.mimeType = mimeType

      const mr = new MediaRecorder(dest.stream, { mimeType })
      this.mediaRecorder = mr

      this.setupRecorderHandlers(mr, mimeType)

      mr.start(1000)
      this.state = 'recording'
      this.startTimer()

      await saveSessionMeta({
        sessionId: this.sessionId,
        type: 'tab',
        startTime: this.startTime,
        duration: 0,
        state: 'recording',
        tabLabel: this.tabLabel,
        includeMic: this.includeMic,
        mimeType,
        updatedAt: Date.now(),
      })

      this.notify()
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === 'NotAllowedError') {
        this.error = 'Tab sharing was cancelled. Click "Share Tab Audio" to try again.'
      } else {
        this.error = e instanceof Error ? e.message : 'Failed to capture tab audio.'
      }
      this.state = 'idle'
      this.notify()
    }
  }

  // â”€â”€ Recorder Handlers Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  private setupRecorderHandlers(mr: MediaRecorder, mimeType: string) {
    let chunkIndexCounter = 0

    mr.ondataavailable = (e) => {
      if (e.data.size > 0) {
        this.chunks.push(e.data)

        if (this.headerChunk === null) {
          this.headerChunk = e.data
          this.latestChunk = new Blob([e.data], { type: mimeType })
        } else {
          this.latestChunk = new Blob([this.headerChunk, e.data], { type: mimeType })
        }

        // Auto-persist chunk to IndexedDB in real time
        if (this.sessionId) {
          saveRecordingChunk(this.sessionId, chunkIndexCounter++, e.data).catch(() => {})
        }
      }
    }

    mr.onstop = () => {
      const blob = new Blob(this.chunks, { type: mimeType })
      const url = URL.createObjectURL(blob)
      this.audioBlob = blob
      this.audioUrl = url
      this.state = 'stopped'

      // Clean up IndexedDB active session upon explicit completion
      clearRecordingSession().catch(() => {})

      this.notify()
    }
  }

  // â”€â”€ Timer Interval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  private startTimer() {
    this.clearTimer()
    this.timerId = setInterval(async () => {
      if (this.state !== 'recording') return

      const now = Date.now()
      const elapsedSec = Math.floor((now - this.startTime - this.totalPausedMs) / 1000)
      this.duration = Math.max(0, elapsedSec)

      // Auto-submit 10-minute background chunks for mic recording
      if (
        this.recordingType === 'mic' &&
        this.duration >= CHUNK_INTERVAL_SEC &&
        this.duration - this.lastChunkSubmittedAt >= CHUNK_INTERVAL_SEC &&
        !this.submittingChunk &&
        this.headerChunk !== null
      ) {
        const chunkStart = this.lastChunkSubmittedAt
        const chunkEnd = this.duration
        this.lastChunkSubmittedAt = this.duration

        const blobStart = this.nextChunkBlobIndex
        const blobsSnapshot = this.chunks.slice(blobStart)
        this.nextChunkBlobIndex = this.chunks.length

        this.submitBackgroundChunk(blobsSnapshot, chunkStart, chunkEnd).catch(() => {})
      }

      // Update session meta duration in IndexedDB periodically
      if (this.sessionId && this.duration % 5 === 0) {
        saveSessionMeta({
          sessionId: this.sessionId,
          type: this.recordingType || 'mic',
          startTime: this.startTime,
          duration: this.duration,
          state: this.state,
          tabLabel: this.tabLabel,
          includeMic: this.includeMic,
          advancedOpts: this.advancedOpts,
          mimeType: this.mimeType,
          updatedAt: Date.now(),
        }).catch(() => {})
      }

      this.notify()
    }, 1000)
  }

  // â”€â”€ Background Chunk Submission (10-min splits) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  private async submitBackgroundChunk(blobs: Blob[], chunkStartSec: number, chunkEndSec: number) {
    if (this.submittingChunk || blobs.length === 0 || !this.headerChunk) return
    this.submittingChunk = true
    const currIndex = this.chunkIndex
    this.chunkIndex += 1

    try {
      const chunkBlob = new Blob([this.headerChunk, ...blobs.slice(1)], { type: this.mimeType })
      const form = new FormData()
      form.append('file', chunkBlob, `chunk_${currIndex}.webm`)
      form.append('chunk_index', String(currIndex))
      form.append('chunk_start_sec', String(chunkStartSec))
      form.append('chunk_end_sec', String(chunkEndSec))
      if (this.advancedOpts?.meetingPrompt) form.append('meeting_prompt', this.advancedOpts.meetingPrompt)
      form.append('use_vocabulary', this.advancedOpts?.useVocabularyInPrompt ? 'true' : 'false')

      const res = await api.post('/audio/chunk', form)
      const { chunk_id } = res.data
      this.chunkIds.push(chunk_id)
      console.log(`[RecordingService] Chunk ${currIndex} submitted â†’ ${chunk_id}`)
    } catch (e) {
      console.error('[RecordingService] Chunk submission error:', e)
    } finally {
      this.submittingChunk = false
    }
  }

  // â”€â”€ 3. Pause Recording â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  public pause() {
    if (this.state !== 'recording' || !this.mediaRecorder) return
    try {
      if (this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.pause()
      }
      this.state = 'paused'
      this.pauseStartTime = Date.now()
      this.notify()

      if (this.sessionId) {
        saveSessionMeta({
          sessionId: this.sessionId,
          type: this.recordingType || 'mic',
          startTime: this.startTime,
          duration: this.duration,
          state: 'paused',
          tabLabel: this.tabLabel,
          includeMic: this.includeMic,
          advancedOpts: this.advancedOpts,
          mimeType: this.mimeType,
          updatedAt: Date.now(),
        }).catch(() => {})
      }
    } catch (e) {
      console.error('[RecordingService] Pause failed:', e)
    }
  }

  // â”€â”€ 4. Resume Recording â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  public resume() {
    if (this.state !== 'paused' || !this.mediaRecorder) return
    try {
      if (this.pauseStartTime) {
        this.totalPausedMs += Date.now() - this.pauseStartTime
        this.pauseStartTime = null
      }
      if (this.mediaRecorder.state === 'paused') {
        this.mediaRecorder.resume()
      }
      this.state = 'recording'
      this.notify()

      if (this.sessionId) {
        saveSessionMeta({
          sessionId: this.sessionId,
          type: this.recordingType || 'mic',
          startTime: this.startTime,
          duration: this.duration,
          state: 'recording',
          tabLabel: this.tabLabel,
          includeMic: this.includeMic,
          advancedOpts: this.advancedOpts,
          mimeType: this.mimeType,
          updatedAt: Date.now(),
        }).catch(() => {})
      }
    } catch (e) {
      console.error('[RecordingService] Resume failed:', e)
    }
  }

  // ── 5. Stop Recording ────────────────────────────────────────
  public stop() {
    this.clearTimer()
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop()
      } catch {}
    }

    const currentStream = this.stream
    const currentMicStream = this.micStream
    const currentAudioCtx = this.audioCtx

    this.stream = null
    this.micStream = null
    this.audioCtx = null
    this.dest = null
    this.analyser = null

    setTimeout(() => {
      try {
        currentStream?.getTracks().forEach((t) => {
          t.stop()
          t.enabled = false
        })
        currentMicStream?.getTracks().forEach((t) => {
          t.stop()
          t.enabled = false
        })
        if (currentAudioCtx && currentAudioCtx.state !== 'closed') {
          currentAudioCtx.close().catch(() => {})
        }
      } catch (e) {
        console.warn('[RecordingService] Error cleaning up audio tracks:', e)
      }
    }, 150)
  }

  // ── 6. Reset Recording State ─────────────────────────────────
  public reset() {
    this.stop()
    this.resetState()
    clearRecordingSession().catch(() => {})
    this.notify()
  }

  public cancel() {
    this.reset()
  }

  private resetState() {
    this.clearTimer()
    this.state = 'idle'
    this.recordingType = null
    this.duration = 0
    this.startTime = 0
    this.pauseStartTime = null
    this.totalPausedMs = 0
    this.chunks = []
    this.headerChunk = null
    this.latestChunk = null
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl)
    }
    this.audioBlob = null
    this.audioUrl = null
    this.error = null
    this.tabLabel = null
    this.nextChunkBlobIndex = 0
    this.chunkIds = []
    this.chunkIndex = 0
    this.lastChunkSubmittedAt = 0
    this.submittingChunk = false
    this.sessionId = null
  }

  // â”€â”€ 7. Check & Recover Un-submitted Crash Session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  public async checkCrashRecovery() {
    if (this.state === 'recording' || this.state === 'paused') return
    try {
      const result = await getRecoverableSession()
      if (result) {
        const { meta, recoveredBlob } = result
        const url = URL.createObjectURL(recoveredBlob)
        this.recoveredSession = {
          sessionId: meta.sessionId,
          type: meta.type,
          duration: meta.duration || 0,
          blob: recoveredBlob,
          url,
        }
        console.log(`[RecordingService] Recovered crash session (${meta.duration}s, ${(recoveredBlob.size / 1024 / 1024).toFixed(2)} MB)`)
        this.notify()
      }
    } catch (e) {
      console.warn('[RecordingService] Crash recovery check error:', e)
    }
  }

  public clearCrashRecovery() {
    if (this.recoveredSession?.url) {
      URL.revokeObjectURL(this.recoveredSession.url)
    }
    this.recoveredSession = null
    clearRecordingSession().catch(() => {})
    this.notify()
  }
}

export const recordingService = new RecordingService()

