/**
 * IndexedDB helper for real-time audio chunk persistence and crash recovery.
 *
 * Stores chunks and session metadata as audio is recorded.
 * If the app or browser closes unexpectedly during recording,
 * the recorded chunks can be recovered upon restart.
 */

const DB_NAME = 'voicesum_recording_db'
const DB_VERSION = 1
const STORE_SESSION = 'active_session'
const STORE_CHUNKS = 'recording_chunks'

export interface SessionMeta {
  sessionId: string
  type: 'mic' | 'tab'
  startTime: number
  duration: number
  state: 'recording' | 'paused' | 'stopped'
  tabLabel?: string | null
  includeMic?: boolean
  advancedOpts?: { meetingPrompt?: string; useVocabularyInPrompt?: boolean } | null
  mimeType: string
  updatedAt: number
}

export interface ChunkRecord {
  id?: number
  sessionId: string
  chunkIndex: number
  blob: Blob
  timestamp: number
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result

      if (!db.objectStoreNames.contains(STORE_SESSION)) {
        db.createObjectStore(STORE_SESSION, { keyPath: 'id' })
      }

      if (!db.objectStoreNames.contains(STORE_CHUNKS)) {
        const chunkStore = db.createObjectStore(STORE_CHUNKS, {
          keyPath: 'id',
          autoIncrement: true,
        })
        chunkStore.createIndex('sessionId', 'sessionId', { unique: false })
      }
    }

    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

/** Save or update active session metadata */
export async function saveSessionMeta(meta: SessionMeta): Promise<void> {
  try {
    const db = await openDb()
    const tx = db.transaction(STORE_SESSION, 'readwrite')
    const store = tx.objectStore(STORE_SESSION)
    store.put({ id: 'current_session', ...meta })
    await new Promise((res, rej) => {
      tx.oncomplete = res
      tx.onerror = rej
    })
  } catch (err) {
    console.warn('[RecordingDb] Failed to save session meta:', err)
  }
}

/** Save an audio chunk to IndexedDB */
export async function saveRecordingChunk(
  sessionId: string,
  chunkIndex: number,
  blob: Blob
): Promise<void> {
  try {
    const db = await openDb()
    const tx = db.transaction(STORE_CHUNKS, 'readwrite')
    const store = tx.objectStore(STORE_CHUNKS)
    store.add({
      sessionId,
      chunkIndex,
      blob,
      timestamp: Date.now(),
    })
    await new Promise((res, rej) => {
      tx.oncomplete = res
      tx.onerror = rej
    })
  } catch (err) {
    console.warn('[RecordingDb] Failed to save chunk:', err)
  }
}

/** Clear active session and its chunks */
export async function clearRecordingSession(): Promise<void> {
  try {
    const db = await openDb()
    const tx = db.transaction([STORE_SESSION, STORE_CHUNKS], 'readwrite')
    tx.objectStore(STORE_SESSION).delete('current_session')
    tx.objectStore(STORE_CHUNKS).clear()
    await new Promise((res, rej) => {
      tx.oncomplete = res
      tx.onerror = rej
    })
  } catch (err) {
    console.warn('[RecordingDb] Failed to clear session:', err)
  }
}

/** Check if there is an un-submitted recoverable session from a crash/reload */
export async function getRecoverableSession(): Promise<{
  meta: SessionMeta
  recoveredBlob: Blob
} | null> {
  try {
    const db = await openDb()

    // 1. Fetch active session meta
    const sessionMeta = await new Promise<SessionMeta | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE_SESSION, 'readonly')
      const req = tx.objectStore(STORE_SESSION).get('current_session')
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })

    if (!sessionMeta || !sessionMeta.sessionId) return null

    // 2. Fetch all chunks for this session
    const chunks = await new Promise<ChunkRecord[]>((resolve, reject) => {
      const tx = db.transaction(STORE_CHUNKS, 'readonly')
      const index = tx.objectStore(STORE_CHUNKS).index('sessionId')
      const req = index.getAll(sessionMeta.sessionId)
      req.onsuccess = () => resolve(req.result || [])
      req.onerror = () => reject(req.error)
    })

    if (chunks.length === 0) return null

    // Sort chunks by chunkIndex
    chunks.sort((a, b) => a.chunkIndex - b.chunkIndex)

    const blobs = chunks.map((c) => c.blob)
    const recoveredBlob = new Blob(blobs, { type: sessionMeta.mimeType || 'audio/webm' })

    if (recoveredBlob.size < 1000) return null

    return { meta: sessionMeta, recoveredBlob }
  } catch (err) {
    console.warn('[RecordingDb] Failed to recover session:', err)
    return null
  }
}
