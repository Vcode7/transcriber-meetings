/**
 * AI Chat API — SSE streaming + REST endpoints for the standalone AI Chat tab.
 */
import { useAuthStore } from '../store/auth'
import type { AIChatMeeting, AIChatMessage, SyncableMeeting, ResyncResult, ManualMomItem, ManualMomDetail } from '../types/recording'
import { BASE_URL } from '../config'

function getAuthHeaders(): Record<string, string> {
  const token = useAuthStore.getState().accessToken
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * Stream an AI Chat response via SSE.
 */
export async function streamAIChat(
  message: string,
  onChunk: (text: string) => void,
  onMeta?: (meta: {
    meeting_sources?: Array<{ id: string; name: string; date: string }>
    query_type?: 'direct' | 'retrieval'
  }) => void,
  onError?: (error: string) => void,
  onDone?: () => void,
  signal?: AbortSignal,
  selectedMeetingId?: string,
  maxResults?: number,
  sessionId?: string,
): Promise<void> {
  const body: Record<string, unknown> = { message }
  if (selectedMeetingId) {
    body.selected_meeting_id = selectedMeetingId
  }
  if (maxResults !== undefined) {
    body.max_results = maxResults
  }
  if (sessionId) {
    body.session_id = sessionId
  }
  return _streamSSE(
    `${BASE_URL}/ai-chat/send`,
    body,
    onChunk,
    onMeta,
    onError,
    onDone,
    signal,
  )
}

/**
 * Get meetings that have Stage 2 data in ChromaDB.
 */
export async function getAIChatMeetings(): Promise<AIChatMeeting[]> {
  const response = await fetch(`${BASE_URL}/ai-chat/meetings`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Get AI Chat sessions.
 */
export async function getAIChatSessions(): Promise<Array<{
  id: string
  title: string
  selected_meeting_id?: string
  created_at: string
  updated_at: string
}>> {
  const response = await fetch(`${BASE_URL}/ai-chat/sessions`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Create a new AI Chat session.
 */
export async function createAIChatSession(
  title = 'New Chat',
  selectedMeetingId?: string,
  id?: string,
): Promise<{
  id: string
  title: string
  selected_meeting_id?: string
  created_at: string
  updated_at: string
}> {
  const payload: Record<string, unknown> = {
    title,
    selected_meeting_id: selectedMeetingId || null,
  }
  if (id) {
    payload.id = id
  }
  const response = await fetch(`${BASE_URL}/ai-chat/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Update an AI Chat session.
 */
export async function updateAIChatSession(
  sessionId: string,
  updates: { title?: string; selected_meeting_id?: string },
): Promise<void> {
  const response = await fetch(`${BASE_URL}/ai-chat/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(updates),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
}

/**
 * Delete an AI Chat session.
 */
export async function deleteAIChatSession(sessionId: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/ai-chat/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
}

/**
 * Get AI Chat history (optionally filtered by session).
 */
export async function getAIChatHistory(sessionId?: string): Promise<AIChatMessage[]> {
  const url = sessionId
    ? `${BASE_URL}/ai-chat/history?session_id=${encodeURIComponent(sessionId)}`
    : `${BASE_URL}/ai-chat/history`
  const response = await fetch(url, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Clear AI Chat history (optionally filtered by session).
 */
export async function clearAIChatHistory(sessionId?: string): Promise<void> {
  const url = sessionId
    ? `${BASE_URL}/ai-chat/history?session_id=${encodeURIComponent(sessionId)}`
    : `${BASE_URL}/ai-chat/history`
  const response = await fetch(url, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
}

/**
 * Get meetings that have Stage 2 discussion points and their ChromaDB sync status.
 */
export async function getSyncableMeetings(): Promise<SyncableMeeting[]> {
  const response = await fetch(`${BASE_URL}/ai-chat/syncable-meetings`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Resync Stage 2 points for a meeting into ChromaDB.
 */
export async function resyncMeeting(recordingId: string, force = false): Promise<ResyncResult> {
  const response = await fetch(`${BASE_URL}/ai-chat/resync-meeting/${recordingId}?force=${force}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
  })
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}))
    throw new Error(errData.detail || `HTTP ${response.status}`)
  }
  return response.json()
}


// ── Internal SSE streaming helper ────────────────────────────────────────────

async function _streamSSE(
  url: string,
  body: Record<string, unknown>,
  onChunk: (text: string) => void,
  onMeta?: (meta: any) => void,
  onError?: (error: string) => void,
  onDone?: () => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const err = await response.json()
      detail = err.detail || detail
    } catch {
      // ignore
    }
    onError?.(detail)
    onDone?.()
    return
  }

  const reader = response.body?.getReader()
  if (!reader) {
    onError?.('No response stream')
    onDone?.()
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      let currentEvent = ''

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          const currentData = line.slice(6)
          try {
            const parsed = JSON.parse(currentData)
            if (currentEvent === 'chunk' && typeof parsed === 'string') {
              onChunk(parsed)
            } else if (currentEvent === 'metadata') {
              try {
                const meta = typeof parsed === 'string' ? JSON.parse(parsed) : parsed
                onMeta?.(meta)
              } catch {
                // metadata parse failure is non-fatal
              }
            } else if (currentEvent === 'error' && typeof parsed === 'string') {
              onError?.(parsed)
            } else if (currentEvent === 'done') {
              onDone?.()
              return
            }
          } catch {
            // JSON parse failure — skip
          }
          currentEvent = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  onDone?.()
}

// ── Manual MoM API Client ────────────────────────────────────────────────────

export interface UploadManualMomsResponse {
  status: 'success' | 'error'
  results: ManualMomItem[]
  success_count: number
  fail_count: number
  errors: Array<{ filename: string; error: string }>
}

/**
 * Upload multiple MoM files simultaneously.
 */
export async function uploadManualMoms(files: File[]): Promise<UploadManualMomsResponse> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch(`${BASE_URL}/ai-chat/manual-mom/upload`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
    },
    body: formData,
  })

  if (!response.ok) {
    const errText = await response.text().catch(() => '')
    throw new Error(`Upload failed (${response.status}): ${errText || response.statusText}`)
  }

  return response.json()
}

/**
 * Fetch all uploaded manual MoMs.
 */
export async function getManualMoms(): Promise<ManualMomItem[]> {
  const response = await fetch(`${BASE_URL}/ai-chat/manual-mom/list`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Fetch detailed content and metadata for a specific manual MoM.
 */
export async function getManualMomDetail(momId: string): Promise<ManualMomDetail> {
  const response = await fetch(`${BASE_URL}/ai-chat/manual-mom/${momId}`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Delete a manual MoM and remove its vectors from ChromaDB.
 */
export async function deleteManualMom(momId: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/ai-chat/manual-mom/${momId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
}

/**
 * Re-index a manual MoM into ChromaDB.
 */
export async function reindexManualMom(momId: string): Promise<{ status: string; chunks_indexed: number }> {
  const response = await fetch(`${BASE_URL}/ai-chat/manual-mom/${momId}/reindex`, {
    method: 'POST',
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}
