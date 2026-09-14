/**
 * AI Chat API — SSE streaming + REST endpoints for the standalone AI Chat tab.
 */
import { useAuthStore } from '../store/auth'
import type { AIChatMeeting, AIChatMessage } from '../types/recording'

const BASE_URL = 'http://127.0.0.1:8000'

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
): Promise<void> {
  const body: Record<string, unknown> = { message }
  if (selectedMeetingId) {
    body.selected_meeting_id = selectedMeetingId
  }
  if (maxResults !== undefined) {
    body.max_results = maxResults
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
 * Get AI Chat history.
 */
export async function getAIChatHistory(): Promise<AIChatMessage[]> {
  const response = await fetch(`${BASE_URL}/ai-chat/history`, {
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

/**
 * Clear AI Chat history.
 */
export async function clearAIChatHistory(): Promise<void> {
  const response = await fetch(`${BASE_URL}/ai-chat/history`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
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
