/**
 * Collection AI Chat API — SSE streaming + REST endpoints.
 */
import api from './client'
import { useAuthStore } from '../store/auth'
import type { ChatMessage } from '../types/recording'

const BASE_URL = 'http://127.0.0.1:8000'

function getAuthHeaders(): Record<string, string> {
  const token = useAuthStore.getState().accessToken
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * Stream a chat response via SSE using fetch + ReadableStream.
 * Calls onChunk for each text chunk, onMeta for metadata, onDone when complete.
 */
export async function streamChat(
  collectionId: string,
  message: string,
  onChunk: (text: string) => void,
  onMeta?: (meta: { cited_meetings?: string[]; meeting_names?: Record<string, string>; retrieval_plan?: string; max_context?: number }) => void,
  onError?: (error: string) => void,
  onDone?: () => void,
  signal?: AbortSignal,
  maxContext?: number,
): Promise<void> {
  return _streamSSE(
    `${BASE_URL}/collections/${collectionId}/ai/chat`,
    { message, max_context: maxContext },
    onChunk,
    onMeta,
    onError,
    onDone,
    signal,
  )
}

/**
 * Stream a meeting comparison report via SSE.
 */
export async function streamComparison(
  collectionId: string,
  meetingIdA: string,
  meetingIdB: string,
  onChunk: (text: string) => void,
  onMeta?: (meta: { cited_meetings?: string[]; meeting_names?: Record<string, string> }) => void,
  onError?: (error: string) => void,
  onDone?: () => void,
  signal?: AbortSignal,
): Promise<void> {
  return _streamSSE(
    `${BASE_URL}/collections/${collectionId}/ai/compare`,
    { meeting_id_a: meetingIdA, meeting_id_b: meetingIdB },
    onChunk,
    onMeta,
    onError,
    onDone,
    signal,
  )
}

/**
 * Stream a topic growth report via SSE.
 */
export async function streamTopicGrowth(
  collectionId: string,
  topic: string,
  onChunk: (text: string) => void,
  onMeta?: (meta: { cited_meetings?: string[]; meeting_names?: Record<string, string> }) => void,
  onError?: (error: string) => void,
  onDone?: () => void,
  signal?: AbortSignal,
): Promise<void> {
  return _streamSSE(
    `${BASE_URL}/collections/${collectionId}/ai/topic-growth`,
    { topic },
    onChunk,
    onMeta,
    onError,
    onDone,
    signal,
  )
}

/**
 * Get chat history for a collection.
 */
export async function getChatHistory(collectionId: string): Promise<ChatMessage[]> {
  const res = await api.get(`/collections/${collectionId}/ai/history`)
  return res.data
}

/**
 * Clear all chat history for a collection.
 */
export async function clearChatHistory(collectionId: string): Promise<void> {
  await api.delete(`/collections/${collectionId}/ai/history`)
}

/**
 * Export report as a downloadable Markdown file.
 */
export async function exportReport(collectionId: string, content: string, filename: string = 'report'): Promise<void> {
  const res = await api.post(
    `/collections/${collectionId}/ai/export`,
    { content, filename },
    { responseType: 'blob' },
  )
  // Trigger download
  const blob = new Blob([res.data], { type: 'text/markdown; charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${filename}.md`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
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

      // Parse SSE events from buffer
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete last line in buffer

      let currentEvent = ''
      let currentData = ''

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          currentData = line.slice(6)

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
            // JSON parse failure for a chunk — skip
          }

          currentEvent = ''
          currentData = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  onDone?.()
}
