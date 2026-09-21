
import { useAuthStore } from '../store/auth'

const API_BASE = 'http://127.0.0.1:8000'


/**
 * Calls the backend PDF endpoint, receives the file as a blob,
 * and triggers a browser download.
 *
 * @param recordingId  Recording ID
 * @param filename     Display name for the downloaded file (optional)
 * @param formData     Optional FormData with 'images' and 'documents' file fields.
 *                     If omitted, a plain POST is sent (backward compatible).
 * @throws Error with message on failure
 */
export async function downloadPdfReport(
  recordingId: string,
  filename?: string,
  formData?: FormData,
): Promise<void> {
  const token = useAuthStore.getState().accessToken

  const headers: Record<string, string> = {
    Authorization: token ? `Bearer ${token}` : '',
  }

  // Always send FormData so fields like include_transcription are transmitted
  // even when no file attachments are present.
  const body = formData ?? new FormData()

  const response = await fetch(`${API_BASE}/pdf/${recordingId}`, {
    method: 'POST',
    credentials: 'include',
    headers,
    body,
  })

  if (!response.ok) {
    let detail = `Server error ${response.status}`
    try {
      const json = await response.json()
      detail = json.detail || detail
    } catch {
      // non-JSON error body
    }
    throw new Error(detail)
  }

  const blob = await response.blob()

  const rawName = filename?.replace(/\.(wav|mp3|webm|mp4|m4a|ogg|flac)$/i, '') ?? 'meeting'
  const safeName = rawName.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 60) || 'meeting'
  const downloadName = `VoiceSum_Report_${safeName}.pdf`

  const disposition = response.headers.get('Content-Disposition')
  const serverFilename = disposition?.match(/filename="([^"]+)"/)?.[1]

  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = serverFilename || downloadName
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
