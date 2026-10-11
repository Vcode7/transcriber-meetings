import { isAxiosError } from 'axios'

export function getApiErrorDetail(error: unknown, fallback: string): string
export function getApiErrorDetail(error: unknown): string | undefined
export function getApiErrorDetail(error: unknown, fallback?: string): string | undefined {
  if (isAxiosError(error)) {
    const data = error.response?.data
    if (data && typeof data === 'object' && 'detail' in data) {
      const detail = data.detail
      if (typeof detail === 'string') return detail
    }
    if (error.message) return error.message
  }

  if (error instanceof Error && error.message) return error.message
  return fallback
}

