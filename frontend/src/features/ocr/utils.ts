import { isAxiosError } from 'axios'
import type { OcrFieldSchema, OcrSchema, OcrSettings, OcrSettingValue } from './types'

export type FieldErrors = Record<string, string>

/** Field is shown only when its parent setting has one of the listed values. */
export function isFieldVisible(field: OcrFieldSchema, values: OcrSettings): boolean {
  if (!field.depends_on) return true
  return field.depends_on.values.includes(values[field.depends_on.field])
}

/** Client-side checks mirroring the server's validators (the server stays authoritative). */
export function validateSettings(values: OcrSettings, schema: OcrSchema | null): FieldErrors {
  const errors: FieldErrors = {}
  if (!schema) return errors
  for (const field of schema.fields) {
    const value = values[field.name]
    if (field.type === 'int' || field.type === 'float') {
      if (typeof value !== 'number' || Number.isNaN(value)) {
        errors[field.name] = 'Enter a number'
        continue
      }
      if (field.type === 'int' && !Number.isInteger(value)) {
        errors[field.name] = 'Must be a whole number'
      } else if (field.min !== undefined && value < field.min) {
        errors[field.name] = `Must be at least ${field.min}`
      } else if (field.max !== undefined && value > field.max) {
        errors[field.name] = `Must be at most ${field.max}`
      }
    } else if (field.type === 'enum' && field.options && !field.options.some(o => o.value === value)) {
      errors[field.name] = 'Choose one of the options'
    }
  }
  const block = values.binarize_block_size
  if (!errors.binarize_block_size && typeof block === 'number' && block % 2 === 0) {
    errors.binarize_block_size = 'Must be an odd number'
  }
  if (
    !errors.det_limit_side_len &&
    values.det_limit_type === 'max' &&
    Number(values.det_limit_side_len) > Number(values.det_max_side_limit)
  ) {
    errors.det_limit_side_len = 'Cannot exceed the maximum side length in max resize mode'
  }
  return errors
}

/** Map a FastAPI 422 response to field errors; returns {} when it isn't one. */
export function serverFieldErrors(error: unknown): FieldErrors {
  if (!isAxiosError(error) || error.response?.status !== 422) return {}
  const detail = (error.response.data as { detail?: unknown })?.detail
  if (!Array.isArray(detail)) return {}
  const errors: FieldErrors = {}
  for (const item of detail) {
    const loc: unknown[] = Array.isArray(item?.loc) ? item.loc : []
    const field = loc.filter(p => typeof p === 'string' && p !== 'body' && p !== 'settings').pop()
    const msg = String(item?.msg ?? 'Invalid value').replace(/^Value error, /, '')
    errors[typeof field === 'string' ? field : '_form'] = msg
  }
  return errors
}

export function settingsEqual(a: OcrSettings | null, b: OcrSettings | null): boolean {
  if (!a || !b) return a === b
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  for (const k of keys) if (a[k] !== b[k]) return false
  return true
}

export function formatValue(field: OcrFieldSchema, value: OcrSettingValue | undefined): string {
  if (value === undefined) return '—'
  if (field.type === 'bool') return value ? 'On' : 'Off'
  if (field.type === 'enum') return field.options?.find(o => o.value === value)?.label ?? String(value)
  if (typeof value === 'number') {
    const text = field.type === 'float' ? String(Number(value.toFixed(3))) : value.toLocaleString()
    return field.unit ? `${text} ${field.unit}` : text
  }
  return String(value)
}

export function formatBytes(bytes: number | undefined | null): string {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

export function formatMs(ms: number | undefined | null): string {
  if (ms === undefined || ms === null) return '—'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export function copyText(text: string): Promise<void> {
  return navigator.clipboard.writeText(text)
}

export function downloadText(text: string, filename: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
