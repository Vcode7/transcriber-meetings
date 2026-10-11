import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import {
  AlertCircle, CheckCircle, Cpu, FileSearch, FlaskConical, Loader, RotateCcw, Save, ScanText, SlidersHorizontal, Undo2,
} from 'lucide-react'
import { getApiErrorDetail } from '../lib/errors'
import {
  fetchOcrSettings, fetchOcrStatus, resetOcrSettings, saveOcrSettings,
} from '../features/ocr/api/ocrApi'
import OcrSettingsForm from '../features/ocr/components/OcrSettingsForm'
import OcrTestPanel from '../features/ocr/components/OcrTestPanel'
import VlmOcrPanel from '../features/ocr/components/VlmOcrPanel'
import type { OcrPreset, OcrSchema, OcrSettings as OcrSettingsValues, OcrSettingValue, OcrStatus } from '../features/ocr/types'
import { serverFieldErrors, settingsEqual, validateSettings, type FieldErrors } from '../features/ocr/utils'

type Tab = 'pipeline' | 'test' | 'vlm'

export default function OcrSettings() {
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [schema, setSchema] = useState<OcrSchema | null>(null)
  const [presets, setPresets] = useState<OcrPreset[]>([])
  const [saved, setSaved] = useState<OcrSettingsValues | null>(null)
  const [form, setForm] = useState<OcrSettingsValues | null>(null)
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)
  const [status, setStatus] = useState<OcrStatus | null>(null)
  const [serverErrors, setServerErrors] = useState<FieldErrors>({})
  const [saving, setSaving] = useState(false)
  const [savedFlash, setSavedFlash] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const [tab, setTab] = useState<Tab>('pipeline')
  const [vlmOpened, setVlmOpened] = useState(false)

  // Mount the experimental panel only once it is opened (it queries the Ollama server).
  useEffect(() => { if (tab === 'vlm') setVlmOpened(true) }, [tab])

  const load = useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const [data, engine] = await Promise.all([
        fetchOcrSettings(),
        fetchOcrStatus().catch(() => null),
      ])
      setSchema(data.schema)
      setPresets(data.presets)
      setSaved(data.settings)
      setForm(data.settings)
      setUpdatedAt(data.updated_at)
      setStatus(engine)
    } catch (err) {
      setLoadError(getApiErrorDetail(err, 'Could not load OCR settings.'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const dirty = useMemo(() => !settingsEqual(form, saved), [form, saved])
  const errors = useMemo<FieldErrors>(
    () => ({ ...(form ? validateSettings(form, schema) : {}), ...serverErrors }),
    [form, schema, serverErrors],
  )
  const hasErrors = Object.keys(errors).length > 0

  useEffect(() => {
    if (!dirty) return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const updateField = (name: string, value: OcrSettingValue) => {
    setForm(prev => (prev ? { ...prev, [name]: value } : prev))
    setServerErrors(prev => {
      if (!(name in prev) && !('_form' in prev)) return prev
      const next = { ...prev }
      delete next[name]
      delete next._form
      return next
    })
  }

  const applyPreset = (preset: OcrPreset) => {
    setForm({ ...preset.settings })
    setServerErrors({})
    toast.info(`${preset.label} preset loaded — review and save to apply it.`)
  }

  const handleSave = async () => {
    if (!form) return
    if (hasErrors) {
      toast.error('Fix the highlighted settings before saving.')
      setTab('pipeline')
      return
    }
    setSaving(true)
    try {
      const res = await saveOcrSettings(form)
      setSaved(res.settings)
      setForm(res.settings)
      setUpdatedAt(res.updated_at)
      setServerErrors({})
      setSavedFlash(true)
      window.setTimeout(() => setSavedFlash(false), 2200)
      toast.success('OCR settings saved. All OCR tasks now use them.')
      fetchOcrStatus().then(setStatus).catch(() => undefined)
    } catch (err) {
      const fieldErrors = serverFieldErrors(err)
      if (Object.keys(fieldErrors).length) {
        setServerErrors(fieldErrors)
        setTab('pipeline')
        toast.error('Some settings were rejected. Check the highlighted fields.')
      } else {
        toast.error(getApiErrorDetail(err, 'Failed to save OCR settings.'))
      }
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    setResetting(true)
    try {
      const res = await resetOcrSettings()
      setSaved(res.settings)
      setForm(res.settings)
      setUpdatedAt(res.updated_at)
      setServerErrors({})
      setConfirmReset(false)
      toast.success('OCR settings restored to the PP-OCRv6 defaults.')
    } catch (err) {
      toast.error(getApiErrorDetail(err, 'Failed to reset OCR settings.'))
    } finally {
      setResetting(false)
    }
  }

  const changedCount = form && saved ? Object.keys(form).filter(k => form[k] !== saved[k]).length : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div className="panel-header" style={{ flexShrink: 0, justifyContent: 'space-between', padding: '.65rem 1.5rem', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, flexShrink: 0, background: 'hsl(var(--accent) / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px solid hsl(var(--accent) / .3)',
          }}>
            <ScanText size={18} style={{ color: 'hsl(var(--accent))' }} />
          </div>
          <div style={{ minWidth: 0 }}>
            <h1 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>OCR Settings</h1>
            <p style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', margin: 0 }}>
              PP-OCRv6 preprocessing, detection and recognition parameters, used by every OCR task
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {dirty && (
            <span style={{
              fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--accent))', background: 'hsl(var(--accent) / .1)',
              padding: '3px 8px', borderRadius: 6,
            }}>
              {changedCount} unsaved change{changedCount === 1 ? '' : 's'}
            </span>
          )}
          {dirty && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => { setForm(saved); setServerErrors({}) }}
              disabled={saving}
              style={headerButton}
            >
              <Undo2 size={13} /> Discard
            </button>
          )}
          {confirmReset ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.74rem', color: 'hsl(var(--ink))' }}>
              Restore defaults?
              <button type="button" className="btn btn-danger" onClick={handleReset} disabled={resetting} style={headerButton}>
                {resetting ? <Loader size={13} className="spin" /> : <RotateCcw size={13} />} Reset
              </button>
              <button type="button" className="btn btn-ghost" onClick={() => setConfirmReset(false)} disabled={resetting} style={headerButton}>
                Cancel
              </button>
            </span>
          ) : (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setConfirmReset(true)}
              disabled={!form || saving}
              style={headerButton}
              title="Restore the official PP-OCRv6 defaults"
            >
              <RotateCcw size={13} /> Reset to defaults
            </button>
          )}
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleSave}
            disabled={!form || saving || (!dirty && !savedFlash)}
            style={{ ...headerButton, fontWeight: 700, background: savedFlash ? 'hsl(140 70% 45%)' : undefined }}
          >
            {saving ? <Loader size={13} className="spin" /> : savedFlash ? <CheckCircle size={13} /> : <Save size={13} />}
            {saving ? 'Saving…' : savedFlash ? 'Saved' : 'Save settings'}
          </button>
        </div>
      </div>

      {/* Status strip */}
      <div style={{
        display: 'flex', gap: '1.1rem', padding: '.5rem 1.5rem', background: 'hsl(var(--muted) / .25)',
        borderBottom: '1px solid hsl(var(--border) / .3)', fontSize: '.72rem', color: 'hsl(var(--pencil))',
        alignItems: 'center', flexWrap: 'wrap',
      }}>
        <StatusItem icon={<Cpu size={13} />}>
          Engine:{' '}
          <strong style={{ color: 'hsl(var(--ink))' }}>
            {status ? status.model : 'PP-OCRv6 Medium'}
          </strong>
          {status && (
            <>
              {' · '}
              <span style={{ color: status.models_present ? 'hsl(var(--ink))' : 'hsl(var(--destructive))', fontWeight: 600 }}>
                {status.models_present ? (status.gpu_available ? 'GPU available' : 'CPU only') : 'Model files missing'}
              </span>
              {' · '}{status.loaded ? 'loaded' : 'idle (loads on first use)'}
            </>
          )}
        </StatusItem>
        <Divider />
        <StatusItem icon={<CheckCircle size={13} />}>
          {updatedAt ? <>Last saved <strong style={{ color: 'hsl(var(--ink))' }}>{new Date(updatedAt).toLocaleString()}</strong></> : 'Using default settings'}
        </StatusItem>
        {status && (
          <>
            <Divider />
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '.66rem' }} title="Active settings fingerprint (also written to the logs)">
              {status.settings_fingerprint.split(' ')[0]}
            </span>
          </>
        )}
      </div>

      {/* Tabs */}
      <div role="tablist" style={{
        display: 'flex', gap: 6, padding: '.6rem 1.5rem', background: 'hsl(var(--card))',
        borderBottom: '1px solid hsl(var(--border) / .4)', overflowX: 'auto', flexShrink: 0,
      }}>
        {([
          { id: 'pipeline', label: 'Pipeline settings', icon: <SlidersHorizontal size={14} />, color: 'hsl(var(--accent))' },
          { id: 'test', label: 'Test OCR', icon: <FileSearch size={14} />, color: 'hsl(205 85% 48%)' },
          { id: 'vlm', label: 'Experimental VLM', icon: <FlaskConical size={14} />, color: 'hsl(35 90% 45%)' },
        ] as const).map(t => {
          const active = tab === t.id
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setTab(t.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '.45rem .9rem', borderRadius: 8,
                fontSize: '.78rem', fontWeight: active ? 700 : 500, cursor: 'pointer', whiteSpace: 'nowrap',
                fontFamily: 'Inter, sans-serif', transition: 'all .15s ease',
                color: active ? t.color : 'hsl(var(--pencil))',
                background: active ? t.color.replace(')', ' / .1)') : 'transparent',
                border: active ? `1.5px solid ${t.color.replace(')', ' / .35)')}` : '1px solid transparent',
              }}
            >
              {t.icon}{t.label}
              {t.id === 'pipeline' && hasErrors && <AlertCircle size={13} style={{ color: 'hsl(var(--destructive))' }} />}
            </button>
          )
        })}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem 2.5rem' }}>
        <div style={{ maxWidth: 1120, margin: '0 auto' }}>
          {loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: '4rem 0', color: 'hsl(var(--pencil))' }}>
              <Loader size={28} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontSize: '.8rem' }}>Loading OCR settings…</span>
            </div>
          )}

          {!loading && loadError && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: '3rem 1rem',
              border: '2px dashed hsl(var(--border))', borderRadius: 12, textAlign: 'center',
            }}>
              <AlertCircle size={28} style={{ color: 'hsl(var(--destructive))' }} />
              <div style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>OCR settings could not be loaded</div>
              <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))' }}>{loadError}</div>
              <button type="button" className="btn btn-secondary" onClick={load} style={headerButton}>Try again</button>
            </div>
          )}

          {!loading && !loadError && schema && form && saved && (
            <>
              {/* Panels stay mounted so in-progress tests survive tab switches */}
              <div hidden={tab !== 'pipeline'}>
                <OcrSettingsForm
                  schema={schema}
                  values={form}
                  saved={saved}
                  presets={presets}
                  errors={errors}
                  onChange={updateField}
                  onApplyPreset={applyPreset}
                />
              </div>
              <div hidden={tab !== 'test'}>
                <OcrTestPanel values={form} isDirty={dirty} hasErrors={hasErrors} />
              </div>
              <div hidden={tab !== 'vlm'}>
                {vlmOpened && <VlmOcrPanel />}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

const headerButton = {
  fontSize: '.76rem', padding: '.42rem .85rem', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 6,
} as const

function StatusItem({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>{icon}<span>{children}</span></div>
}

function Divider() {
  return <div style={{ width: 1, height: 12, background: 'hsl(var(--border))' }} />
}
