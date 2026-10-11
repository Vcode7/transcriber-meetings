import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import axios from 'axios'
import { toast } from 'sonner'
import {
  AlertCircle, AlertTriangle, Code, Copy, Eye, FlaskConical, Info, Loader, Play, RefreshCw, Server, Sparkles, Square,
} from 'lucide-react'
import { getApiErrorDetail } from '../../../lib/errors'
import { useUIStore } from '../../../store/ui'
import { fetchVlmModels, runVlmOcr } from '../api/ocrApi'
import type { VlmModel, VlmPromptMode, VlmResult } from '../types'
import { copyText, formatBytes, formatMs } from '../utils'
import { buildPreviewDocument } from '../vlmPreview'
import { FileDropzone } from './FileDropzone'
import { Banner, SectionCard, Stat } from './Fields'

const ACCEPT = ['.pdf', '.png', '.jpg', '.jpeg', '.webp']
const MAX_MB = 25
const AMBER = 'hsl(35 90% 45%)'

export default function VlmOcrPanel() {
  const [models, setModels] = useState<VlmModel[]>([])
  const [serverUrl, setServerUrl] = useState('')
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [model, setModel] = useState('')
  const [promptMode, setPromptMode] = useState<VlmPromptMode>('ocr_layout')
  const [maxTokens, setMaxTokens] = useState(8192)
  const [page, setPage] = useState(1)
  const [showCustom, setShowCustom] = useState(false)
  const [customPrompt, setCustomPrompt] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState<VlmResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const loadModels = useCallback(async () => {
    setLoadingModels(true)
    setModelsError(null)
    try {
      const res = await fetchVlmModels()
      setServerUrl(res.server_url)
      setModels(res.models)
      if (!res.success) setModelsError(res.error ?? 'Could not reach the Ollama server.')
      setModel(current => {
        if (current && res.models.some(m => m.name === current)) return current
        const preferred = res.models.find(m => m.is_chandra) ?? res.models.find(m => m.vision) ?? res.models[0]
        return preferred?.name ?? ''
      })
    } catch (err) {
      setModelsError(getApiErrorDetail(err, 'Could not load Ollama models.'))
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => { loadModels() }, [loadModels])

  useEffect(() => {
    if (!running) return
    const started = Date.now()
    const timer = window.setInterval(() => setElapsed(Date.now() - started), 500)
    return () => window.clearInterval(timer)
  }, [running])

  useEffect(() => () => controllerRef.current?.abort(), [])

  const selected = models.find(m => m.name === model)
  const hasChandra = models.some(m => m.is_chandra)
  const isPdf = file?.name.toLowerCase().endsWith('.pdf') ?? false
  const tokensValid = Number.isInteger(maxTokens) && maxTokens >= 256 && maxTokens <= 16384
  const pageValid = !isPdf || (Number.isInteger(page) && page >= 1)

  const run = async () => {
    if (!file || !model) return
    const controller = new AbortController()
    controllerRef.current = controller
    setRunning(true)
    setElapsed(0)
    setError(null)
    setResult(null)
    try {
      const res = await runVlmOcr(
        file,
        { model, page: isPdf ? page : 1, promptMode, maxTokens, customPrompt: showCustom ? customPrompt : undefined },
        controller.signal,
      )
      setResult(res)
      if (!res.html.trim()) toast.info('The model returned no HTML.')
    } catch (err) {
      if (axios.isCancel(err)) toast.info('VLM OCR cancelled')
      else setError(getApiErrorDetail(err, 'The VLM OCR run failed.'))
    } finally {
      setRunning(false)
      controllerRef.current = null
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <div style={{
        display: 'flex', gap: 10, alignItems: 'flex-start', padding: '.75rem .95rem', borderRadius: 10,
        background: 'hsl(35 90% 50% / .1)', border: '1.5px solid hsl(35 90% 50% / .35)',
      }}>
        <FlaskConical size={18} style={{ color: AMBER, flexShrink: 0, marginTop: 1 }} />
        <div style={{ fontSize: '.76rem', lineHeight: 1.5, color: 'hsl(var(--ink))' }}>
          <strong style={{ color: AMBER }}>Experimental · testing only.</strong>{' '}
          Runs Chandra OCR 2 through your Ollama server and renders the HTML it returns. This is not used by
          document uploads, scanned PDFs or any other OCR in the app, which keep using PP-OCRv6.
        </div>
      </div>

      <SectionCard
        title="Vision-language OCR"
        icon={<Sparkles size={16} />}
        color={AMBER}
        description="Choose an Ollama model, upload a page, and compare the model's structured HTML with the source image."
        actions={
          <button
            type="button"
            className="btn btn-ghost"
            onClick={loadModels}
            disabled={loadingModels || running}
            style={{ fontSize: '.72rem', padding: '.3rem .6rem', display: 'flex', alignItems: 'center', gap: 5 }}
          >
            <RefreshCw size={12} className={loadingModels ? 'spin' : ''} /> Refresh models
          </button>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
            <Server size={12} /> Ollama server:
            <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'hsl(var(--ink))' }}>{serverUrl || '…'}</span>
            <span>(set on the Settings page)</span>
          </div>

          {modelsError && <Banner tone="error" icon={<AlertCircle size={14} />}>{modelsError}</Banner>}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '.75rem' }}>
            <Labeled label="Model">
              <select
                value={model}
                onChange={e => setModel(e.target.value)}
                disabled={running || loadingModels || !models.length}
                style={inputStyle}
              >
                {!models.length && <option value="">{loadingModels ? 'Loading models…' : 'No Ollama models found'}</option>}
                {models.map(m => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                    {m.parameter_size ? ` · ${m.parameter_size}` : m.size ? ` · ${formatBytes(m.size)}` : ''}
                    {m.vision ? ' · vision' : m.vision === false ? ' · text only' : ''}
                    {m.running ? ' · loaded' : ''}
                  </option>
                ))}
              </select>
            </Labeled>
            <Labeled label="Output format">
              <div role="radiogroup" style={{ display: 'flex', gap: 4 }}>
                {([['ocr_layout', 'Layout blocks'], ['ocr', 'Plain HTML']] as const).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="radio"
                    aria-checked={promptMode === value}
                    onClick={() => setPromptMode(value)}
                    disabled={running || showCustom}
                    style={{
                      flex: 1, padding: '6px 8px', borderRadius: 6, fontSize: '.74rem', cursor: 'pointer',
                      fontWeight: promptMode === value ? 700 : 500, fontFamily: 'Inter, sans-serif',
                      border: promptMode === value ? `1.5px solid ${AMBER}` : '1px solid hsl(var(--border))',
                      background: promptMode === value ? 'hsl(35 90% 50% / .1)' : 'hsl(var(--card))',
                      color: 'hsl(var(--ink))', opacity: showCustom ? 0.5 : 1,
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </Labeled>
            <Labeled label="Max output tokens" error={tokensValid ? undefined : '256 – 16,384'}>
              <input
                type="number" min={256} max={16384} step={256}
                value={Number.isNaN(maxTokens) ? '' : maxTokens}
                onChange={e => setMaxTokens(parseInt(e.target.value, 10))}
                disabled={running}
                style={{ ...inputStyle, fontFamily: 'JetBrains Mono, monospace' }}
              />
            </Labeled>
            {isPdf && (
              <Labeled label="PDF page" error={pageValid ? undefined : 'Enter a page number'}>
                <input
                  type="number" min={1} step={1}
                  value={Number.isNaN(page) ? '' : page}
                  onChange={e => setPage(parseInt(e.target.value, 10))}
                  disabled={running}
                  style={{ ...inputStyle, fontFamily: 'JetBrains Mono, monospace' }}
                />
              </Labeled>
            )}
          </div>

          {!loadingModels && models.length > 0 && !hasChandra && (
            <Banner tone="info" icon={<Info size={14} />}>
              No Chandra model is installed in Ollama. Pull a Chandra OCR 2 build (for example a community GGUF
              conversion) with <code>ollama pull &lt;model&gt;</code>, then refresh. Other vision models also work for comparison.
            </Banner>
          )}
          {selected && selected.vision === false && (
            <Banner tone="warning" icon={<AlertTriangle size={14} />}>
              {selected.name} does not report image support, so it cannot read the page.
            </Banner>
          )}

          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.74rem', cursor: 'pointer', color: 'hsl(var(--ink))' }}>
              <input
                type="checkbox"
                checked={showCustom}
                onChange={e => setShowCustom(e.target.checked)}
                disabled={running}
                style={{ accentColor: AMBER }}
              />
              Use a custom prompt instead of Chandra's built-in prompt
            </label>
            {showCustom && (
              <textarea
                value={customPrompt}
                onChange={e => setCustomPrompt(e.target.value)}
                rows={5}
                maxLength={20000}
                placeholder="OCR this image to HTML…"
                disabled={running}
                style={{ ...inputStyle, width: '100%', marginTop: 6, resize: 'vertical', fontFamily: 'JetBrains Mono, monospace', boxSizing: 'border-box' }}
              />
            )}
          </div>

          <FileDropzone
            accept={ACCEPT}
            file={file}
            onFile={f => { setFile(f); setResult(null); setError(null); setPage(1) }}
            maxMb={MAX_MB}
            disabled={running}
            hint={`One page per run · PDF, PNG, JPG, WEBP · up to ${MAX_MB} MB`}
          />

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={run}
              disabled={!file || !model || running || !tokensValid || !pageValid || (showCustom && !customPrompt.trim())}
              style={{ fontSize: '.8rem', padding: '.5rem 1.1rem', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              {running ? <Loader size={14} className="spin" /> : <Play size={14} />}
              {running ? `Generating… ${formatMs(elapsed)}` : 'Run VLM OCR'}
            </button>
            {running && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => controllerRef.current?.abort()}
                style={{ fontSize: '.78rem', padding: '.45rem .9rem', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Square size={12} /> Cancel
              </button>
            )}
            {running && (
              <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                Large vision models can take several minutes per page, especially on first load.
              </span>
            )}
          </div>

          {error && <Banner tone="error" icon={<AlertCircle size={14} />}>{error}</Banner>}
        </div>
      </SectionCard>

      {result && <VlmResultView result={result} />}
    </div>
  )
}

function VlmResultView({ result }: { result: VlmResult }) {
  const [view, setView] = useState<'rendered' | 'source'>('rendered')
  const [outlines, setOutlines] = useState(true)
  const dark = useUIStore(s => s.theme) === 'dark'
  const srcDoc = useMemo(() => buildPreviewDocument(result.html, { outlines, dark }), [result.html, outlines, dark])

  const copy = async () => {
    try {
      await copyText(result.html)
      toast.success('HTML copied to clipboard')
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }

  return (
    <SectionCard title="Model output" icon={<Eye size={16} />} color={AMBER}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '.5rem', marginBottom: '.9rem' }}>
        <Stat label="Model" value={result.model} />
        <Stat label="Time" value={formatMs(result.elapsed_ms)} />
        <Stat label="Output tokens" value={result.output_tokens?.toLocaleString() ?? '—'} />
        <Stat label="Prompt tokens" value={result.prompt_tokens?.toLocaleString() ?? '—'} />
        <Stat label="Image" value={`${result.image_size[0]}×${result.image_size[1]}`} />
        {result.page_count > 1 && <Stat label="Page" value={`${result.page} of ${result.page_count}`} />}
      </div>

      {result.truncated && (
        <div style={{ marginBottom: '.8rem' }}>
          <Banner tone="warning" icon={<AlertTriangle size={14} />}>
            The output stopped at the token limit, so the HTML may be incomplete. Increase “Max output tokens” and run again.
          </Banner>
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 300px', maxWidth: 520 }}>
          <div style={paneTitle}>Source</div>
          <img
            src={result.source_preview}
            alt={`Source ${result.filename}${result.page_count > 1 ? `, page ${result.page}` : ''}`}
            style={{ width: '100%', borderRadius: 8, border: '1px solid hsl(var(--border) / .6)', background: 'white' }}
          />
        </div>
        <div style={{ flex: '2 1 380px', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
            <div role="tablist" style={{ display: 'flex', gap: 4 }}>
              <TabButton active={view === 'rendered'} onClick={() => setView('rendered')} icon={<Eye size={12} />} label="Rendered" />
              <TabButton active={view === 'source'} onClick={() => setView('source')} icon={<Code size={12} />} label="HTML" />
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
              {view === 'rendered' && (
                <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '.72rem', cursor: 'pointer', color: 'hsl(var(--pencil))' }}>
                  <input type="checkbox" checked={outlines} onChange={e => setOutlines(e.target.checked)} style={{ accentColor: AMBER }} />
                  Show layout blocks
                </label>
              )}
              <button type="button" className="btn btn-ghost" onClick={copy} style={{ fontSize: '.7rem', padding: '.2rem .5rem', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Copy size={12} /> Copy HTML
              </button>
            </div>
          </div>
          {view === 'rendered' ? (
            <iframe
              title="VLM OCR HTML preview"
              sandbox=""
              referrerPolicy="no-referrer"
              srcDoc={srcDoc}
              style={{
                width: '100%', height: 640, borderRadius: 8, border: '1px solid hsl(var(--border) / .6)',
                background: dark ? '#0f1b2d' : '#ffffff', display: 'block', resize: 'vertical',
              }}
            />
          ) : (
            <pre style={{
              margin: 0, height: 640, overflow: 'auto', padding: '.8rem', borderRadius: 8, boxSizing: 'border-box',
              border: '1px solid hsl(var(--border) / .6)', background: 'hsl(var(--paper) / .35)', color: 'hsl(var(--ink))',
              fontFamily: 'JetBrains Mono, monospace', fontSize: '.72rem', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            }}>
              {result.html || result.raw || '(empty response)'}
            </pre>
          )}
        </div>
      </div>
    </SectionCard>
  )
}

function TabButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: ReactNode; label: string }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 5, padding: '.3rem .7rem', borderRadius: 6, cursor: 'pointer',
        fontSize: '.74rem', fontWeight: active ? 700 : 500, fontFamily: 'Inter, sans-serif',
        color: active ? AMBER : 'hsl(var(--pencil))',
        background: active ? 'hsl(35 90% 50% / .1)' : 'transparent',
        border: active ? '1.5px solid hsl(35 90% 50% / .4)' : '1px solid transparent',
      }}
    >
      {icon}{label}
    </button>
  )
}

function Labeled({ label, error, children }: { label: string; error?: string; children: ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 }}>
      <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{label}</span>
      {children}
      {error && <span style={{ fontSize: '.68rem', color: 'hsl(var(--destructive))', fontWeight: 600 }}>{error}</span>}
    </div>
  )
}

const inputStyle: CSSProperties = {
  padding: '6px 8px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
  fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', minWidth: 0,
}

const paneTitle: CSSProperties = {
  fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 6,
}
