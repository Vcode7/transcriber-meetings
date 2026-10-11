import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import axios from 'axios'
import { toast } from 'sonner'
import {
  AlertCircle, CheckCircle, ChevronDown, ChevronRight, Copy, Cpu, Download, FileSearch, Loader, Play, Square,
} from 'lucide-react'
import { getApiErrorDetail } from '../../../lib/errors'
import { runOcrTest } from '../api/ocrApi'
import type { OcrSettings, OcrTestPage, OcrTestResult } from '../types'
import { copyText, downloadText, formatMs, formatPercent, serverFieldErrors } from '../utils'
import { FileDropzone } from './FileDropzone'
import { Banner, SectionCard, Stat } from './Fields'

const ACCEPT = ['.pdf', '.png', '.jpg', '.jpeg', '.webp', '.docx', '.pptx', '.doc', '.ppt']
const MAX_MB = 25

interface OcrTestPanelProps {
  values: OcrSettings
  isDirty: boolean
  hasErrors: boolean
}

export default function OcrTestPanel({ values, isDirty, hasErrors }: OcrTestPanelProps) {
  const [file, setFile] = useState<File | null>(null)
  const [useUnsaved, setUseUnsaved] = useState(true)
  const [maxPages, setMaxPages] = useState(5)
  const [showBoxes, setShowBoxes] = useState(true)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState<OcrTestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!running) return
    const started = Date.now()
    const timer = window.setInterval(() => setElapsed(Date.now() - started), 250)
    return () => window.clearInterval(timer)
  }, [running])

  useEffect(() => () => controllerRef.current?.abort(), [])

  const sendUnsaved = isDirty && useUnsaved
  const isPdf = file?.name.toLowerCase().endsWith('.pdf') ?? false

  const run = async () => {
    if (!file) return
    const controller = new AbortController()
    controllerRef.current = controller
    setRunning(true)
    setElapsed(0)
    setError(null)
    setResult(null)
    try {
      const res = await runOcrTest(
        file,
        { settings: sendUnsaved ? values : undefined, maxPages, showBoxes },
        controller.signal,
      )
      setResult(res)
      if (!res.text.trim()) toast.info('OCR finished, but no text was found.')
    } catch (err) {
      if (axios.isCancel(err)) {
        toast.info('OCR test cancelled')
      } else {
        const fieldErrors = serverFieldErrors(err)
        const firstField = Object.entries(fieldErrors)[0]
        setError(
          firstField
            ? `Unsaved setting "${firstField[0]}" is invalid: ${firstField[1]}`
            : getApiErrorDetail(err, 'The OCR test failed.'),
        )
      }
    } finally {
      setRunning(false)
      controllerRef.current = null
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <SectionCard
        title="Test OCR"
        icon={<FileSearch size={16} />}
        color="hsl(205 85% 50%)"
        description="Upload a PDF, image or Office document to run it through the production OCR pipeline and inspect the extracted text."
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
          <FileDropzone
            accept={ACCEPT}
            file={file}
            onFile={f => { setFile(f); setResult(null); setError(null) }}
            maxMb={MAX_MB}
            disabled={running}
            hint={`PDF, PNG, JPG, WEBP, DOCX, PPTX · up to ${MAX_MB} MB`}
          />

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.6rem 1.2rem', alignItems: 'center' }}>
            <Option label="Settings">
              {isDirty ? (
                <select
                  value={useUnsaved ? 'unsaved' : 'saved'}
                  onChange={e => setUseUnsaved(e.target.value === 'unsaved')}
                  disabled={running}
                  style={selectStyle}
                >
                  <option value="unsaved">Current form (unsaved changes)</option>
                  <option value="saved">Saved settings</option>
                </select>
              ) : (
                <span style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', fontWeight: 600 }}>Saved settings</span>
              )}
            </Option>
            {isPdf && (
              <Option label="Pages">
                <select value={maxPages} onChange={e => setMaxPages(Number(e.target.value))} disabled={running} style={selectStyle}>
                  {[1, 2, 3, 5, 10].map(n => <option key={n} value={n}>First {n}</option>)}
                </select>
              </Option>
            )}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.76rem', color: 'hsl(var(--ink))', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={e => setShowBoxes(e.target.checked)}
                disabled={running}
                style={{ accentColor: 'hsl(var(--accent))' }}
              />
              Show detected text regions
            </label>
          </div>

          {sendUnsaved && hasErrors && (
            <Banner tone="error" icon={<AlertCircle size={14} />}>
              Fix the highlighted settings on the Pipeline tab before testing with unsaved changes.
            </Banner>
          )}

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={run}
              disabled={!file || running || (sendUnsaved && hasErrors)}
              style={{ fontSize: '.8rem', padding: '.5rem 1.1rem', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              {running ? <Loader size={14} className="spin" /> : <Play size={14} />}
              {running ? `Running OCR… ${formatMs(elapsed)}` : 'Run OCR'}
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
                The first run loads the OCR models and can take a few extra seconds.
              </span>
            )}
          </div>

          {error && <Banner tone="error" icon={<AlertCircle size={14} />}>{error}</Banner>}
        </div>
      </SectionCard>

      {result && <TestResults result={result} />}
    </div>
  )
}

function TestResults({ result }: { result: OcrTestResult }) {
  const provider = result.provider === 'CUDAExecutionProvider' ? 'GPU (CUDA)' : result.provider ? 'CPU' : '—'
  const stem = result.filename.replace(/\.[^.]+$/, '')
  return (
    <SectionCard
      title="Results"
      icon={<CheckCircle size={16} />}
      color="hsl(150 60% 40%)"
      actions={
        <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace' }}>
          {result.used_unsaved_settings ? 'unsaved settings' : 'saved settings'} · {result.settings_fingerprint.split(' ')[0]}
        </span>
      }
    >
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '.5rem', marginBottom: '1rem',
      }}>
        <Stat label="Pages" value={result.kind === 'pdf' ? `${result.pages_processed} of ${result.page_count}` : String(result.page_count)} />
        <Stat label="Characters" value={result.char_count.toLocaleString()} />
        <Stat label="Words" value={result.word_count.toLocaleString()} />
        <Stat label="Avg. confidence" value={formatPercent(result.avg_confidence)} />
        <Stat label="Time" value={formatMs(result.elapsed_ms)} />
        <Stat label="Engine" value={provider} icon={<Cpu size={12} />} />
      </div>

      {result.pages.map((page, idx) => (
        <PageResult key={page.page} page={page} defaultOpen={idx === 0} multiPage={result.pages.length > 1} />
      ))}

      {result.pages.length > 1 && (
        <div style={{ marginTop: '1rem' }}>
          <TextBlock
            title="Full extracted text"
            text={result.text}
            onDownload={() => downloadText(result.text, `${stem}_ocr.txt`)}
            rows={14}
          />
        </div>
      )}
    </SectionCard>
  )
}

function PageResult({ page, defaultOpen, multiPage }: { page: OcrTestPage; defaultOpen: boolean; multiPage: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const source = SOURCE_LABEL[page.source]
  const meta: string[] = []
  if (page.line_count !== undefined) meta.push(`${page.line_count} lines`)
  if (page.avg_confidence !== undefined && page.line_count) meta.push(`${formatPercent(page.avg_confidence)} conf.`)
  if (page.render_dpi) meta.push(`${page.render_dpi} dpi`)
  if (page.rotation) meta.push(`rotated ${page.rotation}°`)
  if (page.skew_angle) meta.push(`deskewed ${page.skew_angle}°`)
  if (page.embedded_images) meta.push(`${page.embedded_images_with_text ?? 0}/${page.embedded_images} images with text`)
  if (page.elapsed_ms !== undefined) meta.push(formatMs(page.elapsed_ms))

  return (
    <div style={{ border: '1px solid hsl(var(--border) / .5)', borderRadius: 10, marginBottom: '.6rem', overflow: 'hidden' }}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '.6rem .8rem', cursor: 'pointer',
          background: 'hsl(var(--paper) / .45)', border: 'none', textAlign: 'left', fontFamily: 'Inter, sans-serif',
        }}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
          {multiPage || page.source !== 'document' ? `Page ${page.page}` : 'Document'}
        </span>
        <span style={{
          fontSize: '.64rem', fontWeight: 700, padding: '1px 6px', borderRadius: 6,
          color: source.color, background: source.color.replace(')', ' / .12)'),
        }}>
          {source.label}
        </span>
        <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', marginLeft: 'auto', textAlign: 'right' }}>
          {meta.join(' · ')}
        </span>
      </button>
      {open && (
        <div style={{ padding: '.8rem', display: 'flex', flexWrap: 'wrap', gap: '.8rem', alignItems: 'flex-start' }}>
          {page.preview && (
            <a
              href={page.preview}
              target="_blank"
              rel="noreferrer"
              title="Open full-size preview"
              style={{ flex: '1 1 300px', maxWidth: 520, display: 'block' }}
            >
              <img
                src={page.preview}
                alt={`Page ${page.page} with detected text regions`}
                style={{ width: '100%', borderRadius: 8, border: '1px solid hsl(var(--border) / .6)', background: 'white' }}
              />
            </a>
          )}
          <div style={{ flex: '2 1 340px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
            <TextBlock title="Extracted text" text={page.text} rows={page.preview ? 16 : 10} />
            {page.applied_steps && page.applied_steps.length > 0 && (
              <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                Preprocessing applied: <strong style={{ color: 'hsl(var(--ink))' }}>{page.applied_steps.join(', ')}</strong>
              </div>
            )}
            {page.lines && page.lines.length > 0 && <LineConfidence lines={page.lines} />}
          </div>
        </div>
      )}
    </div>
  )
}

function LineConfidence({ lines }: { lines: { text: string; score: number }[] }) {
  return (
    <details style={{ fontSize: '.72rem' }}>
      <summary style={{ cursor: 'pointer', color: 'hsl(var(--pencil))', fontWeight: 600 }}>
        Line confidence ({lines.length})
      </summary>
      <div style={{ maxHeight: 220, overflowY: 'auto', marginTop: 6, border: '1px solid hsl(var(--border) / .4)', borderRadius: 8 }}>
        {lines.map((line, i) => (
          <div key={i} style={{
            display: 'flex', gap: 8, padding: '3px 8px', alignItems: 'baseline',
            borderBottom: '1px solid hsl(var(--border) / .25)',
          }}>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, minWidth: 44, textAlign: 'right',
              color: line.score >= 0.9 ? 'hsl(150 60% 36%)' : line.score >= 0.75 ? 'hsl(35 90% 40%)' : 'hsl(var(--destructive))',
            }}>
              {(line.score * 100).toFixed(0)}%
            </span>
            <span style={{ color: 'hsl(var(--ink))', wordBreak: 'break-word' }}>{line.text}</span>
          </div>
        ))}
      </div>
    </details>
  )
}

function TextBlock({ title, text, rows, onDownload }: { title: string; text: string; rows: number; onDownload?: () => void }) {
  const copy = async () => {
    try {
      await copyText(text)
      toast.success('Copied to clipboard')
    } catch {
      toast.error('Could not copy to the clipboard')
    }
  }
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
          {title} <span style={{ fontWeight: 500, color: 'hsl(var(--pencil))' }}>· {text.length.toLocaleString()} chars</span>
        </span>
        <div style={{ display: 'flex', gap: 4 }}>
          <button type="button" className="btn btn-ghost" onClick={copy} disabled={!text} style={smallButton}>
            <Copy size={12} /> Copy
          </button>
          {onDownload && (
            <button type="button" className="btn btn-ghost" onClick={onDownload} disabled={!text} style={smallButton}>
              <Download size={12} /> .txt
            </button>
          )}
        </div>
      </div>
      <textarea
        readOnly
        value={text || '(no text found)'}
        rows={rows}
        style={{
          width: '100%', resize: 'vertical', padding: '.6rem .7rem', borderRadius: 8, boxSizing: 'border-box',
          border: '1px solid hsl(var(--border) / .6)', background: 'hsl(var(--paper) / .35)',
          fontFamily: 'JetBrains Mono, monospace', fontSize: '.74rem', lineHeight: 1.5, color: 'hsl(var(--ink))',
        }}
      />
    </div>
  )
}

const SOURCE_LABEL: Record<OcrTestPage['source'], { label: string; color: string }> = {
  ocr: { label: 'OCR', color: 'hsl(205 85% 45%)' },
  text_layer: { label: 'Text layer', color: 'hsl(150 60% 36%)' },
  document: { label: 'Document extractor', color: 'hsl(280 60% 52%)' },
}

const selectStyle: CSSProperties = {
  padding: '5px 8px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
  fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', cursor: 'pointer',
}

const smallButton: CSSProperties = {
  fontSize: '.7rem', padding: '.2rem .5rem', display: 'flex', alignItems: 'center', gap: 4, borderRadius: 6,
}

function Option({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>{label}</span>
      {children}
    </div>
  )
}

