import { useState, useCallback, useEffect, useRef } from 'react'
import {
  Database, Upload, Trash2, CheckCircle, Loader, AlertTriangle,
  RefreshCw, Info, FileText, File, X, ChevronRight
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'

interface GlobalContextDoc {
  id: string
  filename: string
  file_hash: string
  embedded: boolean
  chunk_count: number
  created_at: string
  updated_at: string
}

interface StatusInfo {
  embedding_model: string
  embedding_model_dir: string
  total_documents: number
  embedded_documents: number
  total_chunks: number
  vector_store_dir: string
}

const ALLOWED_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  'text/plain',
  'text/markdown',
  'image/png',
  'image/jpeg',
  'image/webp',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.ms-excel',
  'text/csv',
]

const ALLOWED_EXT_LABELS = 'PDF, DOCX, PPTX, TXT, MD, PNG, JPG, EXCEL, CSV'

function formatDate(str: string) {
  try {
    return new Date(str).toLocaleDateString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return str }
}

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'pdf': return '📄'
    case 'docx': return '📝'
    case 'pptx': return '📊'
    case 'txt': case 'md': return '📃'
    case 'png': case 'jpg': case 'jpeg': case 'webp': return '🖼️'
    default: return '📎'
  }
}

export default function GlobalContext() {
  const [docs, setDocs] = useState<GlobalContextDoc[]>([])
  const [status, setStatus] = useState<StatusInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchDocs = useCallback(async () => {
    try {
      const [docsRes, statusRes] = await Promise.all([
        api.get('/global-context/'),
        api.get('/global-context/status'),
      ])
      setDocs(docsRes.data.documents || [])
      setStatus(statusRes.data)
    } catch (e) {
      toast.error('Failed to load global context documents')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchDocs() }, [fetchDocs])

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return
    setUploading(true)
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    try {
      const res = await api.post('/global-context/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const { uploaded, skipped_duplicates } = res.data
      const embedded = uploaded.filter((u: any) => u.embedded).length
      const failed = uploaded.filter((u: any) => u.error).length
      const msgs = []
      if (embedded > 0) msgs.push(`${embedded} document${embedded !== 1 ? 's' : ''} indexed`)
      if (skipped_duplicates > 0) msgs.push(`${skipped_duplicates} duplicate${skipped_duplicates !== 1 ? 's' : ''} skipped`)
      if (failed > 0) msgs.push(`${failed} failed`)
      toast.success(msgs.join(', ') || 'Upload complete')
      await fetchDocs()
    } catch (e: any) {
      const msg = e?.response?.data?.detail || 'Upload failed'
      toast.error(msg)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    uploadFiles(files)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length) uploadFiles(files)
    e.target.value = ''
  }

  const deleteDoc = async (doc: GlobalContextDoc) => {
    if (!confirm(`Delete "${doc.filename}" from the knowledge base?`)) return
    setDeletingId(doc.id)
    try {
      await api.delete(`/global-context/${doc.id}`)
      toast.success(`"${doc.filename}" removed`)
      await fetchDocs()
    } catch {
      toast.error('Failed to delete document')
    } finally {
      setDeletingId(null)
    }
  }

  const reindex = async () => {
    if (!confirm('Re-index all documents? This may take a while.')) return
    setReindexing(true)
    try {
      const res = await api.post('/global-context/reindex')
      toast.success(`Re-indexed ${res.data.processed} documents`)
      await fetchDocs()
    } catch {
      toast.error('Re-index failed')
    } finally {
      setReindexing(false)
    }
  }

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* ── Panel Header ── */}
      <div className="panel-header">
        <div style={{
          width: 34, height: 34, borderRadius: '10px', flexShrink: 0,
          background: 'linear-gradient(135deg, hsl(260,85%,60% / .18), hsl(220,80%,60% / .08))',
          border: '1.5px solid hsl(260,85%,60% / .25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Database size={16} style={{ color: 'hsl(260,85%,65%)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h1>Global Knowledge Base</h1>
          <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
            Upload organization-wide documents that provide background context for all meetings
          </p>
        </div>
        {docs.length > 0 && (
          <button
            onClick={reindex}
            disabled={reindexing}
            className="btn btn-ghost"
            style={{ fontSize: '.82rem', gap: '6px', flexShrink: 0 }}
          >
            {reindexing ? <Loader size={14} className="spin" /> : <RefreshCw size={14} />}
            Re-index All
          </button>
        )}
      </div>

      {/* ── Scrollable Content ── */}
      <div className="page-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', background: 'transparent', padding: '2rem 2.5rem 4rem' }}>
        
        {/* Model info text */}
        <div style={{
          fontSize: '.85rem',
          color: 'hsl(var(--pencil))',
          lineHeight: 1.5,
          marginTop: '-0.5rem',
        }}>
          These are indexed using <strong>{status?.embedding_model || 'Qwen3-Embedding-0.6B'}</strong> and
          retrieved automatically when generating Raw MoM.
        </div>

        {/* ── Stats Bar ── */}
        {status && (
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem',
          }}>
            {[
              { label: 'Documents', value: status.total_documents },
              { label: 'Indexed', value: status.embedded_documents },
              { label: 'Total Chunks', value: status.total_chunks.toLocaleString() },
            ].map(({ label, value }) => (
              <div key={label} style={{
                borderRadius: 12, padding: '1rem 1.25rem',
                background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--border) / .5)',
              }}>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'hsl(var(--ink))', lineHeight: 1 }}>{value}</div>
                <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', marginTop: 4, textTransform: 'uppercase', letterSpacing: '.04em', fontWeight: 600 }}>{label}</div>
              </div>
            ))}
          </div>
        )}

        {/* ── Upload Zone ── */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            borderRadius: 16,
            border: `2px dashed ${dragging ? 'hsl(260,85%,65%)' : 'hsl(var(--border))'}`,
            background: dragging
              ? 'hsl(260,85%,60% / .06)'
              : 'hsl(var(--card))',
            padding: '2.5rem',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: '0.75rem', cursor: 'pointer',
            transition: 'all 0.2s ease',
            position: 'relative',
          }}
        >
          {uploading ? (
            <>
              <Loader size={32} className="spin" style={{ color: 'hsl(260,85%,65%)' }} />
              <p style={{ margin: 0, fontSize: '.9rem', color: 'hsl(var(--pencil))' }}>
                Processing and indexing…
              </p>
            </>
          ) : (
            <>
              <div style={{
                width: 48, height: 48, borderRadius: 12,
                background: 'hsl(260,85%,60% / .12)',
                border: '1.5px solid hsl(260,85%,60% / .2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Upload size={22} style={{ color: 'hsl(260,85%,65%)' }} />
              </div>
              <div style={{ textAlign: 'center' }}>
                <p style={{ margin: 0, fontSize: '.95rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                  Drag & drop files here
                </p>
                <p style={{ margin: '4px 0 0', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                  or click to browse · {ALLOWED_EXT_LABELS} · Max 50 MB
                </p>
              </div>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        {/* ── Info Banner ── */}
        <div style={{
          borderRadius: 10, padding: '.75rem 1rem',
          background: 'hsl(220,80%,60% / .07)',
          border: '1px solid hsl(220,80%,60% / .2)',
          display: 'flex', alignItems: 'flex-start', gap: '10px',
        }}>
          <Info size={15} style={{ color: 'hsl(220,80%,65%)', flexShrink: 0, marginTop: 1 }} />
          <p style={{ margin: 0, fontSize: '.8rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>
            Documents uploaded here are retrieved by the <strong>Raw MoM pipeline</strong> to
            provide organizational context for any meeting. Ideal for: company glossaries,
            project specs, product docs, process manuals, org charts.
          </p>
        </div>

        {/* ── Document List ── */}
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
            <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
          </div>
        ) : docs.length === 0 ? (
          <div style={{
            textAlign: 'center', padding: '3rem',
            borderRadius: 12, border: '1.5px dashed hsl(var(--border) / .5)',
            color: 'hsl(var(--pencil))',
          }}>
            <Database size={32} style={{ margin: '0 auto 12px', opacity: 0.35 }} />
            <p style={{ margin: 0, fontSize: '.9rem' }}>No documents yet.</p>
            <p style={{ margin: '4px 0 0', fontSize: '.8rem' }}>Upload organizational documents to get started.</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{
              fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--pencil))',
              textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '0.25rem',
            }}>
              Knowledge Documents
            </div>
            {docs.map(doc => (
              <div key={doc.id} style={{
                borderRadius: 12, padding: '0.9rem 1.1rem',
                background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--border) / .5)',
                display: 'flex', alignItems: 'center', gap: '0.9rem',
                transition: 'border-color 0.15s',
              }}>
                {/* File Icon */}
                <div style={{ fontSize: '1.4rem', flexShrink: 0, lineHeight: 1 }}>
                  {getFileIcon(doc.filename)}
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: '.9rem', fontWeight: 600, color: 'hsl(var(--ink))',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {doc.filename}
                  </div>
                  <div style={{
                    fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 2,
                    display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap',
                  }}>
                    <span>{formatDate(doc.created_at)}</span>
                    <span style={{ opacity: 0.4 }}>·</span>
                    <span>{doc.chunk_count} chunks</span>
                  </div>
                </div>

                {/* Status Badge */}
                {doc.embedded ? (
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', gap: '4px',
                    fontSize: '.72rem', fontWeight: 700,
                    color: 'hsl(140,70%,45%)',
                    background: 'hsl(140,70%,45% / .12)',
                    border: '1px solid hsl(140,70%,45% / .25)',
                    padding: '3px 8px', borderRadius: 999, flexShrink: 0,
                  }}>
                    <CheckCircle size={11} />
                    Indexed
                  </span>
                ) : (
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', gap: '4px',
                    fontSize: '.72rem', fontWeight: 700,
                    color: 'hsl(40,85%,50%)',
                    background: 'hsl(40,85%,50% / .12)',
                    border: '1px solid hsl(40,85%,50% / .25)',
                    padding: '3px 8px', borderRadius: 999, flexShrink: 0,
                  }}>
                    <AlertTriangle size={11} />
                    Pending
                  </span>
                )}

                {/* Delete */}
                <button
                  onClick={() => deleteDoc(doc)}
                  disabled={deletingId === doc.id}
                  className="icon-btn"
                  style={{
                    color: 'hsl(var(--destructive))',
                    width: 30, height: 30, flexShrink: 0,
                    opacity: deletingId === doc.id ? 0.5 : 1,
                  }}
                  title="Delete document"
                >
                  {deletingId === doc.id
                    ? <Loader size={14} className="spin" />
                    : <Trash2 size={14} />}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
