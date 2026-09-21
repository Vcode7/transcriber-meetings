import { useState, useCallback, useEffect, useRef } from 'react'
import {
  Database, Upload, Trash2, CheckCircle, Loader, AlertTriangle,
  RefreshCw, Info, FileText, File, X, ChevronRight, ChevronDown, FolderUp,
  Eye, Tag, Layers, Sparkles, Video, Mic, Check
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'

interface GlobalContextDoc {
  id: string
  filename: string
  relative_path?: string
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

interface DocDetailData {
  document: GlobalContextDoc
  file_exists: boolean
  file_size: number
  extracted_text: string
  extracted_preview: string
  context_summary: string
  keywords: string[]
  technical_entities: string[]
  acronyms: string[]
  structure_stats: { paragraphs: number; headings: number; tables: number; lists: number }
  chunk_count: number
  chunks: Array<{
    chunk_index: number
    total_chunks?: number
    text: string
    block_type: string
    heading?: string
    section?: string
    chapter?: string
    page_number?: number
    keywords?: string[]
    technical_entities?: string[]
    acronyms?: string[]
    dates?: string[]
    project_names?: string[]
    entities?: string[]
    numbers?: string[]
    important_terms?: string[]
  }>
  pipeline_steps: Array<{ step: string; status: string; detail: string }>
}

interface MeetingContextFile {
  id: string
  filename: string
  type: string
  file_size: number
  file_exists: boolean
  created_at: string
  extracted_text_preview: string
  extracted_text_full: string
  summary: string
  keywords: string[]
  technical_entities: string[]
  acronyms: string[]
  structure_stats: { paragraphs: number; headings: number; tables: number; lists: number }
  chunk_count: number
  chunks: Array<{ chunk_index: number; text: string; block_type: string; heading?: string; keywords?: string[] }>
}

interface MeetingContextGroup {
  recording_id: string
  recording_title: string
  source_type: string
  created_at: string
  file_count: number
  files: MeetingContextFile[]
}

const ALLOWED_EXT_LABELS = 'PDF, DOCX, DOC, PPTX, PPT, TXT, MD, PNG, JPG, EXCEL, CSV'

function formatDate(str: string) {
  try {
    return new Date(str).toLocaleDateString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch { return str }
}

function formatBytes(bytes: number) {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'pdf': return '📄'
    case 'docx': case 'doc': return '📝'
    case 'pptx': case 'ppt': return '📊'
    case 'txt': case 'md': return '📃'
    case 'png': case 'jpg': case 'jpeg': case 'webp': return '🖼️'
    case 'xlsx': case 'xls': case 'csv': return '📈'
    default: return '📎'
  }
}

async function getFilesFromDataTransfer(items: DataTransferItemList): Promise<{ file: File; relPath: string }[]> {
  const result: { file: File; relPath: string }[] = []

  async function traverseEntry(entry: any, path: string) {
    if (entry.isFile) {
      await new Promise<void>((resolve) => {
        entry.file((file: File) => {
          result.push({ file, relPath: path ? `${path}/${file.name}` : file.name })
          resolve()
        })
      })
    } else if (entry.isDirectory) {
      const reader = entry.createReader()
      const entries: any[] = await new Promise((resolve) => {
        reader.readEntries((ents: any[]) => resolve(ents))
      })
      for (const childEntry of entries) {
        await traverseEntry(childEntry, path ? `${path}/${entry.name}` : entry.name)
      }
    }
  }

  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    if (item.kind === 'file') {
      const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null
      if (entry) {
        await traverseEntry(entry, '')
      } else {
        const file = item.getAsFile()
        if (file) result.push({ file, relPath: file.name })
      }
    }
  }

  return result
}

export default function GlobalContext() {
  const [activeTab, setActiveTab] = useState<'global' | 'meeting'>('global')
  const [docs, setDocs] = useState<GlobalContextDoc[]>([])
  const [status, setStatus] = useState<StatusInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  
  // Inspection Modal State
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [docDetail, setDocDetail] = useState<DocDetailData | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [extractingFullText, setExtractingFullText] = useState(false)
  const [showFullExtractedText, setShowFullExtractedText] = useState(false)
  const [activeDetailSection, setActiveDetailSection] = useState<'summary' | 'text' | 'keywords' | 'chunks'>('summary')

  const handleExtractFullText = async () => {
    if (!selectedDocId) return
    setExtractingFullText(true)
    try {
      const res = await api.post(`/global-context/doc/${selectedDocId}/extract-text`)
      if (docDetail) {
        setDocDetail({
          ...docDetail,
          extracted_text: res.data.extracted_text,
          extracted_preview: res.data.extracted_preview,
        })
      }
      setShowFullExtractedText(true)
      toast.success('Full text extracted successfully')
    } catch (err: any) {
      toast.error('Failed to extract full text: ' + (err.response?.data?.detail || err.message))
    } finally {
      setExtractingFullText(false)
    }
  }

  // Meeting Context Tab State
  const [meetings, setMeetings] = useState<MeetingContextGroup[]>([])
  const [meetingsLoading, setMeetingsLoading] = useState(false)
  const [expandedMeetings, setExpandedMeetings] = useState<Record<string, boolean>>({})
  const [expandedFiles, setExpandedFiles] = useState<Record<string, boolean>>({})
  const [expandedContexts, setExpandedContexts] = useState<Record<string, boolean>>({})

  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const fetchDocs = useCallback(async () => {
    try {
      const [docsRes, statusRes] = await Promise.all([
        api.get('/global-context/'),
        api.get('/global-context/status'),
      ])
      setDocs(docsRes.data.documents || [])
      setStatus(statusRes.data)
    } catch {
      toast.error('Failed to load global context documents')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchMeetingContext = useCallback(async () => {
    setMeetingsLoading(true)
    try {
      const res = await api.get('/global-context/meeting-context')
      setMeetings(res.data.meetings || [])
      if (res.data.meetings && res.data.meetings.length > 0) {
        setExpandedMeetings(prev => ({ [res.data.meetings[0].recording_id]: true, ...prev }))
      }
    } catch {
      toast.error('Failed to load meeting context hierarchy')
    } finally {
      setMeetingsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchDocs()
  }, [fetchDocs])

  useEffect(() => {
    if (activeTab === 'meeting') {
      fetchMeetingContext()
    }
  }, [activeTab, fetchMeetingContext])

  const inspectDocument = async (docId: string) => {
    setSelectedDocId(docId)
    setDetailLoading(true)
    setShowFullExtractedText(false)
    setActiveDetailSection('summary')
    try {
      const res = await api.get(`/global-context/doc/${docId}`)
      setDocDetail(res.data)
    } catch {
      toast.error('Failed to inspect document details')
      setSelectedDocId(null)
    } finally {
      setDetailLoading(false)
    }
  }

  const uploadFiles = async (files: File[], relativePaths?: string[]) => {
    if (!files.length) return
    setUploading(true)
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    if (relativePaths && relativePaths.length) {
      form.append('relative_paths', JSON.stringify(relativePaths))
    }
    try {
      const res = await api.post('/global-context/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const { uploaded, skipped_duplicates, skipped_unsupported } = res.data
      const embedded = uploaded.filter((u: any) => u.embedded).length
      const failed = uploaded.filter((u: any) => u.error).length
      const skippedUnsuppCount = skipped_unsupported?.length || 0

      const msgs = []
      if (embedded > 0) msgs.push(`${embedded} document${embedded !== 1 ? 's' : ''} indexed`)
      if (skipped_duplicates > 0) msgs.push(`${skipped_duplicates} duplicate${skipped_duplicates !== 1 ? 's' : ''} skipped`)
      if (skippedUnsuppCount > 0) msgs.push(`${skippedUnsuppCount} unsupported file${skippedUnsuppCount !== 1 ? 's' : ''} skipped`)
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

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)

    const items = e.dataTransfer.items
    if (items && items.length > 0) {
      const fileEntries = await getFilesFromDataTransfer(items)
      if (fileEntries.length > 0) {
        const files = fileEntries.map(e => e.file)
        const relPaths = fileEntries.map(e => e.relPath)
        uploadFiles(files, relPaths)
        return
      }
    }

    const files = Array.from(e.dataTransfer.files)
    uploadFiles(files)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length) uploadFiles(files)
    e.target.value = ''
  }

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length) {
      const relPaths = files.map(f => f.webkitRelativePath || f.name)
      uploadFiles(files, relPaths)
    }
    e.target.value = ''
  }

  const deleteDoc = async (doc: GlobalContextDoc, e?: React.MouseEvent) => {
    if (e) e.stopPropagation()
    if (!confirm(`Delete "${doc.filename}" from the knowledge base?`)) return
    setDeletingId(doc.id)
    try {
      await api.delete(`/global-context/${doc.id}`)
      toast.success(`"${doc.filename}" removed`)
      if (selectedDocId === doc.id) setSelectedDocId(null)
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

  const toggleMeetingExpand = (recordingId: string) => {
    setExpandedMeetings(prev => ({ ...prev, [recordingId]: !prev[recordingId] }))
  }

  const toggleFileExpand = (fileId: string) => {
    setExpandedFiles(prev => ({ ...prev, [fileId]: !prev[fileId] }))
  }

  const toggleContextExpand = (fileId: string) => {
    setExpandedContexts(prev => ({ ...prev, [fileId]: !prev[fileId] }))
  }

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* ── Panel Header ── */}
      <div className="panel-header" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', paddingBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '100%' }}>
          <div style={{
            width: 36, height: 36, borderRadius: '10px', flexShrink: 0,
            background: 'linear-gradient(135deg, hsl(260,85%,60% / .18), hsl(220,80%,60% / .08))',
            border: '1.5px solid hsl(260,85%,60% / .25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Database size={18} style={{ color: 'hsl(260,85%,65%)' }} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1>Knowledge Repository & Context Store</h1>
            <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '2px' }}>
              Manage global organizational documents and inspect meeting context files retrieved during MoM generation
            </p>
          </div>
          {docs.length > 0 && activeTab === 'global' && (
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

        {/* ── Navigation Tabs ── */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid hsl(var(--border) / .3)' }}>
          <button
            onClick={() => setActiveTab('global')}
            style={{
              display: 'flex', alignItems: 'center', gap: '7px',
              padding: '.65rem 1.2rem', fontSize: '.85rem', fontWeight: 700,
              fontFamily: 'Inter, sans-serif', cursor: 'pointer', background: 'none', border: 'none',
              color: activeTab === 'global' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              borderBottom: activeTab === 'global' ? '2.5px solid hsl(var(--accent))' : '2.5px solid transparent',
              transition: 'all .15s ease',
            }}
          >
            <Database size={15} />
            Global Context
            {docs.length > 0 && (
              <span style={{
                fontSize: '.7rem', padding: '1px 6px', borderRadius: 999,
                background: activeTab === 'global' ? 'hsl(var(--accent) / .15)' : 'hsl(var(--muted))',
                color: activeTab === 'global' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              }}>
                {docs.length}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('meeting')}
            style={{
              display: 'flex', alignItems: 'center', gap: '7px',
              padding: '.65rem 1.2rem', fontSize: '.85rem', fontWeight: 700,
              fontFamily: 'Inter, sans-serif', cursor: 'pointer', background: 'none', border: 'none',
              color: activeTab === 'meeting' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              borderBottom: activeTab === 'meeting' ? '2.5px solid hsl(var(--accent))' : '2.5px solid transparent',
              transition: 'all .15s ease',
            }}
          >
            <Layers size={15} />
            Meeting Context
            {meetings.length > 0 && (
              <span style={{
                fontSize: '.7rem', padding: '1px 6px', borderRadius: 999,
                background: activeTab === 'meeting' ? 'hsl(var(--accent) / .15)' : 'hsl(var(--muted))',
                color: activeTab === 'meeting' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              }}>
                {meetings.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ── Scrollable Content ── */}
      <div className="page-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', background: 'transparent', padding: '2rem 2.5rem 4rem' }}>
        
        {/* ── TAB 1: GLOBAL CONTEXT ── */}
        {activeTab === 'global' && (
          <>
            <div style={{
              fontSize: '.85rem',
              color: 'hsl(var(--pencil))',
              lineHeight: 1.5,
              marginTop: '-0.5rem',
            }}>
              These documents are indexed using <strong>{status?.embedding_model || 'Qwen3-Embedding-0.6B'}</strong> and
              retrieved automatically when generating Stage 2 & 3 MoM points. Click any document to inspect full extracted context and keywords.
            </div>

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

            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              style={{
                borderRadius: 16,
                border: `2px dashed ${dragging ? 'hsl(260,85%,65%)' : 'hsl(var(--border))'}`,
                background: dragging
                  ? 'hsl(260,85%,60% / .06)'
                  : 'hsl(var(--card))',
                padding: '2.5rem',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                gap: '0.75rem',
                transition: 'all 0.2s ease',
                position: 'relative',
              }}
            >
              {uploading ? (
                <>
                  <Loader size={32} className="spin" style={{ color: 'hsl(260,85%,65%)' }} />
                  <p style={{ margin: 0, fontSize: '.9rem', color: 'hsl(var(--pencil))' }}>
                    Processing, extracting & indexing documents...
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
                      Drag & drop files or entire folders here
                    </p>
                    <p style={{ margin: '4px 0 0', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                      Supports {ALLOWED_EXT_LABELS} · Max 50 MB per file
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '10px', marginTop: '.75rem' }}>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => fileInputRef.current?.click()}
                      style={{ fontSize: '.84rem', padding: '.5rem 1.1rem' }}
                    >
                      <Upload size={14} style={{ marginRight: '6px' }} /> Upload Files
                    </button>

                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => folderInputRef.current?.click()}
                      style={{ fontSize: '.84rem', padding: '.5rem 1.1rem' }}
                    >
                      <FolderUp size={14} style={{ marginRight: '6px' }} /> Import Folder
                    </button>
                  </div>
                </>
              )}

              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />

              <input
                ref={folderInputRef}
                type="file"
                // @ts-expect-error webkitdirectory is standard in HTML5 directory pickers
                webkitdirectory=""
                directory=""
                multiple
                onChange={handleFolderSelect}
                style={{ display: 'none' }}
              />
            </div>

            <div style={{
              borderRadius: 10, padding: '.75rem 1rem',
              background: 'hsl(220,80%,60% / .07)',
              border: '1px solid hsl(220,80%,60% / .2)',
              display: 'flex', alignItems: 'flex-start', gap: '10px',
            }}>
              <Info size={15} style={{ color: 'hsl(220,80%,65%)', flexShrink: 0, marginTop: 1 }} />
              <p style={{ margin: 0, fontSize: '.8rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>
                Documents uploaded here are retrieved during Stage 2 & 3 MoM enhancement to provide organizational context. Click any document card to view extracted content, summary, and generated keywords.
              </p>
            </div>

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
                <p style={{ margin: '4px 0 0', fontSize: '.8rem' }}>Upload organizational documents or folders to get started.</p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{
                  fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--pencil))',
                  textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '0.25rem',
                }}>
                  Knowledge Documents ({docs.length}) — Click to Inspect
                </div>
                {docs.map(doc => (
                  <div
                    key={doc.id}
                    onClick={() => inspectDocument(doc.id)}
                    style={{
                      borderRadius: 12, padding: '0.9rem 1.1rem',
                      background: 'hsl(var(--card))',
                      border: selectedDocId === doc.id ? '1.5px solid hsl(var(--accent))' : '1.5px solid hsl(var(--border) / .5)',
                      display: 'flex', alignItems: 'center', gap: '0.9rem',
                      cursor: 'pointer', transition: 'all 0.15s ease',
                      boxShadow: selectedDocId === doc.id ? '0 0 0 2px hsl(var(--accent) / .15)' : 'none',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'hsl(var(--accent) / .6)' }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = selectedDocId === doc.id ? 'hsl(var(--accent))' : 'hsl(var(--border) / .5)' }}
                  >
                    <div style={{ fontSize: '1.4rem', flexShrink: 0, lineHeight: 1 }}>
                      {getFileIcon(doc.filename)}
                    </div>

                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <span style={{
                          fontSize: '.9rem', fontWeight: 600, color: 'hsl(var(--ink))',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                          {doc.filename}
                        </span>

                        {doc.relative_path && doc.relative_path !== doc.filename && (
                          <span style={{
                            fontSize: '.72rem', color: 'hsl(var(--accent))',
                            fontFamily: 'JetBrains Mono, monospace', background: 'hsl(var(--accent) / .08)',
                            padding: '2px 7px', borderRadius: '4px', border: '1px solid hsl(var(--accent) / .2)',
                            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '320px',
                          }} title={doc.relative_path}>
                            📁 {doc.relative_path}
                          </span>
                        )}
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

                    <button
                      onClick={(e) => { e.stopPropagation(); inspectDocument(doc.id) }}
                      className="btn btn-ghost"
                      style={{ fontSize: '.75rem', gap: '4px', padding: '.3rem .6rem', flexShrink: 0 }}
                    >
                      <Eye size={13} /> Details
                    </button>

                    {doc.embedded ? (
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: '4px',
                        fontSize: '.72rem', fontWeight: 700,
                        color: 'hsl(140,70%,45%)',
                        background: 'hsl(140,70%,45% / .12)',
                        border: '1px solid hsl(140,70%,45% / .25)',
                        padding: '3px 8px', borderRadius: 999, flexShrink: 0,
                      }}>
                        <CheckCircle size={11} /> Indexed
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
                        <AlertTriangle size={11} /> Pending
                      </span>
                    )}

                    <button
                      onClick={(e) => deleteDoc(doc, e)}
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
          </>
        )}

        {/* ── TAB 2: MEETING CONTEXT HIERARCHY ── */}
        {activeTab === 'meeting' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{
              borderRadius: 10, padding: '.75rem 1rem',
              background: 'hsl(260,85%,60% / .07)',
              border: '1px solid hsl(260,85%,60% / .2)',
              display: 'flex', alignItems: 'flex-start', gap: '10px',
            }}>
              <Info size={15} style={{ color: 'hsl(260,85%,65%)', flexShrink: 0, marginTop: 1 }} />
              <p style={{ margin: 0, fontSize: '.8rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>
                Meeting Context hierarchy showing uploaded attachments per meeting. Click any meeting to expand files, extracted context, and generated keywords/structured data.
              </p>
            </div>

            {meetingsLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
                <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              </div>
            ) : meetings.length === 0 ? (
              <div style={{
                textAlign: 'center', padding: '3rem',
                borderRadius: 12, border: '1.5px dashed hsl(var(--border) / .5)',
                color: 'hsl(var(--pencil))',
              }}>
                <Layers size={32} style={{ margin: '0 auto 12px', opacity: 0.35 }} />
                <p style={{ margin: 0, fontSize: '.9rem' }}>No meeting context uploaded yet.</p>
                <p style={{ margin: '4px 0 0', fontSize: '.8rem' }}>Upload attachments or agenda files to a meeting recording to see them listed here.</p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {meetings.map(m => {
                  const isMtgExpanded = expandedMeetings[m.recording_id]
                  return (
                    <div
                      key={m.recording_id}
                      style={{
                        borderRadius: 14,
                        background: 'hsl(var(--card))',
                        border: '1.5px solid hsl(var(--border) / .5)',
                        overflow: 'hidden',
                        transition: 'all .2s ease',
                      }}
                    >
                      <div
                        onClick={() => toggleMeetingExpand(m.recording_id)}
                        style={{
                          padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '10px',
                          cursor: 'pointer', background: isMtgExpanded ? 'hsl(var(--accent) / .04)' : 'none',
                          borderBottom: isMtgExpanded ? '1px solid hsl(var(--border) / .3)' : 'none',
                        }}
                      >
                        {isMtgExpanded ? <ChevronDown size={16} style={{ color: 'hsl(var(--accent))' }} /> : <ChevronRight size={16} style={{ color: 'hsl(var(--pencil))' }} />}
                        
                        <div style={{
                          width: 30, height: 30, borderRadius: '8px',
                          background: 'hsl(var(--accent) / .12)', border: '1px solid hsl(var(--accent) / .25)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        }}>
                          {m.source_type === 'video' ? <Video size={15} style={{ color: 'hsl(var(--accent))' }} /> : <Mic size={15} style={{ color: 'hsl(var(--accent))' }} />}
                        </div>

                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                            {m.recording_title}
                          </span>
                          <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 2, display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span>{formatDate(m.created_at)}</span>
                            <span>·</span>
                            <span>{m.file_count} file{m.file_count !== 1 ? 's' : ''} attached</span>
                          </div>
                        </div>

                        <span style={{
                          fontSize: '.72rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
                          padding: '3px 9px', borderRadius: 999,
                          background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))',
                          border: '1px solid hsl(var(--accent) / .25)',
                        }}>
                          {m.file_count} File{m.file_count !== 1 ? 's' : ''}
                        </span>
                      </div>

                      {isMtgExpanded && (
                        <div style={{ padding: '.8rem 1.25rem 1.25rem 2.2rem', display: 'flex', flexDirection: 'column', gap: '.8rem', borderTop: '1px dashed hsl(var(--border) / .2)' }}>
                          {m.files.map(f => {
                            const isFileExpanded = expandedFiles[f.id]
                            const isContextExpanded = expandedContexts[f.id]
                            return (
                              <div
                                key={f.id}
                                style={{
                                  borderRadius: 10,
                                  background: 'hsl(var(--paper))',
                                  border: '1px solid hsl(var(--border) / .4)',
                                  padding: '.75rem 1rem',
                                  display: 'flex', flexDirection: 'column', gap: '.6rem',
                                }}
                              >
                                <div
                                  onClick={() => toggleFileExpand(f.id)}
                                  style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
                                >
                                  {isFileExpanded ? <ChevronDown size={14} style={{ color: 'hsl(var(--accent))' }} /> : <ChevronRight size={14} style={{ color: 'hsl(var(--pencil))' }} />}
                                  <span style={{ fontSize: '1.1rem' }}>{getFileIcon(f.filename)}</span>
                                  <span style={{ fontSize: '.85rem', fontWeight: 600, color: 'hsl(var(--ink))', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {f.filename}
                                  </span>
                                  <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                                    {formatBytes(f.file_size)}
                                  </span>
                                  <span style={{
                                    fontSize: '.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 4,
                                    background: f.type === 'context' ? 'hsl(260,80%,60% / .12)' : 'hsl(200,80%,50% / .12)',
                                    color: f.type === 'context' ? 'hsl(260,80%,60%)' : 'hsl(200,80%,50%)',
                                    border: `1px solid ${f.type === 'context' ? 'hsl(260,80%,60% / .25)' : 'hsl(200,80%,50% / .25)'}`,
                                    textTransform: 'uppercase',
                                  }}>
                                    {f.type}
                                  </span>
                                </div>

                                {isFileExpanded && (
                                  <div style={{ paddingLeft: '1.4rem', borderLeft: '2px solid hsl(var(--accent) / .2)', display: 'flex', flexDirection: 'column', gap: '.75rem', marginTop: '.2rem' }}>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
                                      <div
                                        onClick={() => toggleContextExpand(f.id)}
                                        style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}
                                      >
                                        {isContextExpanded ? <ChevronDown size={13} style={{ color: 'hsl(var(--accent))' }} /> : <ChevronRight size={13} style={{ color: 'hsl(var(--pencil))' }} />}
                                        <FileText size={13} style={{ color: 'hsl(var(--accent))' }} />
                                        Extracted Context & Summary
                                      </div>

                                      <div style={{
                                        padding: '.6rem .8rem', borderRadius: 8,
                                        background: 'hsl(var(--card))', border: '1px solid hsl(var(--border) / .3)',
                                        fontSize: '.78rem', color: 'hsl(var(--pencil))', lineHeight: 1.5,
                                      }}>
                                        {f.summary ? (
                                          <p style={{ margin: 0, fontWeight: 500, color: 'hsl(var(--ink))' }}>{f.summary}</p>
                                        ) : f.extracted_text_preview ? (
                                          <p style={{ margin: 0 }}>{f.extracted_text_preview}</p>
                                        ) : (
                                          <p style={{ margin: 0, fontStyle: 'italic' }}>No text extracted from file.</p>
                                        )}
                                      </div>

                                      {isContextExpanded && f.extracted_text_full && (
                                        <div style={{
                                          padding: '.75rem', borderRadius: 8,
                                          background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)',
                                          maxHeight: '200px', overflowY: 'auto', fontSize: '.75rem',
                                          fontFamily: 'monospace', whiteSpace: 'pre-wrap', color: 'hsl(var(--ink))',
                                        }}>
                                          {f.extracted_text_full}
                                        </div>
                                      )}
                                    </div>

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
                                      <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                        <Tag size={13} style={{ color: 'hsl(var(--accent))' }} />
                                        Keywords & Extracted Metadata
                                      </div>

                                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                        {f.keywords.map(kw => (
                                          <span key={kw} style={{
                                            fontSize: '.7rem', fontWeight: 600, padding: '2px 7px', borderRadius: 6,
                                            background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))',
                                            border: '1px solid hsl(var(--accent) / .2)',
                                          }}>
                                            #{kw}
                                          </span>
                                        ))}

                                        {f.technical_entities.map(ent => (
                                          <span key={ent} style={{
                                            fontSize: '.7rem', fontWeight: 600, padding: '2px 7px', borderRadius: 6,
                                            background: 'hsl(140,70%,45% / .1)', color: 'hsl(140,70%,38%)',
                                            border: '1px solid hsl(140,70%,45% / .2)',
                                          }}>
                                            ⚙ {ent}
                                          </span>
                                        ))}

                                        {f.acronyms.map(acr => (
                                          <span key={acr} style={{
                                            fontSize: '.7rem', fontWeight: 600, padding: '2px 7px', borderRadius: 6,
                                            background: 'hsl(40,85%,50% / .1)', color: 'hsl(40,85%,40%)',
                                            border: '1px solid hsl(40,85%,50% / .2)',
                                          }}>
                                            🔤 {acr}
                                          </span>
                                        ))}

                                        {f.keywords.length === 0 && f.technical_entities.length === 0 && f.acronyms.length === 0 && (
                                          <span style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                                            No structured keywords generated.
                                          </span>
                                        )}
                                      </div>

                                      <div style={{ display: 'flex', gap: '12px', fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: '2px' }}>
                                        <span>Paragraphs: {f.structure_stats.paragraphs}</span>
                                        <span>·</span>
                                        <span>Headings: {f.structure_stats.headings}</span>
                                        <span>·</span>
                                        <span>Tables: {f.structure_stats.tables}</span>
                                        <span>·</span>
                                        <span>Chunks: {f.chunk_count}</span>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── DOCUMENT INSPECTION MODAL ── */}
      {selectedDocId && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0, 0, 0, 0.65)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem',
        }}>
          <div style={{
            width: '100%', maxWidth: '850px', maxHeight: '88vh',
            background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--border) / .6)',
            borderRadius: '16px', display: 'flex', flexDirection: 'column',
            boxShadow: '0 20px 50px rgba(0,0,0,0.35)', overflow: 'hidden',
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '1.2rem 1.5rem', borderBottom: '1px solid hsl(var(--border) / .3)',
              display: 'flex', alignItems: 'center', gap: '1rem', background: 'hsl(var(--paper))',
            }}>
              <div style={{ fontSize: '1.8rem', lineHeight: 1 }}>
                {getFileIcon(docDetail?.document?.filename || '')}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {docDetail?.document?.filename || 'Document Details'}
                </h2>
                <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: 2, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{docDetail?.document?.relative_path || docDetail?.document?.filename}</span>
                  <span>·</span>
                  <span>{formatBytes(docDetail?.file_size || 0)}</span>
                </div>
              </div>

              <button
                onClick={() => setSelectedDocId(null)}
                className="icon-btn"
                style={{ width: 32, height: 32, flexShrink: 0 }}
              >
                <X size={18} />
              </button>
            </div>

            {/* Modal Body */}
            {detailLoading || !docDetail ? (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '5rem' }}>
                <Loader size={32} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflowY: 'auto', padding: '1.5rem', gap: '1.5rem' }}>
                
                {/* Processing Pipeline Status */}
                <div>
                  <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.6rem' }}>
                    Document Processing & Indexing Pipeline
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '.6rem' }}>
                    {docDetail.pipeline_steps.map(step => (
                      <div key={step.step} style={{
                        padding: '.6rem .75rem', borderRadius: 10,
                        background: step.status === 'completed' ? 'hsl(140,70%,45% / .08)' : 'hsl(var(--paper))',
                        border: `1px solid ${step.status === 'completed' ? 'hsl(140,70%,45% / .25)' : 'hsl(var(--border) / .3)'}`,
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '.75rem', fontWeight: 700, color: step.status === 'completed' ? 'hsl(140,70%,38%)' : 'hsl(var(--pencil))' }}>
                          {step.status === 'completed' ? <Check size={12} /> : <Loader size={12} className="spin" />}
                          {step.step}
                        </div>
                        <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {step.detail}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Section Switcher */}
                <div style={{ display: 'flex', gap: '6px', borderBottom: '1px solid hsl(var(--border) / .3)', paddingBottom: '.4rem' }}>
                  {[
                    { id: 'summary', label: 'Context Summary', icon: Sparkles },
                    { id: 'text', label: 'Extracted Content', icon: FileText },
                    { id: 'keywords', label: 'Keywords & Entities', icon: Tag },
                    { id: 'chunks', label: `Vector Chunks (${docDetail.chunk_count})`, icon: Layers },
                  ].map(sec => {
                    const Icon = sec.icon
                    const isActive = activeDetailSection === sec.id
                    return (
                      <button
                        key={sec.id}
                        onClick={() => setActiveDetailSection(sec.id as any)}
                        style={{
                          display: 'flex', alignItems: 'center', gap: '6px',
                          padding: '.45rem .8rem', fontSize: '.8rem', fontWeight: 700,
                          borderRadius: 8, cursor: 'pointer', border: 'none',
                          background: isActive ? 'hsl(var(--accent) / .12)' : 'transparent',
                          color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                          transition: 'all .15s ease',
                        }}
                      >
                        <Icon size={14} /> {sec.label}
                      </button>
                    )
                  })}
                </div>

                {/* Section Content */}
                {activeDetailSection === 'summary' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{
                      padding: '1rem 1.2rem', borderRadius: 12,
                      background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .4)',
                      fontSize: '.85rem', color: 'hsl(var(--ink))', lineHeight: 1.6, whiteSpace: 'pre-wrap',
                    }}>
                      {docDetail.context_summary || 'No context summary available for this file.'}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                      <div style={{ padding: '.75rem', borderRadius: 8, background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)' }}>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'hsl(var(--ink))' }}>{docDetail.structure_stats.paragraphs}</div>
                        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase' }}>Paragraphs</div>
                      </div>
                      <div style={{ padding: '.75rem', borderRadius: 8, background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)' }}>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'hsl(var(--ink))' }}>{docDetail.structure_stats.headings}</div>
                        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase' }}>Headings</div>
                      </div>
                      <div style={{ padding: '.75rem', borderRadius: 8, background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)' }}>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'hsl(var(--ink))' }}>{docDetail.structure_stats.tables}</div>
                        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase' }}>Tables</div>
                      </div>
                      <div style={{ padding: '.75rem', borderRadius: 8, background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)' }}>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'hsl(var(--ink))' }}>{docDetail.chunk_count}</div>
                        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase' }}>Vector Chunks</div>
                      </div>
                    </div>
                  </div>
                )}

                {activeDetailSection === 'text' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                    {docDetail.extracted_text || docDetail.extracted_preview ? (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
                            Showing {showFullExtractedText ? 'full extracted text' : 'text preview (first 1,200 chars)'}
                          </span>
                          <button
                            onClick={() => setShowFullExtractedText(!showFullExtractedText)}
                            className="btn btn-secondary"
                            style={{ fontSize: '.75rem', padding: '.25rem .6rem' }}
                          >
                            {showFullExtractedText ? 'Show Preview' : 'Show Full Text'}
                          </button>
                        </div>

                        <div style={{
                          padding: '1rem', borderRadius: 10,
                          background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .4)',
                          maxHeight: '350px', overflowY: 'auto', fontSize: '.8rem',
                          fontFamily: 'JetBrains Mono, monospace', whiteSpace: 'pre-wrap', color: 'hsl(var(--ink))',
                          lineHeight: 1.5,
                        }}>
                          {showFullExtractedText ? docDetail.extracted_text : docDetail.extracted_preview}
                        </div>
                      </>
                    ) : (
                      <div style={{
                        padding: '2rem 1.5rem', borderRadius: 12,
                        background: 'hsl(var(--paper))', border: '1px dashed hsl(var(--border) / .4)',
                        display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: '1rem',
                      }}>
                        <div style={{
                          width: 48, height: 48, borderRadius: '50%',
                          background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}>
                          <FileText size={24} />
                        </div>
                        <div>
                          <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                            Full Text Extraction Available On-Demand
                          </div>
                          <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: 4, maxWidth: 460, lineHeight: 1.5 }}>
                            Document inspection opens instantly using indexed ChromaDB vector chunks. Click below to run text extraction and preview the full raw text on demand.
                          </div>
                        </div>

                        <button
                          onClick={handleExtractFullText}
                          disabled={extractingFullText}
                          className="btn btn-primary"
                          style={{
                            display: 'flex', alignItems: 'center', gap: '8px',
                            padding: '.55rem 1.2rem', fontSize: '.82rem', fontWeight: 700,
                            borderRadius: 8, cursor: extractingFullText ? 'not-allowed' : 'pointer',
                          }}
                        >
                          {extractingFullText ? (
                            <>
                              <Loader size={16} className="spin" /> Extracting Text...
                            </>
                          ) : (
                            <>
                              <Sparkles size={16} /> Extract Full Text
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {activeDetailSection === 'keywords' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div>
                      <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: '.4rem' }}>
                        Extracted Keywords ({docDetail.keywords.length})
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {docDetail.keywords.map(kw => (
                          <span key={kw} style={{
                            fontSize: '.74rem', fontWeight: 600, padding: '3px 8px', borderRadius: 6,
                            background: 'hsl(var(--accent) / .12)', color: 'hsl(var(--accent))',
                            border: '1px solid hsl(var(--accent) / .25)',
                          }}>
                            #{kw}
                          </span>
                        ))}
                        {docDetail.keywords.length === 0 && <span style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>None extracted.</span>}
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: '.4rem' }}>
                        Technical Entities ({docDetail.technical_entities.length})
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {docDetail.technical_entities.map(ent => (
                          <span key={ent} style={{
                            fontSize: '.74rem', fontWeight: 600, padding: '3px 8px', borderRadius: 6,
                            background: 'hsl(140,70%,45% / .12)', color: 'hsl(140,70%,38%)',
                            border: '1px solid hsl(140,70%,45% / .25)',
                          }}>
                            ⚙ {ent}
                          </span>
                        ))}
                        {docDetail.technical_entities.length === 0 && <span style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>None extracted.</span>}
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: '.4rem' }}>
                        Acronyms ({docDetail.acronyms.length})
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {docDetail.acronyms.map(acr => (
                          <span key={acr} style={{
                            fontSize: '.74rem', fontWeight: 600, padding: '3px 8px', borderRadius: 6,
                            background: 'hsl(40,85%,50% / .12)', color: 'hsl(40,85%,40%)',
                            border: '1px solid hsl(40,85%,50% / .25)',
                          }}>
                            🔤 {acr}
                          </span>
                        ))}
                        {docDetail.acronyms.length === 0 && <span style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>None extracted.</span>}
                      </div>
                    </div>
                  </div>
                )}

                {activeDetailSection === 'chunks' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem', maxHeight: '380px', overflowY: 'auto' }}>
                    {docDetail.chunks.map(chk => (
                      <div key={chk.chunk_index} style={{
                        padding: '.75rem 1rem', borderRadius: 10,
                        background: 'hsl(var(--paper))', border: '1px solid hsl(var(--border) / .3)',
                        display: 'flex', flexDirection: 'column', gap: '.5rem',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                          <span style={{ fontWeight: 700, color: 'hsl(var(--accent))' }}>
                            Chunk #{chk.chunk_index + 1} {chk.total_chunks ? `of ${chk.total_chunks}` : ''}
                          </span>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            {chk.section && <span style={{ fontStyle: 'italic' }}>{chk.section}</span>}
                            {chk.page_number && <span>Page {chk.page_number}</span>}
                            <span style={{ textTransform: 'uppercase', fontWeight: 700, padding: '1px 6px', borderRadius: 4, background: 'hsl(var(--accent) / .08)', color: 'hsl(var(--accent))' }}>{chk.block_type}</span>
                          </div>
                        </div>

                        {chk.heading && <div style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{chk.heading}</div>}

                        <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>{chk.text}</div>

                        {/* Chunk Metadata Tags */}
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', paddingTop: '.25rem', borderTop: '1px dashed hsl(var(--border) / .2)' }}>
                          {chk.project_names?.map(p => (
                            <span key={p} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(280,70%,50% / .12)', color: 'hsl(280,70%,45%)' }}>🚀 {p}</span>
                          ))}
                          {chk.dates?.map(d => (
                            <span key={d} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(200,80%,50% / .12)', color: 'hsl(200,80%,40%)' }}>📅 {d}</span>
                          ))}
                          {chk.technical_entities?.map(e => (
                            <span key={e} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(140,70%,45% / .12)', color: 'hsl(140,70%,35%)' }}>⚙ {e}</span>
                          ))}
                          {chk.entities?.map(e => (
                            <span key={e} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(340,75%,50% / .12)', color: 'hsl(340,75%,45%)' }}>🏢 {e}</span>
                          ))}
                          {chk.numbers?.map(n => (
                            <span key={n} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(40,85%,50% / .12)', color: 'hsl(40,85%,35%)' }}>🔢 {n}</span>
                          ))}
                          {chk.keywords?.map(k => (
                            <span key={k} style={{ fontSize: '.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4, background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))' }}>#{k}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                    {docDetail.chunks.length === 0 && <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>No vector chunks stored.</p>}
                  </div>
                )}

              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
