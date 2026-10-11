import React, { useState, useEffect, useRef } from 'react'
import {
  X,
  UploadCloud,
  FileText,
  MessageSquare,
  Trash2,
  RotateCcw,
  Search,
  CheckCircle2,
  AlertCircle,
  Users,
  Calendar,
  ListOrdered,
  CheckSquare,
  Award,
  Layers,
  Copy,
  Check,
  Sparkles,
  FileUp,
  FileCode,
  ArrowRight,
  RefreshCw,
  FolderOpen
} from 'lucide-react'
import {
  uploadManualMoms,
  getManualMoms,
  getManualMomDetail,
  deleteManualMom,
  reindexManualMom,
} from '../api/aiChat'
import type { ManualMomItem, ManualMomDetail } from '../types/recording'

interface ManualMomLibraryModalProps {
  isOpen: boolean
  onClose: () => void
  onSelectMeetingForChat: (meetingId: string) => void
  onMoMsChanged?: () => void
}

export const ManualMomLibraryModal: React.FC<ManualMomLibraryModalProps> = ({
  isOpen,
  onClose,
  onSelectMeetingForChat,
  onMoMsChanged,
}) => {
  // ── State ──────────────────────────────────────────────────────────────────
  const [moms, setMoms] = useState<ManualMomItem[]>([])
  const [selectedMomId, setSelectedMomId] = useState<string | null>(null)
  const [selectedDetail, setSelectedDetail] = useState<ManualMomDetail | null>(null)
  const [loadingList, setLoadingList] = useState(false)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'structured' | 'raw' | 'chunks'>('structured')

  // Upload state
  const [uploading, setUploading] = useState(false)
  const [uploadProgressMsg, setUploadProgressMsg] = useState('')
  const [uploadResults, setUploadResults] = useState<{
    success: number
    fail: number
    errors: string[]
  } | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Actions state
  const [actionLoading, setActionLoading] = useState(false)
  const [copiedRaw, setCopiedRaw] = useState(false)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  // ── Fetch MoMs ─────────────────────────────────────────────────────────────
  const loadMomsList = async (preferredSelectId?: string) => {
    setLoadingList(true)
    try {
      const data = await getManualMoms()
      setMoms(data)
      if (preferredSelectId) {
        setSelectedMomId(preferredSelectId)
      } else if (!selectedMomId && data.length > 0) {
        setSelectedMomId(data[0].id)
      } else if (selectedMomId && !data.some(m => m.id === selectedMomId) && data.length > 0) {
        setSelectedMomId(data[0].id)
      }
    } catch (err) {
      console.error('Failed to load manual MoMs:', err)
    } finally {
      setLoadingList(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadMomsList()
    } else {
      setUploadResults(null)
      setDeleteConfirmId(null)
    }
  }, [isOpen])

  // ── Fetch Selected Detail ──────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedMomId) {
      setSelectedDetail(null)
      return
    }

    let isCurrent = true
    setLoadingDetail(true)
    getManualMomDetail(selectedMomId)
      .then(detail => {
        if (isCurrent) {
          setSelectedDetail(detail)
        }
      })
      .catch(err => {
        console.error('Failed to load MoM detail:', err)
        if (isCurrent) setSelectedDetail(null)
      })
      .finally(() => {
        if (isCurrent) setLoadingDetail(false)
      })

    return () => {
      isCurrent = false
    }
  }, [selectedMomId])

  if (!isOpen) return null

  // ── Upload Handler ─────────────────────────────────────────────────────────
  const handleFilesSelected = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const fileArray = Array.from(files)

    setUploading(true)
    setUploadResults(null)
    setUploadProgressMsg(`Uploading and parsing ${fileArray.length} file(s)...`)

    try {
      const res = await uploadManualMoms(fileArray)
      const errMsgs = res.errors?.map(e => `${e.filename}: ${e.error}`) || []
      setUploadResults({
        success: res.success_count || 0,
        fail: res.fail_count || 0,
        errors: errMsgs,
      })

      // Reload list and select the newest uploaded file
      const firstNewId = res.results?.[0]?.id
      await loadMomsList(firstNewId)
      onMoMsChanged?.()
    } catch (err: any) {
      setUploadResults({
        success: 0,
        fail: fileArray.length,
        errors: [err.message || 'Upload failed'],
      })
    } finally {
      setUploading(false)
      setUploadProgressMsg('')
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // ── Re-index Handler ───────────────────────────────────────────────────────
  const handleReindex = async () => {
    if (!selectedMomId) return
    setActionLoading(true)
    try {
      await reindexManualMom(selectedMomId)
      const updated = await getManualMomDetail(selectedMomId)
      setSelectedDetail(updated)
      await loadMomsList(selectedMomId)
      onMoMsChanged?.()
    } catch (err) {
      console.error('Failed to reindex MoM:', err)
    } finally {
      setActionLoading(false)
    }
  }

  // ── Delete Handler ─────────────────────────────────────────────────────────
  const handleDelete = async (id: string) => {
    setActionLoading(true)
    try {
      await deleteManualMom(id)
      setDeleteConfirmId(null)
      await loadMomsList()
      onMoMsChanged?.()
    } catch (err) {
      console.error('Failed to delete MoM:', err)
    } finally {
      setActionLoading(false)
    }
  }

  // ── Copy Raw Text ──────────────────────────────────────────────────────────
  const handleCopyRaw = () => {
    if (!selectedDetail?.raw_text) return
    navigator.clipboard.writeText(selectedDetail.raw_text)
    setCopiedRaw(true)
    setTimeout(() => setCopiedRaw(false), 2000)
  }

  // ── Filtered List ──────────────────────────────────────────────────────────
  const filteredMoms = moms.filter(m => {
    if (!searchQuery.trim()) return true
    const q = searchQuery.toLowerCase()
    return (
      m.title.toLowerCase().includes(q) ||
      m.filename.toLowerCase().includes(q) ||
      (m.meeting_date && m.meeting_date.toLowerCase().includes(q))
    )
  })

  const formatFileSize = (bytes: number): string => {
    if (!bytes || bytes <= 0) return '0 B'
    const kb = bytes / 1024
    if (kb < 1024) return `${kb.toFixed(1)} KB`
    return `${(kb / 1024).toFixed(2)} MB`
  }

  const formatExt = (filename: string, fileType: string): string => {
    const ext = fileType || filename.split('.').pop() || 'doc'
    return ext.replace('.', '').toUpperCase()
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(10, 15, 29, 0.75)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: '1.25rem',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && !uploading) onClose()
      }}
    >
      <div
        style={{
          background: 'hsl(var(--card))',
          border: '1.5px solid hsl(var(--border) / 0.5)',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '1280px',
          height: '90vh',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.5)',
          overflow: 'hidden',
        }}
      >
        {/* ═══════ Modal Header ═══════ */}
        <div
          style={{
            padding: '1rem 1.5rem',
            borderBottom: '1px solid hsl(var(--border) / 0.3)',
            background: 'hsl(var(--paper))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '.85rem' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, hsl(280 80% 55% / .2), hsl(210 80% 55% / .2))',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1.5px solid hsl(280 70% 60% / .4)',
              }}
            >
              <FileUp size={22} style={{ color: 'hsl(280 80% 60%)' }} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                  Meeting MoM Library & Upload
                </h2>
                <span
                  style={{
                    fontSize: '.72rem',
                    fontWeight: 700,
                    padding: '2px 9px',
                    borderRadius: '6px',
                    background: 'hsl(280 80% 50% / .18)',
                    color: 'hsl(280 80% 45%)',
                    letterSpacing: '0.02em',
                  }}
                >
                  Manual MoM
                </span>
                <span
                  style={{
                    fontSize: '.74rem',
                    fontWeight: 600,
                    padding: '2px 8px',
                    borderRadius: '12px',
                    background: 'hsl(var(--muted) / .5)',
                    color: 'hsl(var(--pencil))',
                  }}
                >
                  {moms.length} uploaded
                </span>
              </div>
              <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', margin: '2px 0 0', fontWeight: 500 }}>
                Upload multiple meeting minutes (.pdf, .docx, .txt, .md, .xlsx, .pptx). Extracted content & metadata are indexed for AI Chat retrieval.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            disabled={uploading}
            className="icon-btn"
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '9px',
              border: '1px solid hsl(var(--border) / .3)',
              background: 'hsl(var(--card))',
              color: 'hsl(var(--pencil))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: uploading ? 'not-allowed' : 'pointer',
              transition: 'all .15s',
            }}
            title="Close"
          >
            <X size={18} />
          </button>
        </div>

        {/* ═══════ Multi-File Upload Strip ═══════ */}
        <div
          style={{
            padding: '.85rem 1.5rem',
            background: isDragOver
              ? 'hsl(280 80% 50% / .1)'
              : 'hsl(var(--paper-deep) / .4)',
            borderBottom: '1px solid hsl(var(--border) / .25)',
            display: 'flex',
            flexDirection: 'column',
            gap: '.5rem',
            transition: 'background .2s',
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragOver(true)
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setIsDragOver(false)
            if (!uploading && e.dataTransfer.files) {
              handleFilesSelected(e.dataTransfer.files)
            }
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem' }}>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md,.pptx,.ppt,.xlsx,.xls,.csv"
                style={{ display: 'none' }}
                onChange={(e) => handleFilesSelected(e.target.files)}
                disabled={uploading}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                style={{
                  height: '38px',
                  padding: '0 1.1rem',
                  borderRadius: '9px',
                  background: 'linear-gradient(135deg, hsl(280 80% 55%), hsl(260 80% 60%))',
                  color: '#fff',
                  border: 'none',
                  fontWeight: 600,
                  fontSize: '.84rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '.5rem',
                  cursor: uploading ? 'not-allowed' : 'pointer',
                  boxShadow: '0 4px 12px hsl(280 80% 55% / .3)',
                  transition: 'opacity .15s',
                  opacity: uploading ? 0.7 : 1,
                }}
              >
                {uploading ? (
                  <>
                    <RefreshCw size={16} className="spin-animate" />
                    <span>Processing Files...</span>
                  </>
                ) : (
                  <>
                    <UploadCloud size={17} />
                    <span>Select MoM Files (Multiple)</span>
                  </>
                )}
              </button>

              <span style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                or drag and drop multiple MoM files here (PDF, Word DOCX, TXT, Markdown, PowerPoint, Excel)
              </span>
            </div>

            {/* Supported formats pills */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '.35rem', flexWrap: 'wrap' }}>
              {['PDF', 'DOCX', 'TXT', 'MD', 'PPTX', 'XLSX'].map((ext) => (
                <span
                  key={ext}
                  style={{
                    fontSize: '.68rem',
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border) / .3)',
                    color: 'hsl(var(--ink) / .8)',
                  }}
                >
                  {ext}
                </span>
              ))}
            </div>
          </div>

          {/* Upload progress message */}
          {uploadProgressMsg && (
            <div
              style={{
                fontSize: '.78rem',
                color: 'hsl(280 70% 50%)',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
              }}
            >
              <RefreshCw size={13} className="spin-animate" />
              <span>{uploadProgressMsg}</span>
            </div>
          )}

          {/* Upload feedback banner */}
          {uploadResults && (
            <div
              style={{
                padding: '.55rem .85rem',
                borderRadius: '8px',
                fontSize: '.78rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background:
                  uploadResults.fail === 0
                    ? 'hsl(142 70% 45% / .12)'
                    : 'hsl(38 90% 50% / .12)',
                border:
                  uploadResults.fail === 0
                    ? '1px solid hsl(142 70% 45% / .3)'
                    : '1px solid hsl(38 90% 50% / .3)',
                color:
                  uploadResults.fail === 0
                    ? 'hsl(142 70% 35%)'
                    : 'hsl(38 80% 40%)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                {uploadResults.fail === 0 ? (
                  <CheckCircle2 size={16} />
                ) : (
                  <AlertCircle size={16} />
                )}
                <span>
                  <strong>{uploadResults.success} file(s)</strong> processed and indexed successfully into ChromaDB.
                  {uploadResults.fail > 0 && ` (${uploadResults.fail} failed)`}
                </span>
              </div>
              <button
                onClick={() => setUploadResults(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '2px',
                  color: 'inherit',
                  display: 'flex',
                }}
              >
                <X size={14} />
              </button>
            </div>
          )}
        </div>

        {/* ═══════ Body Layout: Master-Detail ═══════ */}
        <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
          {/* ────── Left Pane: Uploaded MoM List ────── */}
          <div
            style={{
              width: '360px',
              borderRight: '1px solid hsl(var(--border) / .3)',
              background: 'hsl(var(--paper))',
              display: 'flex',
              flexDirection: 'column',
              flexShrink: 0,
            }}
          >
            {/* Search filter */}
            <div
              style={{
                padding: '.65rem .85rem',
                borderBottom: '1px solid hsl(var(--border) / .2)',
                display: 'flex',
                alignItems: 'center',
                gap: '.5rem',
                background: 'hsl(var(--card))',
              }}
            >
              <Search size={14} style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
              <input
                type="text"
                placeholder="Search uploaded MoMs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  flex: 1,
                  border: 'none',
                  background: 'transparent',
                  outline: 'none',
                  fontSize: '.82rem',
                  color: 'hsl(var(--ink))',
                }}
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    padding: 0,
                    color: 'hsl(var(--pencil))',
                  }}
                >
                  <X size={13} />
                </button>
              )}
            </div>

            {/* MoMs Scroll List */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '.5rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '.35rem',
              }}
            >
              {loadingList ? (
                <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'hsl(var(--pencil))', fontSize: '.8rem' }}>
                  <RefreshCw size={18} className="spin-animate" style={{ margin: '0 auto .5rem' }} />
                  Loading uploaded MoMs...
                </div>
              ) : filteredMoms.length === 0 ? (
                <div style={{ padding: '3rem 1.5rem', textAlign: 'center', color: 'hsl(var(--pencil))' }}>
                  <FolderOpen size={32} style={{ margin: '0 auto .75rem', opacity: 0.4 }} />
                  <p style={{ margin: 0, fontSize: '.84rem', fontWeight: 600 }}>No MoM files found</p>
                  <p style={{ margin: '4px 0 0', fontSize: '.74rem', opacity: 0.8 }}>
                    {searchQuery ? 'Try another search query' : 'Upload your first meeting notes above'}
                  </p>
                </div>
              ) : (
                filteredMoms.map((mom) => {
                  const isSelected = mom.id === selectedMomId
                  const extLabel = formatExt(mom.filename, mom.file_type)

                  return (
                    <div
                      key={mom.id}
                      onClick={() => setSelectedMomId(mom.id)}
                      style={{
                        padding: '.75rem .85rem',
                        borderRadius: '10px',
                        cursor: 'pointer',
                        border: isSelected
                          ? '1.5px solid hsl(280 80% 55% / .6)'
                          : '1px solid hsl(var(--border) / .2)',
                        background: isSelected
                          ? 'hsl(280 80% 55% / .08)'
                          : 'hsl(var(--card))',
                        transition: 'all .15s',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '.4rem',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '.5rem' }}>
                        <div style={{ fontWeight: 600, fontSize: '.84rem', color: isSelected ? 'hsl(280 80% 45%)' : 'hsl(var(--ink))', minWidth: 0, flex: 1, lineHeight: 1.3 }}>
                          {mom.title}
                        </div>
                        <span
                          style={{
                            fontSize: '.62rem',
                            fontWeight: 700,
                            padding: '1px 5px',
                            borderRadius: '4px',
                            background: 'hsl(var(--muted) / .7)',
                            color: 'hsl(var(--ink) / .8)',
                            flexShrink: 0,
                          }}
                        >
                          {extLabel}
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '.25rem' }}>
                          <Calendar size={11} />
                          {mom.meeting_date || 'No Date'}
                        </span>
                        <span>•</span>
                        <span>{formatFileSize(mom.file_size)}</span>
                      </div>

                      {/* Stat badges */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', flexWrap: 'wrap', marginTop: '2px' }}>
                        <span
                          style={{
                            fontSize: '.62rem',
                            fontWeight: 700,
                            padding: '1px 5px',
                            borderRadius: '4px',
                            background: 'hsl(280 80% 50% / .14)',
                            color: 'hsl(280 80% 45%)',
                          }}
                        >
                          Manual MoM
                        </span>

                        {mom.chunks_count > 0 && (
                          <span
                            style={{
                              fontSize: '.62rem',
                              fontWeight: 600,
                              padding: '1px 5px',
                              borderRadius: '4px',
                              background: 'hsl(205 80% 50% / .1)',
                              color: 'hsl(205 80% 45%)',
                            }}
                          >
                            {mom.chunks_count} chunks
                          </span>
                        )}

                        {mom.action_items_count > 0 && (
                          <span
                            style={{
                              fontSize: '.62rem',
                              fontWeight: 600,
                              padding: '1px 5px',
                              borderRadius: '4px',
                              background: 'hsl(142 70% 45% / .12)',
                              color: 'hsl(142 70% 35%)',
                            }}
                          >
                            {mom.action_items_count} actions
                          </span>
                        )}
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </div>

          {/* ────── Right Pane: Spacious Content & Metadata Preview ────── */}
          <div
            style={{
              flex: 1,
              background: 'hsl(var(--card))',
              display: 'flex',
              flexDirection: 'column',
              minWidth: 0,
              overflow: 'hidden',
            }}
          >
            {loadingDetail ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'hsl(var(--pencil))' }}>
                <RefreshCw size={24} className="spin-animate" style={{ marginBottom: '.75rem' }} />
                <span>Loading extracted MoM content...</span>
              </div>
            ) : !selectedDetail ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'hsl(var(--pencil))', padding: '2rem' }}>
                <FileText size={42} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                <h3 style={{ margin: 0, fontSize: '1.1rem', color: 'hsl(var(--ink))' }}>No MoM Selected</h3>
                <p style={{ margin: '6px 0 0', fontSize: '.84rem' }}>
                  Select an uploaded MoM from the left or upload new files to view extracted content.
                </p>
              </div>
            ) : (
              <>
                {/* ──── Detail Top Header & Action Bar ──── */}
                <div
                  style={{
                    padding: '1.1rem 1.6rem',
                    borderBottom: '1px solid hsl(var(--border) / .3)',
                    background: 'hsl(var(--paper))',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '.85rem',
                    flexShrink: 0,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', flexWrap: 'wrap', marginBottom: '4px' }}>
                        <span
                          style={{
                            fontSize: '.66rem',
                            fontWeight: 700,
                            padding: '2px 7px',
                            borderRadius: '5px',
                            background: 'hsl(280 80% 50% / .18)',
                            color: 'hsl(280 80% 45%)',
                            letterSpacing: '0.02em',
                          }}
                        >
                          Manual MoM
                        </span>
                        <span
                          style={{
                            fontSize: '.66rem',
                            fontWeight: 700,
                            padding: '2px 6px',
                            borderRadius: '5px',
                            background: 'hsl(var(--muted) / .6)',
                            color: 'hsl(var(--pencil))',
                          }}
                        >
                          {formatExt(selectedDetail.filename, selectedDetail.file_type)}
                        </span>
                        <span style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))' }}>
                          Original: {selectedDetail.filename} ({formatFileSize(selectedDetail.file_size)})
                        </span>
                      </div>

                      <h1 style={{ fontSize: '1.45rem', fontWeight: 800, margin: 0, color: 'hsl(var(--ink))', lineHeight: 1.25 }}>
                        {selectedDetail.title}
                      </h1>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '.5rem', flexWrap: 'wrap', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem', fontWeight: 500 }}>
                          <Calendar size={13} style={{ color: 'hsl(var(--accent))' }} />
                          Date: <strong style={{ color: 'hsl(var(--ink))' }}>{selectedDetail.meeting_date || 'N/A'}</strong>
                        </span>

                        <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem', fontWeight: 500 }}>
                          <Layers size={13} style={{ color: 'hsl(280 80% 55%)' }} />
                          Vectors in ChromaDB: <strong style={{ color: 'hsl(var(--ink))' }}>{selectedDetail.chunks?.length || 0} chunks</strong>
                        </span>
                      </div>
                    </div>

                    {/* Quick CTA Actions */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', flexShrink: 0 }}>
                      {/* Primary CTA: Chat with this MoM */}
                      <button
                        onClick={() => {
                          onSelectMeetingForChat(selectedDetail.id)
                          onClose()
                        }}
                        style={{
                          height: '38px',
                          padding: '0 1.15rem',
                          borderRadius: '9px',
                          background: 'linear-gradient(135deg, hsl(280 80% 55%), hsl(220 80% 60%))',
                          color: '#fff',
                          border: 'none',
                          fontWeight: 700,
                          fontSize: '.84rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '.5rem',
                          cursor: 'pointer',
                          boxShadow: '0 4px 12px hsl(280 80% 55% / .3)',
                          transition: 'transform .15s, opacity .15s',
                        }}
                        title="Focus AI Chat on this Manual MoM meeting"
                      >
                        <MessageSquare size={16} />
                        <span>Chat with this MoM</span>
                        <ArrowRight size={14} />
                      </button>

                      {/* Re-index Button */}
                      <button
                        onClick={handleReindex}
                        disabled={actionLoading}
                        className="btn btn-ghost"
                        style={{
                          height: '38px',
                          padding: '0 .85rem',
                          borderRadius: '9px',
                          border: '1px solid hsl(var(--border) / .3)',
                          background: 'hsl(var(--card))',
                          color: 'hsl(var(--ink))',
                          fontWeight: 600,
                          fontSize: '.8rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '.4rem',
                          cursor: actionLoading ? 'not-allowed' : 'pointer',
                        }}
                        title="Re-chunk and update vectors in ChromaDB"
                      >
                        <RotateCcw size={14} className={actionLoading ? 'spin-animate' : ''} />
                        <span>Re-index</span>
                      </button>

                      {/* Delete Button */}
                      {deleteConfirmId === selectedDetail.id ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '.3rem' }}>
                          <button
                            onClick={() => handleDelete(selectedDetail.id)}
                            style={{
                              height: '38px',
                              padding: '0 .75rem',
                              borderRadius: '9px',
                              background: 'hsl(var(--destructive))',
                              color: '#fff',
                              border: 'none',
                              fontSize: '.78rem',
                              fontWeight: 700,
                              cursor: 'pointer',
                            }}
                          >
                            Confirm Delete
                          </button>
                          <button
                            onClick={() => setDeleteConfirmId(null)}
                            style={{
                              height: '38px',
                              padding: '0 .5rem',
                              borderRadius: '9px',
                              background: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border) / .3)',
                              fontSize: '.78rem',
                              cursor: 'pointer',
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirmId(selectedDetail.id)}
                          className="btn btn-ghost"
                          style={{
                            height: '38px',
                            width: '38px',
                            padding: 0,
                            borderRadius: '9px',
                            border: '1px solid hsl(var(--border) / .3)',
                            background: 'hsl(var(--card))',
                            color: 'hsl(var(--destructive))',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                          }}
                          title="Delete this MoM"
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Attendees / Speakers pill bar */}
                  {selectedDetail.speakers && selectedDetail.speakers.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', gap: '.3rem' }}>
                        <Users size={12} />
                        Attendees:
                      </span>
                      {selectedDetail.speakers.map((spk, i) => (
                        <span
                          key={i}
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 500,
                            padding: '2px 8px',
                            borderRadius: '12px',
                            background: 'hsl(var(--paper-deep))',
                            border: '1px solid hsl(var(--border) / .2)',
                            color: 'hsl(var(--ink))',
                          }}
                        >
                          {spk}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Navigation Tabs */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', borderTop: '1px solid hsl(var(--border) / .15)', paddingTop: '.6rem' }}>
                    <button
                      onClick={() => setActiveTab('structured')}
                      style={{
                        padding: '.4rem .9rem',
                        borderRadius: '7px',
                        border: 'none',
                        background: activeTab === 'structured' ? 'hsl(280 80% 55% / .15)' : 'transparent',
                        color: activeTab === 'structured' ? 'hsl(280 80% 45%)' : 'hsl(var(--pencil))',
                        fontWeight: activeTab === 'structured' ? 700 : 500,
                        fontSize: '.82rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.4rem',
                      }}
                    >
                      <Sparkles size={14} />
                      <span>Structured MoM</span>
                    </button>

                    <button
                      onClick={() => setActiveTab('raw')}
                      style={{
                        padding: '.4rem .9rem',
                        borderRadius: '7px',
                        border: 'none',
                        background: activeTab === 'raw' ? 'hsl(280 80% 55% / .15)' : 'transparent',
                        color: activeTab === 'raw' ? 'hsl(280 80% 45%)' : 'hsl(var(--pencil))',
                        fontWeight: activeTab === 'raw' ? 700 : 500,
                        fontSize: '.82rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.4rem',
                      }}
                    >
                      <FileText size={14} />
                      <span>Full Document Text</span>
                    </button>

                    <button
                      onClick={() => setActiveTab('chunks')}
                      style={{
                        padding: '.4rem .9rem',
                        borderRadius: '7px',
                        border: 'none',
                        background: activeTab === 'chunks' ? 'hsl(280 80% 55% / .15)' : 'transparent',
                        color: activeTab === 'chunks' ? 'hsl(280 80% 45%)' : 'hsl(var(--pencil))',
                        fontWeight: activeTab === 'chunks' ? 700 : 500,
                        fontSize: '.82rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.4rem',
                      }}
                    >
                      <Layers size={14} />
                      <span>Retrieval Chunks ({selectedDetail.chunks?.length || 0})</span>
                    </button>
                  </div>
                </div>

                {/* ──── Tab Content Panels ──── */}
                <div style={{ flex: 1, overflowY: 'auto', padding: '1.4rem 1.8rem', minHeight: 0 }}>
                  {/* TAB 1: STRUCTURED MOM */}
                  {activeTab === 'structured' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.4rem', maxWidth: '1000px' }}>
                      {/* Summary Card */}
                      {selectedDetail.summary && (
                        <div
                          style={{
                            padding: '1.1rem 1.3rem',
                            borderRadius: '12px',
                            background: 'hsl(var(--paper))',
                            border: '1.5px solid hsl(var(--border) / .3)',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '.45rem', marginBottom: '.6rem', color: 'hsl(var(--ink))' }}>
                            <Award size={16} style={{ color: 'hsl(var(--accent))' }} />
                            <h3 style={{ fontSize: '.95rem', fontWeight: 700, margin: 0 }}>Executive Summary / Overview</h3>
                          </div>
                          <p style={{ margin: 0, fontSize: '.88rem', lineHeight: 1.6, color: 'hsl(var(--ink) / .9)' }}>
                            {selectedDetail.summary}
                          </p>
                        </div>
                      )}

                      {/* Agendas & Discussion Points */}
                      {selectedDetail.agendas && selectedDetail.agendas.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '.45rem', color: 'hsl(var(--ink))' }}>
                            <ListOrdered size={16} style={{ color: 'hsl(280 80% 55%)' }} />
                            <h3 style={{ fontSize: '.95rem', fontWeight: 700, margin: 0 }}>Agendas & Discussion Points</h3>
                          </div>

                          <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                            {selectedDetail.agendas.map((ag, idx) => (
                              <div
                                key={idx}
                                style={{
                                  padding: '1rem 1.25rem',
                                  borderRadius: '10px',
                                  background: 'hsl(var(--card))',
                                  border: '1px solid hsl(var(--border) / .25)',
                                  boxShadow: '0 2px 6px rgba(0,0,0,0.02)',
                                }}
                              >
                                <h4 style={{ fontSize: '.9rem', fontWeight: 700, margin: '0 0 .5rem', color: 'hsl(var(--ink))' }}>
                                  {ag.title || `Agenda Item ${idx + 1}`}
                                </h4>

                                {ag.discussion_points && ag.discussion_points.length > 0 ? (
                                  <ul style={{ margin: 0, paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
                                    {ag.discussion_points.map((pt, pIdx) => (
                                      <li key={pIdx} style={{ fontSize: '.86rem', lineHeight: 1.5, color: 'hsl(var(--ink) / .85)' }}>
                                        {pt}
                                      </li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p style={{ margin: 0, fontSize: '.86rem', lineHeight: 1.5, color: 'hsl(var(--ink) / .85)' }}>
                                    {ag.content || 'Discussion points noted in meeting.'}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Action Items */}
                      {selectedDetail.action_items && selectedDetail.action_items.length > 0 && (
                        <div
                          style={{
                            padding: '1.1rem 1.3rem',
                            borderRadius: '12px',
                            background: 'hsl(var(--paper))',
                            border: '1.5px solid hsl(142 70% 45% / .25)',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '.45rem', marginBottom: '.75rem', color: 'hsl(var(--ink))' }}>
                            <CheckSquare size={16} style={{ color: 'hsl(142 70% 40%)' }} />
                            <h3 style={{ fontSize: '.95rem', fontWeight: 700, margin: 0 }}>Action Items ({selectedDetail.action_items.length})</h3>
                          </div>

                          <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                            {selectedDetail.action_items.map((act, aIdx) => (
                              <div
                                key={aIdx}
                                style={{
                                  padding: '.6rem .85rem',
                                  borderRadius: '8px',
                                  background: 'hsl(var(--card))',
                                  border: '1px solid hsl(var(--border) / .2)',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'space-between',
                                  gap: '.75rem',
                                }}
                              >
                                <span style={{ fontSize: '.86rem', color: 'hsl(var(--ink))', flex: 1 }}>
                                  {act.task}
                                </span>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', flexShrink: 0 }}>
                                  {act.owner && act.owner !== 'Unassigned' && (
                                    <span
                                      style={{
                                        fontSize: '.72rem',
                                        fontWeight: 600,
                                        padding: '2px 7px',
                                        borderRadius: '4px',
                                        background: 'hsl(205 80% 50% / .12)',
                                        color: 'hsl(205 80% 40%)',
                                      }}
                                    >
                                      @{act.owner}
                                    </span>
                                  )}
                                  {act.due_date && (
                                    <span
                                      style={{
                                        fontSize: '.72rem',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        background: 'hsl(var(--muted) / .7)',
                                        color: 'hsl(var(--pencil))',
                                      }}
                                    >
                                      Due: {act.due_date}
                                    </span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Key Decisions */}
                      {selectedDetail.decisions && selectedDetail.decisions.length > 0 && (
                        <div
                          style={{
                            padding: '1.1rem 1.3rem',
                            borderRadius: '12px',
                            background: 'hsl(var(--paper))',
                            border: '1.5px solid hsl(280 80% 55% / .25)',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '.45rem', marginBottom: '.75rem', color: 'hsl(var(--ink))' }}>
                            <CheckCircle2 size={16} style={{ color: 'hsl(280 80% 55%)' }} />
                            <h3 style={{ fontSize: '.95rem', fontWeight: 700, margin: 0 }}>Key Decisions Made</h3>
                          </div>

                          <ul style={{ margin: 0, paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                            {selectedDetail.decisions.map((dec, dIdx) => (
                              <li key={dIdx} style={{ fontSize: '.86rem', lineHeight: 1.5, color: 'hsl(var(--ink) / .9)' }}>
                                {dec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 2: RAW EXTRACTED DOCUMENT TEXT */}
                  {activeTab === 'raw' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem', height: '100%' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                          Extracted {selectedDetail.raw_text?.length.toLocaleString() || 0} characters from {selectedDetail.filename}
                        </span>
                        <button
                          onClick={handleCopyRaw}
                          style={{
                            padding: '4px 10px',
                            borderRadius: '6px',
                            background: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border) / .3)',
                            fontSize: '.75rem',
                            fontWeight: 600,
                            color: 'hsl(var(--ink))',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.35rem',
                          }}
                        >
                          {copiedRaw ? <Check size={12} style={{ color: 'hsl(142 70% 45%)' }} /> : <Copy size={12} />}
                          <span>{copiedRaw ? 'Copied' : 'Copy Full Text'}</span>
                        </button>
                      </div>

                      <div
                        style={{
                          flex: 1,
                          padding: '1.2rem',
                          borderRadius: '10px',
                          background: 'hsl(var(--paper))',
                          border: '1px solid hsl(var(--border) / .3)',
                          fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                          fontSize: '.86rem',
                          lineHeight: 1.6,
                          color: 'hsl(var(--ink))',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          overflowY: 'auto',
                        }}
                      >
                        {selectedDetail.raw_text || '(Empty text)'}
                      </div>
                    </div>
                  )}

                  {/* TAB 3: RETRIEVAL CHUNKS IN CHROMADB */}
                  {activeTab === 'chunks' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem', maxWidth: '1000px' }}>
                      <div
                        style={{
                          padding: '.75rem 1rem',
                          borderRadius: '8px',
                          background: 'hsl(205 80% 50% / .08)',
                          border: '1px solid hsl(205 80% 50% / .2)',
                          fontSize: '.8rem',
                          color: 'hsl(205 80% 35%)',
                        }}
                      >
                        These exact {selectedDetail.chunks?.length || 0} chunks are indexed into ChromaDB (collection <code>final_rom</code>). When you ask AI Chat questions, the retriever prioritizes these chunks to cite and answer accurately.
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                        {selectedDetail.chunks?.map((chunk, cIdx) => (
                          <div
                            key={cIdx}
                            style={{
                              padding: '.9rem 1.15rem',
                              borderRadius: '10px',
                              background: 'hsl(var(--paper))',
                              border: '1px solid hsl(var(--border) / .25)',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: '.45rem',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '.5rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                                <span
                                  style={{
                                    fontSize: '.7rem',
                                    fontWeight: 700,
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    background: 'hsl(280 80% 55% / .18)',
                                    color: 'hsl(280 80% 45%)',
                                  }}
                                >
                                  Chunk #{chunk.chunk_index}
                                </span>
                                {chunk.agenda_title && (
                                  <span style={{ fontSize: '.76rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                                    {chunk.agenda_title}
                                  </span>
                                )}
                              </div>
                              <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                                {chunk.text.length} chars
                              </span>
                            </div>

                            <p
                              style={{
                                margin: 0,
                                fontSize: '.84rem',
                                lineHeight: 1.55,
                                color: 'hsl(var(--ink) / .88)',
                                whiteSpace: 'pre-wrap',
                                fontFamily: 'Inter, sans-serif',
                              }}
                            >
                              {chunk.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
