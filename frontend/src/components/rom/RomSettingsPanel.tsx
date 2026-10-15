/**
 * ROM settings chosen while a meeting is captured (Upload / Record / Tab Recording).
 * Sent with the recording as `rom_settings`; when transcription finishes the backend runs the
 * ROM pipeline by itself (backend/tasks/rom_auto_pipeline.py).
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import {
  Brain, CheckSquare, ChevronDown, ChevronUp, FileText, ListChecks, Loader,
  Square, Target, Upload, Workflow, X, Sparkles, FileSpreadsheet, FileCode
} from 'lucide-react'
import { getApiErrorDetail } from '../../lib/errors'
import {
  DEFAULT_ROM_SETTINGS, appendRomSettings, deleteStagedFile, describeAgendaExtraction, extractAgendaDocument,
  getRomDefaults, listPreviousMeetings, reextractAgendaItems, stageContextFile,
  type RomOutputType, type RomPipelineSettings,
} from '../../api/romPipeline'
import { AgendaItemsEditor, moveItem, type AgendaItem } from './AgendaItemsEditor'
import { Stage1Options, Stage2Options, type PreviousMeetingOption, type Stage2OptionsValue } from './RomStageOptions'

const DOC_ACCEPT = '.pdf,.docx,.doc,.pptx,.ppt,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv'
const CONTEXT_ACCEPT = '.pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv'

/** State of the panel for one capture page. */
export function useRomPipelineSettings() {
  const [settings, setSettings] = useState<RomPipelineSettings>(DEFAULT_ROM_SETTINGS)
  const [saveAsDefault, setSaveAsDefault] = useState(false)
  const touched = useRef(false)
  const latest = useRef({ settings, saveAsDefault })
  latest.current = { settings, saveAsDefault }

  useEffect(() => {
    getRomDefaults()
      .then(({ defaults }) => {
        if (!touched.current) setSettings(prev => ({ ...prev, ...defaults }))
      })
      .catch(() => {})
  }, [])

  const update = useCallback((patch: Partial<RomPipelineSettings>) => {
    touched.current = true
    setSettings(prev => ({ ...prev, ...patch }))
  }, [])

  /** Add the settings to the capture form (call in the page's existing submit). */
  const appendTo = useCallback(
    (form: FormData) => appendRomSettings(form, latest.current.settings, latest.current.saveAsDefault),
    [],
  )

  /** After a capture is submitted: keep the options, drop this meeting's files and agenda. */
  const resetMeetingInputs = useCallback(() => {
    setSettings(prev => ({ ...prev, context_files: [], agenda_doc: null, agendas: null, agenda_text: null }))
  }, [])

  return { settings, update, saveAsDefault, setSaveAsDefault, appendTo, resetMeetingInputs }
}

export type RomPipelineSettingsState = ReturnType<typeof useRomPipelineSettings>

const OUTPUT_OPTIONS: { value: RomOutputType; label: string; sublabel: string; desc: string; icon: typeof Workflow }[] = [
  {
    value: 'off',
    label: 'Off',
    sublabel: 'Transcript only',
    desc: 'Runs transcription only. You can run the ROM pipeline later anytime from the meeting history.',
    icon: FileText,
  },
  {
    value: 'mom',
    label: 'MoM',
    sublabel: 'Minutes of Meeting',
    desc: 'Auto-executes Stage 1 Chunking → Stage 2 Vector Analysis → Structured Minutes of Meeting.',
    icon: ListChecks,
  },
  {
    value: 'rom',
    label: 'ROM',
    sublabel: 'Full Record of Meeting',
    desc: 'Full 3-stage pipeline: Stage 1 → Stage 2 → Stage 3 Agenda Matching → Final Comprehensive ROM Document.',
    icon: Workflow,
  },
]

const sectionTitle = (icon: ReactNode, label: string, color: string, badge?: string) => (
  <div style={{
    fontSize: '.75rem',
    fontWeight: 700,
    color: 'hsl(var(--ink))',
    letterSpacing: '.03em',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '.55rem',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{
        color,
        display: 'flex',
        padding: '3px',
        borderRadius: 5,
        background: `${color}18`,
      }}>
        {icon}
      </span>
      <span>{label}</span>
    </div>
    {badge && (
      <span style={{
        fontSize: '.68rem',
        fontWeight: 600,
        color,
        background: `${color}15`,
        border: `1px solid ${color}30`,
        padding: '1px 7px',
        borderRadius: 999,
        fontFamily: 'JetBrains Mono, monospace',
      }}>
        {badge}
      </span>
    )}
  </div>
)

const cardStyle = {
  borderRadius: 10,
  border: '1px solid hsl(var(--border) / .35)',
  background: 'hsl(var(--card))',
  padding: '.85rem 1rem',
  display: 'flex',
  flexDirection: 'column' as const,
  gap: '.55rem',
  boxShadow: '0 1px 4px rgba(0, 0, 0, 0.02)',
}

const nextAgendaId = (agendas: AgendaItem[]) => {
  const used = new Set(agendas.map(a => a.agenda_id))
  let n = 1
  while (used.has(`A${n}`)) n++
  return `A${n}`
}

export default function RomSettingsPanel({ state, defaultOpen = false }: {
  state: RomPipelineSettingsState
  defaultOpen?: boolean
}) {
  const { settings, update, saveAsDefault, setSaveAsDefault } = state
  const [open, setOpen] = useState(defaultOpen)
  const [uploadingContext, setUploadingContext] = useState(false)
  const [extractingAgenda, setExtractingAgenda] = useState(false)
  const [previousMeetings, setPreviousMeetings] = useState<PreviousMeetingOption[]>([])
  const [loadingPrevious, setLoadingPrevious] = useState(false)
  const contextInput = useRef<HTMLInputElement>(null)
  const agendaInput = useRef<HTMLInputElement>(null)

  const isRom = settings.output_type === 'rom'
  const agendas = settings.agendas || []

  useEffect(() => {
    if (settings.previous_meeting_mode !== 'select' || previousMeetings.length > 0) return
    setLoadingPrevious(true)
    listPreviousMeetings()
      .then(meetings => {
        setPreviousMeetings(meetings)
        if (!settings.previous_meeting_id) {
          const first = meetings.find(m => m.has_stage2) || meetings[0]
          if (first) update({ previous_meeting_id: first.id })
        }
      })
      .catch(() => {})
      .finally(() => setLoadingPrevious(false))
  }, [settings.previous_meeting_mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const stage2Value: Stage2OptionsValue = {
    meetingTopK: settings.meeting_top_k,
    globalTopK: settings.global_top_k,
    discussionWindowSize: settings.discussion_window_size,
    minSimilarityThreshold: settings.min_similarity_threshold,
    processAllTogether: !!settings.process_all_together,
    previousMeetingMode: settings.previous_meeting_mode,
    previousMeetingId: settings.previous_meeting_id || '',
    previousMeetingTopK: settings.previous_meeting_top_k,
  }

  const onStage2Change = (patch: Partial<Stage2OptionsValue>) => {
    const out: Partial<RomPipelineSettings> = {}
    if (patch.meetingTopK !== undefined) out.meeting_top_k = patch.meetingTopK
    if (patch.globalTopK !== undefined) out.global_top_k = patch.globalTopK
    if (patch.discussionWindowSize !== undefined) out.discussion_window_size = patch.discussionWindowSize
    if (patch.minSimilarityThreshold !== undefined) out.min_similarity_threshold = patch.minSimilarityThreshold
    if (patch.processAllTogether !== undefined) out.process_all_together = patch.processAllTogether
    if (patch.previousMeetingMode !== undefined) out.previous_meeting_mode = patch.previousMeetingMode
    if (patch.previousMeetingId !== undefined) out.previous_meeting_id = patch.previousMeetingId || null
    if (patch.previousMeetingTopK !== undefined) out.previous_meeting_top_k = patch.previousMeetingTopK
    update(out)
  }

  const addContextFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setUploadingContext(true)
    const added = []
    for (const file of Array.from(files)) {
      try {
        added.push(await stageContextFile(file))
      } catch (err) {
        toast.error(getApiErrorDetail(err) || `Failed to add ${file.name}`)
      }
    }
    if (added.length) update({ context_files: [...settings.context_files, ...added] })
    setUploadingContext(false)
    if (contextInput.current) contextInput.current.value = ''
  }

  const removeContextFile = (stagedId: string) => {
    update({ context_files: settings.context_files.filter(f => f.staged_id !== stagedId) })
    deleteStagedFile(stagedId).catch(() => {})
  }

  const uploadAgendaDocument = async (file: File | undefined) => {
    if (!file) return
    setExtractingAgenda(true)
    try {
      const res = await extractAgendaDocument(file, settings.agenda_from_tables)
      if (settings.agenda_doc) deleteStagedFile(settings.agenda_doc.staged_id).catch(() => {})
      update({
        agenda_doc: { staged_id: res.staged_id, filename: res.filename },
        agenda_text: res.agenda_text,
        agendas: res.agendas,
      })
      toast.success(`${res.filename}: ${describeAgendaExtraction(res, settings.agenda_from_tables)}`)
    } catch (err) {
      toast.error(getApiErrorDetail(err) || `Failed to extract the agenda from ${file.name}`)
    } finally {
      setExtractingAgenda(false)
      if (agendaInput.current) agendaInput.current.value = ''
    }
  }

  // With a document already attached, read its agenda items again under the new option.
  const changeAgendaFromTables = async (fromTables: boolean) => {
    update({ agenda_from_tables: fromTables })
    const doc = settings.agenda_doc
    if (!doc) return
    setExtractingAgenda(true)
    try {
      const res = await reextractAgendaItems(doc.staged_id, fromTables)
      update({ agendas: res.agendas })
      toast.success(`${doc.filename}: ${describeAgendaExtraction(res, fromTables)}`)
    } catch (err) {
      toast.error(getApiErrorDetail(err) || `Failed to extract the agenda from ${doc.filename} again`)
    } finally {
      setExtractingAgenda(false)
    }
  }

  const removeAgendaDocument = () => {
    if (settings.agenda_doc) deleteStagedFile(settings.agenda_doc.staged_id).catch(() => {})
    update({ agenda_doc: null, agenda_text: null })
  }

  const setAgendas = (next: AgendaItem[]) => update({ agendas: next })
  const activeOption = OUTPUT_OPTIONS.find(o => o.value === settings.output_type)

  return (
    <div
      className="animate-slide-up"
      style={{
        borderRadius: '12px',
        background: 'hsl(var(--card))',
        border: '1px solid hsl(var(--border) / .35)',
        boxShadow: '0 4px 18px -4px rgba(0, 0, 0, 0.05)',
        overflow: 'hidden',
        marginTop: '1rem',
        transition: 'border-color .2s ease, box-shadow .2s ease',
      }}
    >
      {/* ── Accordion Toggle Header ── */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        id="rom-settings-toggle"
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '1rem 1.25rem',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          color: 'hsl(var(--ink))',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 600,
          fontSize: '.92rem',
          textAlign: 'left',
          transition: 'background .15s ease',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--muted) / .35)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
      >
        <div style={{
          width: 32,
          height: 32,
          borderRadius: '9px',
          background: settings.output_type === 'off'
            ? 'hsl(var(--muted) / .7)'
            : 'linear-gradient(135deg, hsl(275, 80%, 60% / .18), hsl(285, 85%, 55% / .08))',
          border: `1.5px solid ${settings.output_type === 'off' ? 'hsl(var(--border) / .4)' : 'hsl(275, 80%, 60% / .3)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Workflow size={16} style={{ color: settings.output_type === 'off' ? 'hsl(var(--pencil))' : 'hsl(275, 80%, 60%)' }} />
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 700, fontSize: '.92rem', color: 'hsl(var(--ink))' }}>
              ROM Pipeline Settings
            </span>
          </div>
          <p style={{
            fontSize: '.75rem',
            color: 'hsl(var(--pencil))',
            margin: 0,
            fontWeight: 400,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            Automated Minutes of Meeting (MoM) & Record of Meeting (ROM) generation
          </p>
        </div>

        {/* Status Badge */}
        <span style={{
          fontSize: '.72rem',
          fontWeight: 700,
          padding: '.25rem .65rem',
          borderRadius: '999px',
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          color: settings.output_type === 'off'
            ? 'hsl(var(--pencil))'
            : settings.output_type === 'mom'
              ? 'hsl(215, 85%, 55%)'
              : 'hsl(275, 80%, 60%)',
          background: settings.output_type === 'off'
            ? 'hsl(var(--muted) / .6)'
            : settings.output_type === 'mom'
              ? 'hsl(215, 85%, 60% / .12)'
              : 'hsl(275, 80%, 60% / .12)',
          border: `1px solid ${settings.output_type === 'off'
            ? 'hsl(var(--border) / .3)'
            : settings.output_type === 'mom'
              ? 'hsl(215, 85%, 60% / .3)'
              : 'hsl(275, 80%, 60% / .3)'}`,
        }}>
          {settings.output_type !== 'off' && (
            <span style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: settings.output_type === 'mom' ? 'hsl(215, 85%, 55%)' : 'hsl(275, 80%, 60%)',
              boxShadow: `0 0 6px ${settings.output_type === 'mom' ? 'hsl(215, 85%, 55% / .6)' : 'hsl(275, 80%, 60% / .6)'}`,
            }} />
          )}
          {settings.output_type === 'off' ? 'Auto-run Off' : `Auto: ${activeOption?.label}`}
        </span>

        <div style={{ color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center' }}>
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* ── Accordion Body ── */}
      {open && (
        <div style={{
          padding: '1rem 1.25rem 1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          borderTop: '1px solid hsl(var(--border) / .35)',
        }}>
          {/* 1. Output Type Segmented Switch */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(var(--ink))', textTransform: 'uppercase', letterSpacing: '.03em' }}>
                Pipeline Output Mode
              </span>
              <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                Executes after transcription completes
              </span>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 4,
              background: 'hsl(var(--muted) / .5)',
              padding: 4,
              borderRadius: 9,
              border: '1px solid hsl(var(--border) / .3)',
            }}>
              {OUTPUT_OPTIONS.map(o => {
                const active = settings.output_type === o.value
                const IconComponent = o.icon
                return (
                  <button
                    key={o.value}
                    type="button"
                    onClick={() => update({ output_type: o.value })}
                    style={{
                      padding: '.6rem .5rem',
                      border: 'none',
                      borderRadius: 7,
                      cursor: 'pointer',
                      background: active
                        ? (o.value === 'off'
                          ? 'hsl(var(--card))'
                          : o.value === 'mom'
                            ? 'hsl(215, 85%, 60%)'
                            : 'hsl(275, 80%, 60%)')
                        : 'transparent',
                      color: active
                        ? (o.value === 'off' ? 'hsl(var(--ink))' : '#ffffff')
                        : 'hsl(var(--pencil))',
                      boxShadow: active ? '0 2px 6px rgba(0, 0, 0, 0.12)' : 'none',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: 2,
                      transition: 'all .18s cubic-bezier(0.4, 0, 0.2, 1)',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <IconComponent size={14} />
                      <span style={{ fontSize: '.84rem', fontWeight: 700 }}>{o.label}</span>
                    </div>
                    <span style={{
                      fontSize: '.68rem',
                      opacity: active ? (o.value === 'off' ? 0.75 : 0.9) : 0.7,
                      fontWeight: 500,
                    }}>
                      {o.sublabel}
                    </span>
                  </button>
                )
              })}
            </div>

            <p style={{
              fontSize: '.75rem',
              color: 'hsl(var(--pencil))',
              margin: '2px 0 0',
              lineHeight: 1.45,
            }}>
              {activeOption?.desc}
            </p>
          </div>

          {/* When pipeline is NOT off, show options */}
          {settings.output_type !== 'off' && (
            <>
              {/* 2. Stage 1 & Stage 2 Cards */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '.85rem',
              }}>
                {/* Stage 1 */}
                <div style={cardStyle}>
                  {sectionTitle(<Brain size={14} />, 'Stage 1: Chunking', 'hsl(280, 75%, 60%)', 'Audio Window')}
                  <Stage1Options
                    transcriptWindow={settings.transcript_window_minutes}
                    onChange={m => update({ transcript_window_minutes: m })}
                  />
                </div>

                {/* Stage 2 */}
                <div style={cardStyle}>
                  {sectionTitle(<Target size={14} />, 'Stage 2: Vector Context', 'hsl(205, 90%, 55%)', 'ChromaDB')}
                  <Stage2Options
                    value={stage2Value}
                    onChange={onStage2Change}
                    previousMeetings={previousMeetings}
                    loadingPreviousMeetings={loadingPrevious}
                  />
                </div>
              </div>

              {/* 3. Meeting Context Documents */}
              <div style={cardStyle}>
                {sectionTitle(
                  <FileText size={14} />,
                  'Meeting Context Documents',
                  'hsl(205, 90%, 55%)',
                  settings.context_files.length ? `${settings.context_files.length} attached` : undefined
                )}

                <input
                  ref={contextInput}
                  type="file"
                  multiple
                  accept={CONTEXT_ACCEPT}
                  style={{ display: 'none' }}
                  onChange={e => addContextFiles(e.target.files)}
                />

                {settings.context_files.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
                    {settings.context_files.map(f => (
                      <div
                        key={f.staged_id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          padding: '4px 8px',
                          borderRadius: 6,
                          background: 'hsl(205, 90%, 55% / .08)',
                          border: '1px solid hsl(205, 90%, 55% / .25)',
                          fontSize: '.75rem',
                          color: 'hsl(var(--ink))',
                        }}
                      >
                        <FileText size={12} style={{ flexShrink: 0, color: 'hsl(205, 90%, 55%)' }} />
                        <span style={{
                          maxWidth: 200,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontWeight: 500,
                        }}>
                          {f.filename}
                        </span>
                        <button
                          type="button"
                          onClick={() => removeContextFile(f.staged_id)}
                          title="Remove file"
                          style={{
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            color: 'hsl(var(--destructive))',
                            padding: 2,
                            display: 'flex',
                            alignItems: 'center',
                          }}
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <div
                  onClick={() => !uploadingContext && contextInput.current?.click()}
                  style={{
                    border: '1.5px dashed hsl(205, 90%, 55% / .4)',
                    borderRadius: 8,
                    padding: '.65rem 1rem',
                    textAlign: 'center',
                    cursor: uploadingContext ? 'wait' : 'pointer',
                    fontSize: '.78rem',
                    fontWeight: 600,
                    color: 'hsl(205, 90%, 50%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    background: 'hsl(205, 90%, 55% / .04)',
                    transition: 'all .15s ease',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'hsl(205, 90%, 55% / .09)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'hsl(205, 90%, 55% / .04)')}
                >
                  {uploadingContext ? <Loader size={13} className="spin" /> : <Upload size={13} />}
                  <span>{uploadingContext ? 'Uploading context documents…' : 'Attach Supporting Context (PDF, DOCX, PPTX, TXT, Images, CSV)'}</span>
                </div>
              </div>

              {/* 4. Stage 3 Agenda (ROM Only) */}
              {isRom && (
                <div style={cardStyle}>
                  {sectionTitle(<ListChecks size={14} />, 'Stage 3: Meeting Agenda', 'hsl(140, 70%, 45%)', `${agendas.length} items`)}

                  <input
                    ref={agendaInput}
                    type="file"
                    accept={DOC_ACCEPT}
                    style={{ display: 'none' }}
                    onChange={e => uploadAgendaDocument(e.target.files?.[0])}
                  />

                  {/* Extract Agenda from Tables Toggle */}
                  <label
                    htmlFor="rom-agenda-from-tables"
                    title="On: agenda items are read from the document's tables only, ignoring headings and other text (the whole document is scanned when it has no table). Off: lists, headings and other formats anywhere in the document are scanned."
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '.45rem .75rem',
                      borderRadius: 8,
                      background: 'hsl(var(--muted) / .25)',
                      border: '1px solid hsl(var(--border) / .35)',
                      cursor: extractingAgenda ? 'wait' : 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <FileSpreadsheet size={13} style={{ color: 'hsl(140, 70%, 45%)' }} />
                      <div>
                        <div style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                          Extract Agenda from Tables
                        </div>
                        <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>
                          {settings.agenda_from_tables
                            ? 'Reads agenda items from tables only, ignoring headings and other text'
                            : 'Scans the whole document for lists, headings and other formats'}
                        </div>
                      </div>
                    </div>
                    <input
                      id="rom-agenda-from-tables"
                      type="checkbox"
                      checked={settings.agenda_from_tables}
                      disabled={extractingAgenda}
                      onChange={e => changeAgendaFromTables(e.target.checked)}
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 4,
                        accentColor: 'hsl(140, 70%, 45%)',
                        cursor: extractingAgenda ? 'wait' : 'pointer',
                      }}
                    />
                  </label>

                  {settings.agenda_doc ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '.55rem .85rem',
                      borderRadius: 8,
                      background: 'hsl(140, 70%, 45% / .1)',
                      border: '1px solid hsl(140, 70%, 45% / .3)',
                      fontSize: '.78rem',
                    }}>
                      <FileText size={14} style={{ flexShrink: 0, color: 'hsl(140, 70%, 45%)' }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {settings.agenda_doc.filename}
                        </div>
                        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                          Extracted agenda document
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => agendaInput.current?.click()}
                        disabled={extractingAgenda}
                        style={{
                          background: 'hsl(140, 70%, 45% / .15)',
                          border: '1px solid hsl(140, 70%, 45% / .3)',
                          borderRadius: 5,
                          padding: '3px 8px',
                          cursor: 'pointer',
                          color: 'hsl(140, 70%, 35%)',
                          fontSize: '.72rem',
                          fontWeight: 700,
                        }}
                      >
                        {extractingAgenda ? 'Extracting…' : 'Replace File'}
                      </button>
                      <button
                        type="button"
                        onClick={removeAgendaDocument}
                        title="Remove agenda document"
                        style={{
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          color: 'hsl(var(--destructive))',
                          display: 'flex',
                          padding: 2,
                        }}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <div
                      onClick={() => !extractingAgenda && agendaInput.current?.click()}
                      style={{
                        border: '1.5px dashed hsl(140, 70%, 45% / .4)',
                        borderRadius: 8,
                        padding: '.65rem 1rem',
                        textAlign: 'center',
                        cursor: extractingAgenda ? 'wait' : 'pointer',
                        fontSize: '.78rem',
                        fontWeight: 600,
                        color: 'hsl(140, 70%, 40%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 6,
                        background: 'hsl(140, 70%, 45% / .04)',
                        transition: 'all .15s ease',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'hsl(140, 70%, 45% / .09)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'hsl(140, 70%, 45% / .04)')}
                    >
                      {extractingAgenda ? <Loader size={13} className="spin" /> : <Upload size={13} />}
                      <span>{extractingAgenda ? 'Extracting agenda from document…' : 'Upload Agenda Slides or Document (PPT / PDF / DOCX)'}</span>
                    </div>
                  )}

                  <p style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.45 }}>
                    {agendas.length === 0
                      ? 'No agenda items yet: upload a document or add items manually. If empty, agendas will be synthesized automatically from discussion.'
                      : 'The ROM pipeline precisely aligns the meeting discussion against each agenda item below.'}
                  </p>

                  <AgendaItemsEditor
                    compact
                    agendas={agendas}
                    onAdd={item => {
                      setAgendas([...agendas, {
                        agenda_id: nextAgendaId(agendas),
                        title: item.title,
                        description: item.description,
                        presenter: item.presenter || null,
                      }])
                      return true
                    }}
                    onUpdate={(index, patch) => setAgendas(agendas.map((a, i) => (i === index ? { ...a, ...patch } : a)))}
                    onDelete={(_id, _title, index) => setAgendas(agendas.filter((_, i) => i !== index))}
                    onMove={(from, to) => setAgendas(moveItem(agendas, from, to))}
                  />

                  {/* Include Action Points Toggle */}
                  <label
                    htmlFor="rom-include-action-points"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '.45rem .75rem',
                      borderRadius: 8,
                      background: 'hsl(var(--muted) / .25)',
                      border: '1px solid hsl(var(--border) / .35)',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <ListChecks size={13} style={{ color: 'hsl(140, 70%, 45%)' }} />
                      <div>
                        <div style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                          Include Action Points
                        </div>
                        <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>
                          Generate validated action points for every agenda in the Final ROM
                        </div>
                      </div>
                    </div>
                    <input
                      id="rom-include-action-points"
                      type="checkbox"
                      checked={settings.include_action_points}
                      onChange={e => update({ include_action_points: e.target.checked })}
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 4,
                        accentColor: 'hsl(140, 70%, 45%)',
                        cursor: 'pointer',
                      }}
                    />
                  </label>
                </div>
              )}
            </>
          )}

          {/* 5. Save as default */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            paddingTop: '.4rem',
            borderTop: '1px solid hsl(var(--border) / .25)',
          }}>
            <button
              type="button"
              onClick={() => setSaveAsDefault(!saveAsDefault)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: 0,
                fontSize: '.82rem',
                color: 'hsl(var(--ink))',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
              }}
            >
              {saveAsDefault ? (
                <CheckSquare size={16} style={{ color: 'hsl(275, 80%, 60%)' }} />
              ) : (
                <Square size={16} style={{ color: 'hsl(var(--pencil))' }} />
              )}
              <span>Save settings as default</span>
            </button>
            <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
              Persists output mode and Stage 1/2 parameters
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
