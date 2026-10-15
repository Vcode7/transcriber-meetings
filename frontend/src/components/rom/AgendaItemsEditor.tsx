/**
 * Editable agenda list (add, edit, delete, reorder) of ROM Stage 3.
 * Used by the ROM page's Agenda tab (persisting through the Stage 3 agenda endpoints) and by the
 * ROM settings panel on the capture pages (local state, sent with the recording).
 */
import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronUp, List, Loader, Pencil, Plus, Trash2, User, X, Check } from 'lucide-react'

export interface AgendaItem {
  agenda_id?: string
  title: string
  description?: string
  presenter?: string | null
  speaker?: string | null
}

export interface NewAgendaInput {
  title: string
  description: string
  presenter: string
}

interface Props {
  agendas: AgendaItem[]
  /** Return true when the agenda was added (the form is then cleared and closed). */
  onAdd: (item: NewAgendaInput) => Promise<boolean> | boolean
  onUpdate: (index: number, patch: { title: string; description: string }) => Promise<void> | void
  onDelete: (agendaId: string, title: string, index: number) => void
  onMove?: (from: number, to: number) => void
  adding?: boolean
  saving?: boolean
  label?: string
  headerActions?: ReactNode
  /** Rendered between the header/add form and the list (e.g. analysis, timeline controls). */
  beforeList?: ReactNode
  presenterOf?: (agenda: AgendaItem, agendaId: string) => string | null | undefined
  renderIdExtra?: (agenda: AgendaItem, agendaId: string) => ReactNode
  renderMetaExtra?: (agenda: AgendaItem, agendaId: string) => ReactNode
  /** Rendered under each agenda's header (e.g. skip toggle, supporting document). */
  renderExtras?: (agenda: AgendaItem, index: number, agendaId: string) => ReactNode
  compact?: boolean
}

const GREEN = 'hsl(140, 70%, 45%)'

const inputStyle = {
  width: '100%',
  padding: '7px 10px',
  borderRadius: 7,
  border: '1px solid hsl(var(--border) / .4)',
  fontSize: '.82rem',
  fontFamily: 'Inter, sans-serif',
  color: 'hsl(var(--ink))',
  background: 'hsl(var(--card))',
  outline: 'none',
  transition: 'border-color .15s, box-shadow .15s',
}

const iconBtnStyle = {
  background: 'transparent',
  border: 'none',
  cursor: 'pointer',
  color: 'hsl(var(--pencil))',
  padding: '4px',
  borderRadius: '5px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  transition: 'background .15s, color .15s',
}

export const agendaIdOf = (agenda: AgendaItem, index: number) => agenda.agenda_id || `A${index + 1}`

export function AgendaItemsEditor({
  agendas, onAdd, onUpdate, onDelete, onMove, adding = false, saving = false, label = 'Agenda Items',
  headerActions, beforeList, presenterOf, renderIdExtra, renderMetaExtra, renderExtras, compact = false,
}: Props) {
  const [showAddForm, setShowAddForm] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newPresenter, setNewPresenter] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const [editDescription, setEditDescription] = useState('')

  const submitAdd = async () => {
    if (!newTitle.trim()) return
    const ok = await onAdd({ title: newTitle.trim(), description: newDesc.trim(), presenter: newPresenter.trim() })
    if (ok) {
      setShowAddForm(false)
      setNewTitle('')
      setNewDesc('')
      setNewPresenter('')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: compact ? '.65rem' : '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '.35rem .65rem',
          borderRadius: 8,
          background: `${GREEN}15`,
          border: `1px solid ${GREEN}30`,
        }}>
          <span style={{ color: GREEN, display: 'flex' }}><List size={14} /></span>
          <span style={{ fontSize: '.74rem', fontWeight: 700, color: GREEN, textTransform: 'uppercase', letterSpacing: '.04em' }}>
            {label}
          </span>
          <span style={{
            fontSize: '.68rem', fontWeight: 700,
            padding: '1px 6px', borderRadius: 999,
            background: `${GREEN}25`,
            border: `1px solid ${GREEN}45`,
            color: GREEN,
            fontFamily: 'JetBrains Mono, monospace',
          }}>
            {agendas.length}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 6 }}>
          <button
            type="button"
            onClick={() => setShowAddForm(prev => !prev)}
            style={{
              padding: '.35rem .75rem',
              borderRadius: 7,
              border: `1.5px solid ${showAddForm ? 'hsl(var(--destructive) / .3)' : 'hsl(140, 70%, 45% / .5)'}`,
              background: showAddForm ? 'hsl(var(--destructive) / .08)' : 'hsl(140, 70%, 45% / .1)',
              color: showAddForm ? 'hsl(var(--destructive))' : 'hsl(140, 70%, 40%)',
              fontSize: '.75rem',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              cursor: 'pointer',
              fontFamily: 'Inter, sans-serif',
              transition: 'all .15s ease',
            }}
          >
            {showAddForm ? <X size={12} /> : <Plus size={12} />}
            {showAddForm ? 'Cancel' : 'Add Agenda'}
          </button>
          {headerActions}
        </div>
      </div>

      {/* ── Add New Agenda Form (Collapsible) ── */}
      {showAddForm && (
        <div style={{
          borderRadius: 10,
          border: '1.5px solid hsl(140, 70%, 45% / .4)',
          background: 'linear-gradient(180deg, hsl(140, 70%, 45% / .06), hsl(var(--card)))',
          padding: '1rem 1.15rem',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          boxShadow: '0 4px 14px rgba(0, 0, 0, 0.05)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(140, 70%, 40%)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Plus size={14} /> New Agenda Item
            </div>
            <button
              type="button"
              onClick={() => setShowAddForm(false)}
              style={iconBtnStyle}
              title="Close"
            >
              <X size={14} />
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div>
              <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                Agenda Title <span style={{ color: 'hsl(var(--destructive))' }}>*</span>
              </label>
              <input
                value={newTitle}
                onChange={e => setNewTitle(e.target.value)}
                placeholder="e.g. Budget Review & Next Quarter Planning"
                style={inputStyle}
                autoFocus
              />
            </div>
            <div>
              <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                Description / Discussion Details
              </label>
              <textarea
                value={newDesc}
                onChange={e => setNewDesc(e.target.value)}
                rows={2}
                placeholder="Optional overview of topics, context, or expected discussions..."
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </div>
            <div>
              <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                Presenter / Lead (Optional)
              </label>
              <input
                value={newPresenter}
                onChange={e => setNewPresenter(e.target.value)}
                placeholder="e.g. Alice or Team Lead"
                style={inputStyle}
              />
            </div>

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4 }}>
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                style={{
                  padding: '.4rem .85rem',
                  borderRadius: 6,
                  border: '1px solid hsl(var(--border) / .4)',
                  background: 'transparent',
                  color: 'hsl(var(--ink))',
                  fontSize: '.76rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontFamily: 'Inter, sans-serif',
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submitAdd}
                disabled={adding || !newTitle.trim()}
                style={{
                  padding: '.4rem 1.1rem',
                  borderRadius: 6,
                  border: 'none',
                  background: 'hsl(140, 70%, 45%)',
                  color: '#ffffff',
                  fontSize: '.76rem',
                  fontWeight: 700,
                  cursor: adding || !newTitle.trim() ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontFamily: 'Inter, sans-serif',
                  opacity: adding || !newTitle.trim() ? 0.6 : 1,
                  boxShadow: '0 2px 6px rgba(34, 197, 94, 0.25)',
                }}
              >
                {adding ? <Loader size={12} className="spin" /> : <Plus size={12} />}
                {adding ? 'Adding…' : 'Add Agenda Item'}
              </button>
            </div>
          </div>
        </div>
      )}

      {beforeList}

      {/* Agenda Items List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: compact ? '.5rem' : '.75rem' }}>
        {agendas.length === 0 && (
          <div style={{
            padding: '.85rem 1rem',
            borderRadius: 8,
            border: '1px dashed hsl(var(--border) / .5)',
            background: 'hsl(var(--muted) / .15)',
            textAlign: 'center',
            fontSize: '.76rem',
            color: 'hsl(var(--pencil))',
          }}>
            No agenda items defined yet. Add items above or upload an agenda document.
          </div>
        )}

        {agendas.map((agenda, aIdx) => {
          const agendaId = agendaIdOf(agenda, aIdx)
          const isEditing = editingId === agendaId
          const presenterName = presenterOf ? presenterOf(agenda, agendaId) : (agenda.presenter || agenda.speaker)
          const metaExtra = renderMetaExtra?.(agenda, agendaId)

          return (
            <div
              key={agendaId}
              style={{
                borderRadius: 9,
                border: '1px solid hsl(var(--border) / .35)',
                background: 'hsl(var(--card))',
                overflow: 'hidden',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
                transition: 'border-color .15s, box-shadow .15s',
              }}
            >
              {/* Agenda Item Header */}
              <div style={{
                background: 'linear-gradient(90deg, hsl(140, 70%, 50% / .06), transparent)',
                padding: compact ? '.55rem .75rem' : '.75rem 1rem',
                borderBottom: renderExtras ? '1px solid hsl(var(--border) / .3)' : 'none',
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, flexShrink: 0, marginTop: 1 }}>
                  <span style={{
                    background: 'hsl(140, 70%, 45%)',
                    color: '#ffffff',
                    fontWeight: 700,
                    fontSize: '.72rem',
                    padding: '2px 7px',
                    borderRadius: 5,
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    {agendaId}
                  </span>
                  {renderIdExtra?.(agenda, agendaId)}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  {isEditing ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <input
                        value={editTitle}
                        onChange={e => setEditTitle(e.target.value)}
                        placeholder="Agenda Title"
                        style={{
                          ...inputStyle,
                          fontWeight: 600,
                          borderColor: 'hsl(140, 70%, 45%)',
                        }}
                      />
                      <textarea
                        value={editDescription}
                        onChange={e => setEditDescription(e.target.value)}
                        placeholder="Agenda Description / Details"
                        style={{
                          ...inputStyle,
                          minHeight: 48,
                          fontSize: '.76rem',
                          borderColor: 'hsl(140, 70%, 45% / .6)',
                        }}
                      />
                      <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 2 }}>
                        <button
                          type="button"
                          onClick={() => setEditingId(null)}
                          style={{
                            padding: '3px 9px',
                            fontSize: '.72rem',
                            borderRadius: 5,
                            border: '1px solid hsl(var(--border) / .4)',
                            background: 'transparent',
                            cursor: 'pointer',
                            color: 'hsl(var(--ink))',
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          disabled={saving || !editTitle.trim()}
                          onClick={async () => {
                            await onUpdate(aIdx, { title: editTitle.trim(), description: editDescription })
                            setEditingId(null)
                          }}
                          style={{
                            padding: '3px 11px',
                            fontSize: '.72rem',
                            borderRadius: 5,
                            background: 'hsl(140, 70%, 45%)',
                            color: '#ffffff',
                            border: 'none',
                            fontWeight: 700,
                            cursor: 'pointer',
                          }}
                        >
                          {saving ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                          fontSize: compact ? '.84rem' : '.92rem',
                          fontWeight: 700,
                          color: 'hsl(var(--ink))',
                          flex: 1,
                          minWidth: 0,
                          fontFamily: 'Inter, sans-serif',
                        }}>
                          {agenda.title}
                        </div>

                        {onMove && (
                          <div style={{ display: 'flex', gap: 2 }}>
                            <button
                              type="button"
                              onClick={() => onMove(aIdx, aIdx - 1)}
                              disabled={saving || aIdx === 0}
                              title="Move up"
                              style={{ ...iconBtnStyle, opacity: aIdx === 0 ? 0.3 : 1 }}
                            >
                              <ChevronUp size={13} />
                            </button>
                            <button
                              type="button"
                              onClick={() => onMove(aIdx, aIdx + 1)}
                              disabled={saving || aIdx === agendas.length - 1}
                              title="Move down"
                              style={{ ...iconBtnStyle, opacity: aIdx === agendas.length - 1 ? 0.3 : 1 }}
                            >
                              <ChevronDown size={13} />
                            </button>
                          </div>
                        )}

                        <button
                          type="button"
                          onClick={() => {
                            setEditingId(agendaId)
                            setEditTitle(agenda.title || '')
                            setEditDescription(agenda.description || '')
                          }}
                          title="Edit Agenda Item"
                          style={iconBtnStyle}
                        >
                          <Pencil size={12} />
                        </button>

                        <button
                          type="button"
                          onClick={() => onDelete(agendaId, agenda.title, aIdx)}
                          title="Delete Agenda Item"
                          style={{ ...iconBtnStyle, color: 'hsl(var(--destructive))' }}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>

                      {agenda.description && (
                        <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 3, lineHeight: 1.4 }}>
                          {agenda.description}
                        </div>
                      )}

                      {(presenterName || metaExtra) && (
                        <div style={{ marginTop: 5, display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                          {presenterName && (
                            <div style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4,
                              background: 'hsl(280, 75%, 60% / .1)',
                              color: 'hsl(280, 75%, 60%)',
                              border: '1px solid hsl(280, 75%, 60% / .3)',
                              padding: '2px 8px',
                              borderRadius: 6,
                              fontSize: '.7rem',
                              fontWeight: 600,
                            }}>
                              <User size={10} /> Presenter: {presenterName}
                            </div>
                          )}
                          {metaExtra}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              {renderExtras?.(agenda, aIdx, agendaId)}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function moveItem<T>(items: T[], from: number, to: number): T[] {
  if (to < 0 || to >= items.length || from === to) return items
  const next = [...items]
  const [moved] = next.splice(from, 1)
  next.splice(to, 0, moved)
  return next
}
