import { useState, useEffect, useRef, useCallback } from 'react'
import {
  BookOpen, Plus, Trash2, Pencil, Save, Loader, Search, Download, Upload,
  X, CheckCircle, AlertTriangle, Sparkles, FileText, ChevronRight,
  Tag, Zap,
} from 'lucide-react'
import api from '../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface Shortcut {
  id: string
  shortcut: string
  full_form: string
  created_at: string
}

interface VocabWord {
  id: string
  word: string
  created_at: string
}

type Tab = 'shortcuts' | 'vocabulary' | 'import'
type ImportMode = 'rule' | 'ai'

// ── Helpers ──────────────────────────────────────────────────────────────────

function SectionHeader({ icon: Icon, title, color, count }: {
  icon: React.ComponentType<{ size?: number, style?: React.CSSProperties }>
  title: string
  color: string
  count?: number
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.5rem' }}>
      <div style={{
        width: 36, height: 36, borderRadius: '9px',
        background: `${color}18`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: `1.5px solid ${color}35`,
      }}>
        <Icon size={17} style={{ color }} />
      </div>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
        {title}
      </h2>
      {count !== undefined && (
        <span style={{
          fontSize: '.72rem', fontWeight: 700,
          color, background: `${color}12`,
          padding: '.15rem .55rem', borderRadius: '999px',
          border: `1.5px solid ${color}30`,
          fontFamily: 'Inter, sans-serif',
        }}>{count}</span>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// SHORTCUT TAB
// ════════════════════════════════════════════════════════════════════════════

function ShortcutTab() {
  const [items, setItems] = useState<Shortcut[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editShortcut, setEditShortcut] = useState('')
  const [editFull, setEditFull] = useState('')
  const [saving, setSaving] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [newShortcut, setNewShortcut] = useState('')
  const [newFull, setNewFull] = useState('')
  const [adding, setAdding] = useState(false)
  const [showAddRow, setShowAddRow] = useState(false)
  const importRef = useRef<HTMLInputElement>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await api.get('/dictionary/shortcuts')
      setItems(Array.isArray(res.data) ? res.data : [])
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { load() }, [])

  const handleAdd = async () => {
    if (!newShortcut.trim() || !newFull.trim()) return
    setAdding(true)
    try {
      const res = await api.post('/dictionary/shortcuts', { shortcut: newShortcut, full_form: newFull })
      setItems(prev => [res.data, ...prev])
      setNewShortcut('')
      setNewFull('')
      setShowAddRow(false)
    } catch { } finally { setAdding(false) }
  }

  const handleEdit = async (id: string) => {
    setSaving(true)
    try {
      const res = await api.put(`/dictionary/shortcuts/${id}`, { shortcut: editShortcut, full_form: editFull })
      setItems(prev => prev.map(i => i.id === id ? res.data : i))
      setEditingId(null)
    } catch { } finally { setSaving(false) }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this shortcut?')) return
    setDeletingId(id)
    try {
      await api.delete(`/dictionary/shortcuts/${id}`)
      setItems(prev => prev.filter(i => i.id !== id))
    } catch { } finally { setDeletingId(null) }
  }

  const handleExport = async () => {
    const res = await api.get('/dictionary/shortcuts/export', { responseType: 'blob' })
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'shortcuts.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await api.post('/dictionary/shortcuts/import', formData)
      alert(`Imported ${res.data.imported} shortcuts.`)
      load()
    } catch { }
    e.target.value = ''
  }

  const filtered = items.filter(i =>
    i.shortcut.toLowerCase().includes(search.toLowerCase()) ||
    i.full_form.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      <SectionHeader icon={Zap} title="Shortcut Dictionary" color="hsl(14,90%,56%)" count={items.length} />

      {/* Toolbar */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: '160px' }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'hsl(var(--pencil))' }} />
          <input
            id="shortcut-search"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search shortcuts…"
            style={{
              width: '100%', paddingLeft: '30px', padding: '.55rem .75rem .55rem 30px',
              borderRadius: '8px', border: '1.5px solid hsl(var(--ink) / .12)',
              background: 'hsl(var(--muted) / .5)', color: 'hsl(var(--ink))',
              fontFamily: 'Inter, sans-serif', fontSize: '.85rem', outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>
        <button className="btn btn-primary" onClick={() => setShowAddRow(true)} style={{ padding: '.55rem 1rem', fontSize: '.85rem' }} id="add-shortcut-btn">
          <Plus size={13} /> Add
        </button>
        <button className="icon-btn" title="Export CSV" onClick={handleExport} style={{ width: 36, height: 36 }}>
          <Download size={14} />
        </button>
        <button className="icon-btn" title="Import CSV" onClick={() => importRef.current?.click()} style={{ width: 36, height: 36 }}>
          <Upload size={14} />
        </button>
        <input ref={importRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={handleImport} />
      </div>

      {/* Add row */}
      {showAddRow && (
        <div className="dictionary-add-row animate-slide-up" style={{
          display: 'grid', gridTemplateColumns: '1fr 2fr auto auto',
          gap: '8px', marginBottom: '12px', alignItems: 'center',
          padding: '10px', borderRadius: '10px',
          background: 'hsl(var(--accent) / .05)',
          border: '1.5px solid hsl(var(--accent) / .2)',
        }}>
          <input id="new-shortcut-input" className="input" value={newShortcut} onChange={e => setNewShortcut(e.target.value)} placeholder="LLM" style={{ fontSize: '.85rem', padding: '.5rem' }} />
          <input id="new-full-input" className="input" value={newFull} onChange={e => setNewFull(e.target.value)} placeholder="Large Language Model" style={{ fontSize: '.85rem', padding: '.5rem' }}
            onKeyDown={e => e.key === 'Enter' && handleAdd()} />
          <button className="btn btn-primary" onClick={handleAdd} disabled={adding} style={{ padding: '.5rem .9rem', fontSize: '.85rem' }}>
            {adding ? <Loader size={13} className="spin" /> : <Save size={13} />}
          </button>
          <button className="icon-btn" onClick={() => { setShowAddRow(false); setNewShortcut(''); setNewFull('') }} style={{ width: 36, height: 36 }}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
          <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        </div>
      ) : filtered.length === 0 ? (
        <div className="dictionary-table" style={{
          padding: '3.5rem 2rem', textAlign: 'center',
          background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1.5px dashed hsl(var(--ink) / .15)',
          color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontSize: '.9rem',
        }}>
          {search ? 'No shortcuts match your search.' : 'No shortcuts yet. Add your first one above.'}
        </div>
      ) : (
        <div style={{
          border: '1.5px solid hsl(var(--ink) / .1)', borderRadius: '12px', overflow: 'hidden',
          background: 'hsl(var(--card))',
        }}>
          {/* Header */}
          <div className="dictionary-table-row dictionary-table-header" style={{
            display: 'grid', gridTemplateColumns: '140px 1fr auto',
            padding: '.6rem 1rem', background: 'hsl(var(--muted) / .5)',
            borderBottom: '1px solid hsl(var(--ink) / .08)',
            fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '.07em', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif',
          }}>
            <span>Shortcut</span>
            <span>Full Form</span>
            <span />
          </div>
          {filtered.map((item, idx) => (
            <div key={item.id} className="dictionary-table-row" style={{
              display: 'grid', gridTemplateColumns: '140px 1fr auto',
              padding: '.7rem 1rem', alignItems: 'center',
              borderBottom: idx < filtered.length - 1 ? '1px solid hsl(var(--ink) / .06)' : 'none',
              background: editingId === item.id ? 'hsl(var(--accent) / .04)' : 'transparent',
              transition: 'background .12s',
            }}
              onMouseEnter={e => editingId !== item.id && (e.currentTarget.style.background = 'hsl(var(--muted) / .4)')}
              onMouseLeave={e => editingId !== item.id && (e.currentTarget.style.background = 'transparent')}
            >
              {editingId === item.id ? (
                <>
                  <input className="input" value={editShortcut} onChange={e => setEditShortcut(e.target.value)} style={{ fontSize: '.85rem', padding: '.4rem .6rem' }} />
                  <input className="input" value={editFull} onChange={e => setEditFull(e.target.value)} style={{ fontSize: '.85rem', padding: '.4rem .6rem', marginLeft: '8px' }}
                    onKeyDown={e => e.key === 'Enter' && handleEdit(item.id)} />
                  <div style={{ display: 'flex', gap: '6px', marginLeft: '8px' }}>
                    <button className="btn btn-primary" onClick={() => handleEdit(item.id)} disabled={saving} style={{ padding: '.4rem .75rem', fontSize: '.82rem' }}>
                      {saving ? <Loader size={12} className="spin" /> : <Save size={12} />}
                    </button>
                    <button className="icon-btn" onClick={() => setEditingId(null)} style={{ width: 32, height: 32 }}>
                      <X size={13} />
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <span style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem', fontWeight: 700,
                    color: 'hsl(14,90%,56%)',
                    background: 'hsl(14,90%,56% / .1)', padding: '.2rem .6rem',
                    borderRadius: '6px', border: '1px solid hsl(14,90%,56% / .2)',
                    display: 'inline-block',
                  }}>{item.shortcut}</span>
                  <span style={{ fontSize: '.88rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', paddingLeft: '12px' }}>
                    {item.full_form}
                  </span>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button className="icon-btn" style={{ width: 30, height: 30 }} title="Edit"
                      onClick={() => { setEditingId(item.id); setEditShortcut(item.shortcut); setEditFull(item.full_form) }}>
                      <Pencil size={12} />
                    </button>
                    <button className="icon-btn" style={{ width: 30, height: 30, color: 'hsl(var(--destructive) / .7)' }} title="Delete"
                      onClick={() => handleDelete(item.id)} disabled={deletingId === item.id}
                      onMouseEnter={e => (e.currentTarget.style.color = 'hsl(var(--destructive))')}
                      onMouseLeave={e => (e.currentTarget.style.color = 'hsl(var(--destructive) / .7)')}>
                      {deletingId === item.id ? <Loader size={12} className="spin" /> : <Trash2 size={12} />}
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// VOCABULARY TAB
// ════════════════════════════════════════════════════════════════════════════

function VocabularyTab() {
  const [items, setItems] = useState<VocabWord[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [newWord, setNewWord] = useState('')
  const [adding, setAdding] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await api.get('/dictionary/vocabulary')
      setItems(Array.isArray(res.data) ? res.data : [])
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { load() }, [])

  const handleAdd = async () => {
    if (!newWord.trim()) return
    setAdding(true)
    try {
      const res = await api.post('/dictionary/vocabulary', { word: newWord.trim() })
      setItems(prev => [...prev, res.data].sort((a, b) => a.word.localeCompare(b.word)))
      setNewWord('')
    } catch { } finally { setAdding(false) }
  }

  const handleDelete = async (id: string) => {
    setDeletingId(id)
    try {
      await api.delete(`/dictionary/vocabulary/${id}`)
      setItems(prev => prev.filter(i => i.id !== id))
    } catch { } finally { setDeletingId(null) }
  }

  const filtered = items.filter(i => i.word.toLowerCase().includes(search.toLowerCase()))

  return (
    <div>
      <SectionHeader icon={Tag} title="Technical Vocabulary" color="hsl(205,90%,55%)" count={items.length} />

      <div style={{ display: 'flex', gap: '8px', marginBottom: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: '160px' }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'hsl(var(--pencil))' }} />
          <input
            id="vocab-search"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search vocabulary…"
            style={{
              width: '100%', paddingLeft: '30px', padding: '.55rem .75rem .55rem 30px',
              borderRadius: '8px', border: '1.5px solid hsl(var(--ink) / .12)',
              background: 'hsl(var(--muted) / .5)', color: 'hsl(var(--ink))',
              fontFamily: 'Inter, sans-serif', fontSize: '.85rem', outline: 'none', boxSizing: 'border-box',
            }}
          />
        </div>
        <input
          id="new-vocab-input"
          className="input"
          value={newWord}
          onChange={e => setNewWord(e.target.value)}
          placeholder="Add a term…"
          onKeyDown={e => e.key === 'Enter' && handleAdd()}
          style={{ width: '160px', padding: '.55rem .75rem', fontSize: '.85rem' }}
        />
        <button className="btn btn-primary" onClick={handleAdd} disabled={adding} style={{ padding: '.55rem 1rem', fontSize: '.85rem' }} id="add-vocab-btn">
          {adding ? <Loader size={13} className="spin" /> : <Plus size={13} />} Add
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
          <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        </div>
      ) : filtered.length === 0 ? (
        <div style={{
          padding: '3.5rem 2rem', textAlign: 'center',
          background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1.5px dashed hsl(var(--ink) / .15)',
          color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontSize: '.9rem',
        }}>
          {search ? 'No words match your search.' : 'No vocabulary yet. Add technical terms above.'}
        </div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {filtered.map(item => (
            <div key={item.id} className="animate-slide-up" style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '.4rem .75rem .4rem 1rem',
              background: 'hsl(205,90%,55% / .08)',
              border: '1.5px solid hsl(205,90%,55% / .2)',
              borderRadius: '999px',
              transition: 'all .12s',
            }}
              onMouseEnter={e => (e.currentTarget.style.background = 'hsl(205,90%,55% / .14)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'hsl(205,90%,55% / .08)')}
            >
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '.82rem', fontWeight: 700, color: 'hsl(205,90%,55%)' }}>
                {item.word}
              </span>
              <button
                onClick={() => handleDelete(item.id)}
                disabled={deletingId === item.id}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'hsl(205,90%,55% / .6)', padding: '0', display: 'flex',
                  transition: 'color .12s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = 'hsl(var(--destructive))')}
                onMouseLeave={e => (e.currentTarget.style.color = 'hsl(205,90%,55% / .6)')}
              >
                {deletingId === item.id ? <Loader size={11} className="spin" /> : <X size={11} />}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// IMPORT TAB
// ════════════════════════════════════════════════════════════════════════════

function ImportTab() {
  const [mode, setMode] = useState<ImportMode>('rule')
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [ruleWords, setRuleWords] = useState<string[]>([])
  const [aiResult, setAiResult] = useState<{ technical_words: string[], shortcuts: { short: string, full: string }[] } | null>(null)
  const [selectedWords, setSelectedWords] = useState<Set<string>>(new Set())
  const [selectedShortcuts, setSelectedShortcuts] = useState<Set<number>>(new Set())
  const [saving, setSaving] = useState(false)
  const [savedMsg, setSavedMsg] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) { setFile(f); setRuleWords([]); setAiResult(null) }
  }, [])

  const handleExtract = async () => {
    if (!file) return
    setExtracting(true)
    setRuleWords([]); setAiResult(null)
    setSelectedWords(new Set()); setSelectedShortcuts(new Set())
    const formData = new FormData()
    formData.append('file', file)
    try {
      if (mode === 'rule') {
        const res = await api.post('/dictionary/extract/rule-based', formData)
        setRuleWords(res.data.words || [])
        setSelectedWords(new Set(res.data.words || []))
      } else {
        const res = await api.post('/dictionary/extract/ai', formData)
        setAiResult(res.data)
        setSelectedWords(new Set(res.data.technical_words || []))
        setSelectedShortcuts(new Set((res.data.shortcuts || []).map((_: unknown, i: number) => i)))
      }
    } catch (e: any) {
      alert('Extraction failed: ' + (e?.response?.data?.detail || e?.message || 'Unknown error'))
    } finally {
      setExtracting(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      if (ruleWords.length > 0) {
        const words = ruleWords.filter(w => selectedWords.has(w))
        await api.post('/dictionary/vocabulary/bulk', { words })
        setSavedMsg(`Saved ${words.length} vocabulary words.`)
      } else if (aiResult) {
        const words = (aiResult.technical_words || []).filter(w => selectedWords.has(w))
        const shortcuts = (aiResult.shortcuts || [])
          .filter((_, i) => selectedShortcuts.has(i))
          .map(s => ({ short: s.short, full: s.full }))

        if (words.length > 0) await api.post('/dictionary/vocabulary/bulk', { words })
        if (shortcuts.length > 0) {
          for (const s of shortcuts) {
            try {
              await api.post('/dictionary/shortcuts', { shortcut: s.short, full_form: s.full })
            } catch { }
          }
        }
        setSavedMsg(`Saved ${words.length} words + ${shortcuts.length} shortcuts.`)
      }
      setTimeout(() => setSavedMsg(''), 3500)
      setFile(null); setRuleWords([]); setAiResult(null)
    } catch { } finally { setSaving(false) }
  }

  const hasResults = ruleWords.length > 0 || aiResult !== null
  const ACCEPT = '.pdf,.docx,.txt,.md'

  return (
    <div>
      <SectionHeader icon={FileText} title="Import from Document" color="hsl(280,70%,60%)" />

      {/* Mode toggle */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '1.25rem' }}>
        {([
          { id: 'rule' as ImportMode, label: 'Rule-Based', desc: 'Fast & deterministic' },
          { id: 'ai' as ImportMode, label: 'AI Assisted', desc: 'Recommended', badge: true },
        ]).map(m => (
          <button
            key={m.id}
            id={`import-mode-${m.id}`}
            onClick={() => setMode(m.id)}
            style={{
              flex: 1, padding: '.75rem', borderRadius: '10px',
              border: `2px solid ${mode === m.id ? 'hsl(280,70%,60%)' : 'hsl(var(--ink) / .12)'}`,
              background: mode === m.id ? 'hsl(280,70%,60% / .07)' : 'hsl(var(--card))',
              cursor: 'pointer', transition: 'all .15s', textAlign: 'left',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
              <span style={{ fontWeight: 700, fontSize: '.88rem', color: mode === m.id ? 'hsl(280,70%,60%)' : 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
                {m.label}
              </span>
              {m.badge && (
                <span style={{
                  fontSize: '.65rem', fontWeight: 700,
                  color: 'hsl(280,70%,60%)', background: 'hsl(280,70%,60% / .12)',
                  border: '1px solid hsl(280,70%,60% / .25)',
                  padding: '.1rem .45rem', borderRadius: '999px',
                  display: 'flex', alignItems: 'center', gap: '3px',
                }}>
                  <Sparkles size={9} /> Recommended
                </span>
              )}
            </div>
            <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>{m.desc}</div>
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onClick={() => fileRef.current?.click()}
        style={{
          padding: '2.5rem 2rem',
          border: `2px dashed ${dragging ? 'hsl(280,70%,60%)' : file ? 'hsl(130,60%,45%)' : 'hsl(var(--ink) / .2)'}`,
          borderRadius: '12px',
          background: dragging ? 'hsl(280,70%,60% / .05)' : file ? 'hsl(130,60%,45% / .04)' : 'hsl(var(--muted) / .3)',
          textAlign: 'center', cursor: 'pointer', transition: 'all .2s',
          marginBottom: '1rem',
        }}
      >
        <div style={{
          width: 52, height: 52, borderRadius: '50%',
          background: file ? 'hsl(130,60%,45% / .12)' : 'hsl(280,70%,60% / .1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto .85rem',
        }}>
          {file
            ? <CheckCircle size={24} style={{ color: 'hsl(130,60%,45%)' }} />
            : <Upload size={24} style={{ color: 'hsl(280,70%,60%)' }} />
          }
        </div>
        {file ? (
          <>
            <p style={{ fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '.35rem' }}>
              {file.name}
            </p>
            <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
              {(file.size / 1024).toFixed(1)} KB · Click to change file
            </p>
          </>
        ) : (
          <>
            <p style={{ fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '.35rem' }}>
              Drop a document here
            </p>
            <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
              PDF, DOCX, TXT, Markdown · or click to browse
            </p>
          </>
        )}
        <input ref={fileRef} type="file" accept={ACCEPT} style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) { setFile(f); setRuleWords([]); setAiResult(null) }; e.target.value = '' }}
        />
      </div>

      <button
        className="btn btn-primary"
        onClick={handleExtract}
        disabled={!file || extracting}
        style={{ width: '100%', padding: '.75rem', fontSize: '.9rem', marginBottom: '1.25rem' }}
        id="extract-vocab-btn"
      >
        {extracting ? (
          <><Loader size={15} className="spin" /> Extracting with {mode === 'ai' ? 'AI' : 'rules'}…</>
        ) : (
          <>{mode === 'ai' ? <Sparkles size={15} /> : <Zap size={15} />} Extract Vocabulary</>
        )}
      </button>

      {/* Results */}
      {hasResults && (
        <div className="animate-slide-up" style={{
          border: '1.5px solid hsl(var(--ink) / .1)', borderRadius: '12px',
          overflow: 'hidden', background: 'hsl(var(--card))',
        }}>
          <div style={{
            padding: '.75rem 1rem',
            background: 'hsl(var(--muted) / .5)',
            borderBottom: '1px solid hsl(var(--ink) / .08)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={13} style={{ color: 'hsl(45,90%,50%)' }} />
              <span style={{ fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
                Review before saving — select items to include
              </span>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button onClick={() => {
                if (ruleWords.length > 0) setSelectedWords(new Set(ruleWords))
                if (aiResult) { setSelectedWords(new Set(aiResult.technical_words)); setSelectedShortcuts(new Set(aiResult.shortcuts.map((_, i) => i))) }
              }} style={{ fontSize: '.73rem', color: 'hsl(var(--accent))', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>Select all</button>
              <button onClick={() => { setSelectedWords(new Set()); setSelectedShortcuts(new Set()) }}
                style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>Clear</button>
            </div>
          </div>

          <div style={{ padding: '1rem', maxHeight: '380px', overflowY: 'auto' }}>
            {/* Rule-based words */}
            {ruleWords.length > 0 && (
              <div>
                <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.07em', color: 'hsl(var(--pencil))', marginBottom: '.75rem', fontFamily: 'Inter, sans-serif' }}>
                  Extracted Words ({selectedWords.size}/{ruleWords.length} selected)
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {ruleWords.map(w => {
                    const sel = selectedWords.has(w)
                    return (
                      <button key={w} onClick={() => setSelectedWords(prev => { const n = new Set(prev); sel ? n.delete(w) : n.add(w); return n })}
                        style={{
                          padding: '.3rem .7rem', borderRadius: '999px', cursor: 'pointer',
                          border: `1.5px solid ${sel ? 'hsl(205,90%,55% / .4)' : 'hsl(var(--ink) / .12)'}`,
                          background: sel ? 'hsl(205,90%,55% / .1)' : 'hsl(var(--muted) / .4)',
                          color: sel ? 'hsl(205,90%,55%)' : 'hsl(var(--pencil))',
                          fontFamily: 'JetBrains Mono, monospace', fontSize: '.78rem', fontWeight: 600,
                          transition: 'all .12s',
                        }}>
                        {w}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            {/* AI results */}
            {aiResult && (
              <>
                {aiResult.technical_words.length > 0 && (
                  <div style={{ marginBottom: '1.25rem' }}>
                    <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.07em', color: 'hsl(var(--pencil))', marginBottom: '.75rem', fontFamily: 'Inter, sans-serif' }}>
                      Technical Terms ({aiResult.technical_words.filter(w => selectedWords.has(w)).length}/{aiResult.technical_words.length} selected)
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {aiResult.technical_words.map(w => {
                        const sel = selectedWords.has(w)
                        return (
                          <button key={w} onClick={() => setSelectedWords(prev => { const n = new Set(prev); sel ? n.delete(w) : n.add(w); return n })}
                            style={{
                              padding: '.3rem .7rem', borderRadius: '999px', cursor: 'pointer',
                              border: `1.5px solid ${sel ? 'hsl(205,90%,55% / .4)' : 'hsl(var(--ink) / .12)'}`,
                              background: sel ? 'hsl(205,90%,55% / .1)' : 'hsl(var(--muted) / .4)',
                              color: sel ? 'hsl(205,90%,55%)' : 'hsl(var(--pencil))',
                              fontFamily: 'JetBrains Mono, monospace', fontSize: '.78rem', fontWeight: 600,
                              transition: 'all .12s',
                            }}>
                            {w}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
                {aiResult.shortcuts.length > 0 && (
                  <div>
                    <div style={{ fontSize: '.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.07em', color: 'hsl(var(--pencil))', marginBottom: '.75rem', fontFamily: 'Inter, sans-serif' }}>
                      Abbreviations ({aiResult.shortcuts.filter((_, i) => selectedShortcuts.has(i)).length}/{aiResult.shortcuts.length} selected)
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {aiResult.shortcuts.map((s, i) => {
                        const sel = selectedShortcuts.has(i)
                        return (
                          <button key={i} onClick={() => setSelectedShortcuts(prev => { const n = new Set(prev); sel ? n.delete(i) : n.add(i); return n })}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '10px', padding: '.55rem .75rem',
                              borderRadius: '8px', cursor: 'pointer', textAlign: 'left',
                              border: `1.5px solid ${sel ? 'hsl(14,90%,56% / .35)' : 'hsl(var(--ink) / .1)'}`,
                              background: sel ? 'hsl(14,90%,56% / .06)' : 'hsl(var(--muted) / .3)',
                              transition: 'all .12s',
                            }}>
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace', fontSize: '.82rem', fontWeight: 700,
                              color: 'hsl(14,90%,56%)', minWidth: '80px',
                            }}>{s.short}</span>
                            <ChevronRight size={12} style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
                            <span style={{ fontSize: '.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>{s.full}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div style={{ padding: '.85rem 1rem', borderTop: '1px solid hsl(var(--ink) / .08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            {savedMsg && (
              <span style={{ fontSize: '.82rem', color: 'hsl(130,60%,45%)', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle size={13} /> {savedMsg}
              </span>
            )}
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
              <button className="btn" style={{ padding: '.6rem 1rem', fontSize: '.85rem' }}
                onClick={() => { setRuleWords([]); setAiResult(null) }}>
                Discard
              </button>
              <button className="btn btn-primary" onClick={handleSave} disabled={saving || (selectedWords.size === 0 && selectedShortcuts.size === 0)}
                style={{ padding: '.6rem 1.25rem', fontSize: '.85rem' }} id="save-extracted-btn">
                {saving ? <Loader size={13} className="spin" /> : <Save size={13} />}
                Save Selected
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN DICTIONARY PAGE
// ════════════════════════════════════════════════════════════════════════════

const TAB_CONFIG: { id: Tab; label: string; icon: React.ComponentType<{ size?: number, style?: React.CSSProperties }> }[] = [
  { id: 'shortcuts', label: 'Shortcuts', icon: Zap },
  { id: 'vocabulary', label: 'Vocabulary', icon: Tag },
  { id: 'import', label: 'Import from Doc', icon: FileText },
]

export default function DictionaryPage() {
  const [tab, setTab] = useState<Tab>('shortcuts')

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>

      {/* Panel Header */}
      <div className="panel-header">
        <div style={{
          width: 34, height: 34, borderRadius: '10px', flexShrink: 0,
          background: 'hsl(280,70%,60% / .12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '2px solid hsl(280,70%,60% / .3)',
        }}>
          <BookOpen size={16} style={{ color: 'hsl(280,70%,60%)' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1>Dictionary</h1>
          <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
            Shortcuts, vocabulary & document import
          </p>
        </div>
      </div>

      <div className="page-wrapper">

        {/* Tab nav */}
        <div style={{
          display: 'flex', gap: '4px',
          background: 'hsl(var(--muted) / .5)',
          borderRadius: '12px', padding: '4px',
          marginBottom: '2rem',
          border: '1.5px solid hsl(var(--ink) / .08)',
        }}>
          {TAB_CONFIG.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              id={`dict-tab-${id}`}
              onClick={() => setTab(id)}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px',
                padding: '.6rem .5rem',
                borderRadius: '9px',
                border: 'none',
                background: tab === id ? 'hsl(var(--card))' : 'transparent',
                boxShadow: tab === id ? '0 1px 6px rgba(0,0,0,.1)' : 'none',
                cursor: 'pointer',
                fontWeight: 600, fontSize: '.83rem',
                color: tab === id ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                fontFamily: 'Inter, sans-serif',
                transition: 'all .15s',
              }}
            >
              <Icon size={14} style={{ opacity: tab === id ? 1 : 0.6 }} />
              {label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <section className="animate-slide-up" key={tab} style={{ animationDelay: '0.03s', animationFillMode: 'both' }}>
          {tab === 'shortcuts' && <ShortcutTab />}
          {tab === 'vocabulary' && <VocabularyTab />}
          {tab === 'import' && <ImportTab />}
        </section>
      </div>
    </div>
  )
}
