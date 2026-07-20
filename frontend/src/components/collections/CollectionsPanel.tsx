import React, { useState, useEffect } from 'react'
import { Plus, FolderOpen, Loader, Trash2, Edit2, Calendar } from 'lucide-react'
import type { Collection, CollectionDetail } from '../../types/recording'
import { listCollections, getCollection, deleteCollection } from '../../api/collections'
import CreateCollectionDialog from './CreateCollectionDialog'
import CollectionDetailView from './CollectionDetailView'

export default function CollectionsPanel() {
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(true)
  
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [editCollection, setEditCollection] = useState<Collection | null>(null)
  
  const [activeCollectionId, setActiveCollectionId] = useState<string | null>(null)
  const [activeCollectionDetail, setActiveCollectionDetail] = useState<CollectionDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  const fetchCollections = async () => {
    setLoading(true)
    try {
      const data = await listCollections()
      setCollections(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCollections()
  }, [])

  const fetchDetail = async (id: string) => {
    setLoadingDetail(true)
    try {
      const data = await getCollection(id)
      setActiveCollectionDetail(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingDetail(false)
    }
  }

  useEffect(() => {
    if (activeCollectionId) {
      fetchDetail(activeCollectionId)
    } else {
      setActiveCollectionDetail(null)
    }
  }, [activeCollectionId])

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    setDeletingId(id)
    try {
      await deleteCollection(id)
      setCollections(prev => prev.filter(c => c.id !== id))
      setConfirmDeleteId(null)
    } catch (err) {
      console.error(err)
    } finally {
      setDeletingId(null)
    }
  }

  if (activeCollectionId && (loadingDetail || activeCollectionDetail)) {
    if (loadingDetail) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6rem 2rem', flexDirection: 'column', gap: '1.5rem' }}>
          <Loader size={36} className="spin" style={{ color: 'hsl(var(--accent))' }} />
          <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '1rem', color: 'hsl(var(--pencil))', fontWeight: 500 }}>
            Loading collection...
          </p>
        </div>
      )
    }
    return (
      <CollectionDetailView 
        collection={activeCollectionDetail!} 
        onBack={() => setActiveCollectionId(null)}
        onDeleted={() => {
          setActiveCollectionId(null)
          fetchCollections()
        }}
        onUpdated={() => {
          fetchDetail(activeCollectionId)
          fetchCollections()
        }}
      />
    )
  }

  return (
    <div style={{ padding: '1.5rem 2rem 2rem', flex: 1, overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'hsl(var(--ink))', margin: 0 }}>
          Your Collections
        </h2>
        <button className="btn btn-primary" onClick={() => { setEditCollection(null); setCreateDialogOpen(true) }} style={{ padding: '.6rem 1.2rem', gap: '.5rem' }}>
          <Plus size={16} /> Create Collection
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6rem 2rem' }}>
          <Loader size={36} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        </div>
      ) : collections.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '6rem 2rem', maxWidth: '500px', margin: '0 auto' }}>
          <div style={{
            width: '120px', height: '120px', borderRadius: '50%',
            background: 'linear-gradient(135deg, hsl(var(--accent) / .1), hsl(var(--accent) / .05))',
            border: '3px dashed hsl(var(--accent) / .3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 2rem'
          }}>
            <FolderOpen size={48} style={{ color: 'hsl(var(--accent))', opacity: 0.5 }} />
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))' }}>
            No Collections Yet
          </h2>
          <p style={{ fontSize: '1rem', color: 'hsl(var(--pencil))', lineHeight: 1.6, marginBottom: '2rem' }}>
            Group your meetings by project, team, or topic. Create your first collection to get started.
          </p>
          <button className="btn btn-primary" onClick={() => { setEditCollection(null); setCreateDialogOpen(true) }}>
            <Plus size={18} style={{ marginRight: '.5rem' }} /> Create Collection
          </button>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 360px), 1fr))' }}>
          {collections.map(c => (
            <div
              key={c.id}
              onClick={() => setActiveCollectionId(c.id)}
              style={{
                background: 'hsl(var(--card))', border: '2px solid hsl(var(--border) / .15)',
                borderRadius: '14px', padding: '1.5rem', cursor: 'pointer',
                transition: 'all .2s cubic-bezier(0.4, 0, 0.2, 1)',
                display: 'flex', flexDirection: 'column'
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--accent) / .35)'
                el.style.transform = 'translateY(-2px)'
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--border) / .15)'
                el.style.transform = 'translateY(0)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px solid hsl(var(--accent) / .25)' }}>
                  <FolderOpen size={24} style={{ color: 'hsl(var(--accent))' }} />
                </div>
                <div style={{ display: 'flex', gap: '.5rem' }} onClick={e => e.stopPropagation()}>
                  {confirmDeleteId === c.id ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', padding: '.25rem .5rem', background: 'hsl(var(--card))', border: '1px solid hsl(var(--destructive) / .3)', borderRadius: '8px', fontSize: '.8rem' }}>
                      <span style={{ fontWeight: 600 }}>Delete?</span>
                      <button className="btn" onClick={e => handleDelete(e, c.id)} disabled={deletingId === c.id} style={{ padding: '.2rem .5rem', background: 'hsl(var(--destructive))', color: 'white', minHeight: 'unset', height: 'auto' }}>
                        {deletingId === c.id ? '...' : 'Yes'}
                      </button>
                      <button className="btn btn-ghost" onClick={() => setConfirmDeleteId(null)} style={{ padding: '.2rem .5rem', minHeight: 'unset', height: 'auto' }}>No</button>
                    </div>
                  ) : (
                    <>
                      <button className="icon-btn" onClick={(e) => { e.stopPropagation(); setEditCollection(c); setCreateDialogOpen(true); }} style={{ width: '32px', height: '32px', color: 'hsl(var(--pencil))' }}>
                        <Edit2 size={16} />
                      </button>
                      <button className="icon-btn" onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(c.id); }} style={{ width: '32px', height: '32px', color: 'hsl(var(--destructive) / .8)' }}>
                        <Trash2 size={16} />
                      </button>
                    </>
                  )}
                </div>
              </div>

              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: '0 0 .5rem 0' }}>
                {c.name}
              </h3>
              
              {c.description && (
                <p style={{ fontSize: '.9rem', color: 'hsl(var(--pencil))', margin: '0 0 1rem 0', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                  {c.description}
                </p>
              )}
              
              <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '1rem', borderTop: '1px solid hsl(var(--border) / .1)' }}>
                <span style={{ fontSize: '.85rem', fontWeight: 600, color: 'hsl(var(--accent))', background: 'hsl(var(--accent) / .1)', padding: '.25rem .75rem', borderRadius: '8px' }}>
                  {c.meeting_count} {c.meeting_count === 1 ? 'Meeting' : 'Meetings'}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                  <Calendar size={12} /> {new Date(c.updated_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {createDialogOpen && (
        <CreateCollectionDialog
          open={createDialogOpen}
          onClose={() => setCreateDialogOpen(false)}
          editCollection={editCollection}
          onCreated={(c) => {
            setCreateDialogOpen(false)
            fetchCollections()
          }}
        />
      )}
    </div>
  )
}
