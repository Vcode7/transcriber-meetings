import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Database, Loader2, RefreshCw, XCircle } from 'lucide-react'
import { toast } from 'sonner'
import {
  embeddingIndexApi,
  EmbeddingIndexStatus,
  ReindexJobStatus,
} from '../services/embeddingTrainingApi'

interface EmbeddingReindexCardProps {
  /** Bump to reload the index status (e.g. after the embedding model was changed). */
  refreshKey?: number
}

const RUNNING = new Set(['running', 'cancelling'])

const errorDetail = (err: unknown, fallback: string): string =>
  (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || fallback

/**
 * Settings → RAG: re-embed all existing ChromaDB data with the active embedding model
 * (normally the selected trained variant), with progress and per-collection failures.
 */
export default function EmbeddingReindexCard({ refreshKey = 0 }: EmbeddingReindexCardProps) {
  const [status, setStatus] = useState<EmbeddingIndexStatus | null>(null)
  const [job, setJob] = useState<ReindexJobStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [showConfirm, setShowConfirm] = useState(false)
  const [force, setForce] = useState(false)
  const [starting, setStarting] = useState(false)
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const loadStatus = useCallback(async () => {
    try {
      const data = await embeddingIndexApi.getStatus()
      setStatus(data)
      setLoadError(null)
      if (data.reindex_job) setJob(data.reindex_job)
    } catch (err) {
      setLoadError(errorDetail(err, 'Failed to load ChromaDB index status'))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadStatus() }, [loadStatus, refreshKey])

  // Poll the job while it runs
  useEffect(() => {
    if (!job || !RUNNING.has(job.status)) return
    pollRef.current = setTimeout(async () => {
      try {
        const next = await embeddingIndexApi.getReindexStatus()
        setJob(next)
        if (!RUNNING.has(next.status)) {
          if (next.status === 'completed') {
            toast.success(`ChromaDB reindexed with ${next.target_model} (${next.reindexed_collections ?? 0} collection(s)).`)
          } else if (next.status === 'completed_with_errors') {
            toast.warning(`Reindex finished with ${next.failed_collections} failed collection(s). Their original vectors were kept.`)
          } else if (next.status === 'cancelled') {
            toast.info('Reindex cancelled. Collections not yet processed were left unchanged.')
          } else if (next.status === 'failed') {
            toast.error(next.error || 'Reindex failed')
          }
          loadStatus()
        }
      } catch {
        setJob(j => (j ? { ...j } : j)) // retry on next tick
      }
    }, 1500)
    return () => { if (pollRef.current) clearTimeout(pollRef.current) }
  }, [job, loadStatus])

  const isRunning = !!job && RUNNING.has(job.status)
  const info = status?.active_model_info
  const staleCollections = status?.stale_collections ?? 0
  const staleVectors = status?.stale_vectors ?? 0

  const openConfirm = () => {
    setForce(staleCollections === 0)
    setShowConfirm(true)
  }

  const handleStart = async () => {
    setStarting(true)
    try {
      const started = await embeddingIndexApi.startReindex(force)
      setJob(started)
      setShowConfirm(false)
      toast.info(`Reindexing ChromaDB with ${started.target_model}...`)
    } catch (err) {
      toast.error(errorDetail(err, 'Failed to start reindex'))
    } finally {
      setStarting(false)
    }
  }

  const handleCancel = async () => {
    try {
      await embeddingIndexApi.cancelReindex()
      setJob(j => (j ? { ...j, status: 'cancelling' } : j))
    } catch (err) {
      toast.error(errorDetail(err, 'Failed to cancel reindex'))
    }
  }

  const progress = Math.min(100, Math.max(0, job?.progress_percent ?? 0))

  return (
    <div style={{ borderRadius: 12, border: '1.5px solid hsl(265,70%,60%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', marginBottom: '.85rem' }}>
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Database size={16} style={{ color: 'hsl(265,70%,60%)' }} /> ChromaDB Embedding Index
          </h3>
          <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginTop: 4 }}>
            Stored vectors must come from the same model that embeds queries. After selecting a trained variant, re-embed existing data so retrieval, indexing and agenda mapping all use it.
          </div>
        </div>
        <button
          onClick={() => { setLoading(true); loadStatus() }}
          disabled={loading || isRunning}
          title="Refresh index status"
          style={{ background: 'transparent', border: '1px solid hsl(var(--border))', borderRadius: 6, padding: '4px 8px', cursor: 'pointer', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center' }}
        >
          <RefreshCw size={13} className={loading ? 'spin' : ''} />
        </button>
      </div>

      {loadError && (
        <div style={{ fontSize: '.78rem', color: 'hsl(0,80%,50%)', marginBottom: '.75rem' }}>{loadError}</div>
      )}

      {status && info && (
        <>
          {/* Target model + index summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '.7rem .85rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)' }}>
              <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>Active embedding model</div>
              <div style={{ fontSize: '.86rem', fontWeight: 700, color: 'hsl(var(--ink))', wordBreak: 'break-all' }}>{info.name}</div>
              <div style={{ fontSize: '.7rem', marginTop: 2, color: info.is_trained_variant ? 'hsl(150,65%,40%)' : 'hsl(var(--pencil))' }}>
                {info.is_trained_variant ? `Trained variant of ${info.base_model || 'base model'}` : 'Base model (no trained variant selected)'}
                {!info.available && <span style={{ color: 'hsl(0,80%,50%)' }}> · model files missing</span>}
              </div>
            </div>
            <div style={{ padding: '.7rem .85rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)' }}>
              <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>Indexed data</div>
              <div style={{ fontSize: '.86rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                {status.total_vectors.toLocaleString()} vectors in {status.total_collections} collections
              </div>
            </div>
            <div style={{
              padding: '.7rem .85rem', borderRadius: 8,
              background: staleCollections ? 'hsl(35,90%,50%/.1)' : 'hsl(150,65%,40%/.08)',
              border: `1px solid ${staleCollections ? 'hsl(35,90%,50%/.35)' : 'hsl(150,65%,40%/.3)'}`,
            }}>
              <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>Embedded with another model</div>
              <div style={{ fontSize: '.86rem', fontWeight: 700, color: staleCollections ? 'hsl(35,90%,40%)' : 'hsl(150,65%,35%)', display: 'flex', alignItems: 'center', gap: 6 }}>
                {staleCollections ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
                {staleCollections ? `${staleCollections} collections · ${staleVectors.toLocaleString()} vectors` : 'None — index is consistent'}
              </div>
            </div>
          </div>

          {staleCollections > 0 && !isRunning && (
            <div style={{ fontSize: '.75rem', color: 'hsl(var(--ink))', marginBottom: '.85rem', lineHeight: 1.45 }}>
              Stale collections are never queried with mismatched vectors: each one is re-embedded automatically the first time it is used,
              which can slow that request down. Reindexing now does it up front.
            </div>
          )}

          {status.model_switch_locked && !isRunning && (
            <div style={{ fontSize: '.75rem', color: 'hsl(35,90%,40%)', marginBottom: '.85rem' }}>
              {status.model_switch_locked}
            </div>
          )}

          {/* Job progress */}
          {job && job.status !== 'idle' && (
            <div style={{ marginBottom: '1rem', padding: '.85rem 1rem', borderRadius: 8, border: '1px solid hsl(var(--border)/.5)', background: 'hsl(var(--muted)/.25)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.78rem', fontWeight: 600, marginBottom: 6, color: 'hsl(var(--ink))' }}>
                <span>
                  {isRunning
                    ? (job.status === 'cancelling' ? 'Cancelling after the current collection…' : `Reindexing with ${job.target_model}`)
                    : `Last reindex: ${job.status.replace(/_/g, ' ')}${job.target_model ? ` (${job.target_model})` : ''}`}
                </span>
                <span style={{ color: 'hsl(265,70%,55%)' }}>{progress.toFixed(1)}%</span>
              </div>
              <div style={{ width: '100%', height: 8, background: 'hsl(var(--muted)/.6)', borderRadius: 10, overflow: 'hidden' }}>
                <div style={{
                  width: `${progress}%`, height: '100%', borderRadius: 10, transition: 'width .3s ease',
                  background: job.status === 'failed' ? 'hsl(0,80%,55%)' : job.status === 'completed' ? 'hsl(150,65%,45%)' : 'hsl(265,70%,60%)',
                }} />
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: 6 }}>
                <span>Collections: {job.processed_collections ?? 0}/{job.total_collections ?? 0}</span>
                <span>Vectors: {(job.processed_vectors ?? 0).toLocaleString()}/{(job.total_vectors ?? 0).toLocaleString()}</span>
                {!!job.failed_collections && <span style={{ color: 'hsl(0,80%,50%)' }}>Failed: {job.failed_collections}</span>}
                {job.current_collection && <span>Current: {job.current_collection}</span>}
              </div>
              {job.error && <div style={{ fontSize: '.75rem', color: 'hsl(0,80%,50%)', marginTop: 6 }}>{job.error}</div>}
              {!!job.failures?.length && (
                <div style={{ marginTop: 8, fontSize: '.72rem', color: 'hsl(0,75%,45%)', maxHeight: 120, overflowY: 'auto' }}>
                  {job.failures.map(f => (
                    <div key={f.collection} style={{ display: 'flex', gap: 6 }}>
                      <XCircle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
                      <span><strong>{f.collection}</strong> ({f.vectors} vectors, original kept): {f.error}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div style={{ display: 'flex', gap: '.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              onClick={openConfirm}
              disabled={isRunning || !info.available || !!status.model_switch_locked}
              style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '.55rem 1rem', borderRadius: 8, border: 'none',
                background: 'hsl(265,70%,58%)', color: 'white', fontSize: '.8rem', fontWeight: 700,
                cursor: isRunning || !info.available ? 'not-allowed' : 'pointer', opacity: isRunning || !info.available ? 0.6 : 1,
              }}
            >
              {isRunning ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
              Reindex ChromaDB with Trained Variant
            </button>
            {isRunning && job?.status === 'running' && (
              <button
                onClick={handleCancel}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '.5rem .9rem', borderRadius: 8, background: 'hsl(0,80%,50%/.1)', color: 'hsl(0,75%,50%)', border: '1px solid hsl(0,80%,50%/.3)', fontSize: '.78rem', fontWeight: 600, cursor: 'pointer' }}
              >
                <XCircle size={13} /> Cancel
              </button>
            )}
          </div>
        </>
      )}

      {showConfirm && status && info && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 9999, background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
          <div style={{ background: 'hsl(var(--card))', borderRadius: 14, border: '1.5px solid hsl(var(--border))', maxWidth: 520, width: '100%', padding: '1.5rem', boxShadow: '0 20px 50px rgba(0,0,0,0.3)', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
              Reindex ChromaDB with {info.name}?
            </h3>
            <div style={{ fontSize: '.82rem', lineHeight: 1.5, color: 'hsl(var(--ink))', background: 'hsl(var(--muted)/.4)', padding: '.9rem 1rem', borderRadius: 8, border: '1px solid hsl(var(--border)/.6)' }}>
              {!info.is_trained_variant && (
                <div style={{ color: 'hsl(35,90%,40%)', marginBottom: 8 }}>
                  <AlertTriangle size={13} style={{ verticalAlign: '-2px' }} /> The active model is a base model, not a trained variant. Select a variant in Embedding Training (or the RAG model selector) first if you meant to use one.
                </div>
              )}
              {staleCollections > 0
                ? <>{staleCollections} collection(s) with {staleVectors.toLocaleString()} vectors were embedded with a different model and will be re-embedded.</>
                : <>All collections already use this model.</>}
              <br /><br />
              Documents and metadata are not changed — only vectors are recomputed. Each collection is rebuilt in a temporary copy and swapped in only when complete, so a failure leaves it untouched.
              The embedding model cannot be changed while the reindex runs.
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '.8rem', color: 'hsl(var(--ink))', cursor: 'pointer' }}>
              <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} />
              Re-embed every collection (also those already using this model)
            </label>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '.6rem' }}>
              <button
                onClick={() => setShowConfirm(false)}
                disabled={starting}
                style={{ padding: '.5rem 1rem', borderRadius: 8, border: '1px solid hsl(var(--border))', background: 'transparent', color: 'hsl(var(--ink))', fontSize: '.8rem', cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button
                onClick={handleStart}
                disabled={starting || (staleCollections === 0 && !force)}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '.5rem 1rem', borderRadius: 8, border: 'none', background: 'hsl(265,70%,58%)', color: 'white', fontSize: '.8rem', fontWeight: 700, cursor: 'pointer', opacity: starting || (staleCollections === 0 && !force) ? 0.6 : 1 }}
              >
                {starting ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
                Start Reindex
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
