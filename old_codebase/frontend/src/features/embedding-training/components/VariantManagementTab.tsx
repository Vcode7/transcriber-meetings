import React, { useState, useEffect } from 'react';
import {
  Layers, CheckCircle2, Trash2, Cpu, Calendar,
  FileText, Copy, Check, AlertTriangle, RefreshCw,
  Zap, HardDrive, Plus, Sparkles
} from 'lucide-react';
import {
  EmbeddingVariant,
  embeddingTrainingApi
} from '../../../services/embeddingTrainingApi';

interface VariantManagementTabProps {
  onTrainNewClick?: () => void;
}

export default function VariantManagementTab({ onTrainNewClick }: VariantManagementTabProps) {
  const [variants, setVariants] = useState<EmbeddingVariant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const fetchVariants = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await embeddingTrainingApi.getVariants();
      setVariants(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to fetch variants.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVariants();
  }, []);

  const handleSelectVariant = async (variant: EmbeddingVariant) => {
    if (variant.is_active) return;
    setActionLoading(variant.id);
    try {
      await embeddingTrainingApi.selectVariant(variant.id);
      await fetchVariants();
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to activate variant.');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteVariant = async (variantId: string) => {
    setActionLoading(variantId);
    try {
      await embeddingTrainingApi.deleteVariant(variantId);
      setDeleteConfirmId(null);
      await fetchVariants();
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to delete variant.');
    } finally {
      setActionLoading(null);
    }
  };

  const copyPath = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
            Previously Trained Variants
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
            Manage domain-adapted embedding models and choose which variant powers the RAG retrieval pipeline.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={fetchVariants}
            disabled={loading}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: 'hsl(var(--card))', color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))', padding: '0.5rem 0.9rem',
              borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
            }}
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
            Refresh
          </button>

          {onTrainNewClick && (
            <button
              onClick={onTrainNewClick}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                background: 'hsl(var(--accent))', color: 'white',
                border: 'none', padding: '0.5rem 1rem',
                borderRadius: '8px', fontSize: '0.8rem', fontWeight: 700, cursor: 'pointer',
              }}
            >
              <Plus size={14} />
              Train New Variant
            </button>
          )}
        </div>
      </div>

      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: 'hsl(0 80% 50% / 0.1)', color: 'hsl(0 80% 65%)',
          border: '1px solid hsl(0 80% 50% / 0.25)', padding: '1rem',
          borderRadius: '12px', fontSize: '0.875rem',
        }}>
          <AlertTriangle size={18} />
          <div>{error}</div>
        </div>
      )}

      {loading ? (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '4rem', background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1px solid hsl(var(--border))', gap: '0.75rem',
        }}>
          <div className="spin" style={{
            width: 24, height: 24, border: '2px solid hsl(var(--accent) / 0.3)',
            borderTop: '2px solid hsl(var(--accent))', borderRadius: '50%',
          }} />
          <span style={{ fontSize: '0.9rem', color: 'hsl(var(--muted-foreground))' }}>
            Loading trained variants...
          </span>
        </div>
      ) : variants.length === 0 ? (
        <div style={{
          padding: '3.5rem 2rem', background: 'hsl(var(--card))', borderRadius: '16px',
          border: '1px dashed hsl(var(--border))', textAlign: 'center',
        }}>
          <Layers size={40} style={{ color: 'hsl(var(--muted-foreground))', margin: '0 auto 1rem', opacity: 0.5 }} />
          <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
            No Trained Variants Yet
          </h3>
          <p style={{ margin: '0 0 1.5rem', fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', maxWidth: '460px', marginLeft: 'auto', marginRight: 'auto' }}>
            You haven't trained any domain variants yet. Upload ADA technical documents to adapt the base embedding model.
          </p>
          {onTrainNewClick && (
            <button
              onClick={onTrainNewClick}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: '8px',
                background: 'hsl(var(--accent))', color: 'white', border: 'none',
                padding: '0.65rem 1.4rem', borderRadius: '8px', fontWeight: 700,
                fontSize: '0.875rem', cursor: 'pointer',
              }}
            >
              <Sparkles size={16} />
              Train Your First Variant
            </button>
          )}
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1.25rem' }}>
          {variants.map((v) => {
            const isDeleting = actionLoading === v.id;
            return (
              <div
                key={v.id}
                style={{
                  background: 'hsl(var(--card))',
                  border: v.is_active ? '2px solid hsl(150 75% 45%)' : '1px solid hsl(var(--border))',
                  borderRadius: '14px',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1rem',
                  position: 'relative',
                  boxShadow: v.is_active ? '0 0 20px hsl(150 75% 45% / 0.12)' : 'none',
                  transition: 'all 0.2s ease',
                }}
              >
                {/* Top header */}
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div>
                    <h3 style={{ margin: '0 0 4px', fontSize: '1.1rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                      {v.name}
                    </h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                      <Cpu size={13} />
                      Base: <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>{v.base_model}</span>
                    </div>
                  </div>

                  {v.is_active ? (
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: '5px',
                      background: 'hsl(150 75% 45% / 0.15)', color: 'hsl(150 75% 45%)',
                      border: '1px solid hsl(150 75% 45% / 0.3)', padding: '3px 10px',
                      borderRadius: '20px', fontSize: '0.72rem', fontWeight: 700,
                    }}>
                      <CheckCircle2 size={12} /> Active in RAG
                    </span>
                  ) : (
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: '4px',
                      background: 'hsl(var(--muted) / 0.5)', color: 'hsl(var(--muted-foreground))',
                      padding: '3px 8px', borderRadius: '6px', fontSize: '0.72rem', fontWeight: 600,
                    }}>
                      Completed
                    </span>
                  )}
                </div>

                {/* Metadata Grid */}
                <div style={{
                  display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.65rem',
                  background: 'hsl(var(--background))', borderRadius: '10px', padding: '0.75rem',
                  fontSize: '0.78rem',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(var(--muted-foreground))' }}>
                    <Calendar size={13} />
                    <span>{formatDate(v.created_at)}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(var(--muted-foreground))' }}>
                    <FileText size={13} />
                    <span>{v.document_count} Documents ({v.chunk_count} chunks)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(var(--muted-foreground))' }}>
                    <HardDrive size={13} />
                    <span>{v.size_mb} MB on disk</span>
                  </div>
                  {v.metrics?.final_loss !== undefined && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(150 75% 45%)', fontWeight: 600 }}>
                      <span>Loss: {v.metrics.final_loss}</span>
                    </div>
                  )}
                </div>

                {/* Model Path */}
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', marginBottom: '3px' }}>
                    Model Path:
                  </div>
                  <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    background: 'hsl(var(--muted) / 0.3)', border: '1px solid hsl(var(--border) / 0.6)',
                    borderRadius: '6px', padding: '4px 8px', fontSize: '0.72rem', fontFamily: 'monospace',
                    color: 'hsl(var(--foreground))',
                  }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '280px' }} title={v.model_path}>
                      {v.model_path}
                    </span>
                    <button
                      onClick={() => copyPath(v.model_path, v.id)}
                      style={{
                        background: 'transparent', border: 'none', cursor: 'pointer',
                        color: copiedId === v.id ? 'hsl(150 75% 45%)' : 'hsl(var(--muted-foreground))',
                        padding: '2px', display: 'flex', alignItems: 'center',
                      }}
                      title="Copy path"
                    >
                      {copiedId === v.id ? <Check size={13} /> : <Copy size={13} />}
                    </button>
                  </div>
                </div>

                {/* Bottom Actions */}
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  paddingTop: '0.5rem', borderTop: '1px solid hsl(var(--border) / 0.6)',
                }}>
                  {deleteConfirmId === v.id ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '0.75rem', color: 'hsl(0 80% 65%)', fontWeight: 600 }}>
                        Confirm delete?
                      </span>
                      <button
                        onClick={() => handleDeleteVariant(v.id)}
                        disabled={isDeleting}
                        style={{
                          background: 'hsl(0 80% 50%)', color: 'white', border: 'none',
                          padding: '3px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600,
                          cursor: 'pointer',
                        }}
                      >
                        Yes, Delete
                      </button>
                      <button
                        onClick={() => setDeleteConfirmId(null)}
                        style={{
                          background: 'transparent', border: '1px solid hsl(var(--border))',
                          padding: '3px 8px', borderRadius: '4px', fontSize: '0.75rem',
                          cursor: 'pointer', color: 'hsl(var(--foreground))',
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDeleteConfirmId(v.id)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '4px',
                        background: 'transparent', border: 'none', color: 'hsl(var(--muted-foreground))',
                        fontSize: '0.78rem', cursor: 'pointer', padding: '4px 6px', borderRadius: '4px',
                      }}
                    >
                      <Trash2 size={13} /> Delete
                    </button>
                  )}

                  <button
                    onClick={() => handleSelectVariant(v)}
                    disabled={v.is_active || actionLoading === v.id}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '6px',
                      background: v.is_active ? 'hsl(150 75% 45% / 0.12)' : 'hsl(var(--accent))',
                      color: v.is_active ? 'hsl(150 75% 45%)' : 'white',
                      border: v.is_active ? '1px solid hsl(150 75% 45% / 0.3)' : 'none',
                      padding: '0.45rem 1rem', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 700,
                      cursor: v.is_active ? 'default' : 'pointer',
                      boxShadow: v.is_active ? 'none' : '0 2px 8px hsl(var(--accent) / 0.25)',
                    }}
                  >
                    {actionLoading === v.id ? (
                      <div className="spin" style={{
                        width: 12, height: 12, border: '2px solid currentColor',
                        borderTop: '2px solid transparent', borderRadius: '50%',
                      }} />
                    ) : v.is_active ? (
                      <>
                        <Check size={14} /> Active
                      </>
                    ) : (
                      <>
                        <Zap size={14} /> Use / Select
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
