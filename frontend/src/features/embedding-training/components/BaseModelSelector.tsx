import React from 'react';
import { Cpu, CheckCircle2, Layers, HardDrive, Sparkles, ArrowRight } from 'lucide-react';
import { BaseModelInfo } from '../../../services/embeddingTrainingApi';

interface BaseModelSelectorProps {
  models: BaseModelInfo[];
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  onNext: () => void;
  loading?: boolean;
}

export default function BaseModelSelector({
  models,
  selectedModelId,
  onSelectModel,
  onNext,
  loading = false,
}: BaseModelSelectorProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
          Step 1: Select Base Embedding Model
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
          Select the base offline embedding model to adapt to the ADA technical domain. The original base model will remain untouched.
        </p>
      </div>

      {loading ? (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '3rem', background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1px solid hsl(var(--border))', gap: '0.75rem',
        }}>
          <div className="spin" style={{
            width: 20, height: 20, border: '2px solid hsl(var(--accent) / 0.3)',
            borderTop: '2px solid hsl(var(--accent))', borderRadius: '50%',
          }} />
          <span style={{ fontSize: '0.9rem', color: 'hsl(var(--muted-foreground))' }}>
            Discovering available local embedding models...
          </span>
        </div>
      ) : models.length === 0 ? (
        <div style={{
          padding: '2.5rem', background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1px dashed hsl(var(--border))', textAlign: 'center',
        }}>
          <Cpu size={36} style={{ color: 'hsl(var(--muted-foreground))', margin: '0 auto 0.75rem', opacity: 0.6 }} />
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 600 }}>No Base Embedding Models Found</h4>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>
            Please ensure Qwen3-Embedding-0.6B is placed in the runtime/embeddings folder.
          </p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
          {models.map((model) => {
            const isSelected = selectedModelId === model.id;
            return (
              <div
                key={model.id}
                onClick={() => onSelectModel(model.id)}
                style={{
                  background: isSelected
                    ? 'linear-gradient(145deg, hsl(var(--card)) 0%, hsl(var(--accent) / 0.08) 100%)'
                    : 'hsl(var(--card))',
                  border: isSelected ? '2px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                  borderRadius: '12px',
                  padding: '1.25rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  position: 'relative',
                  boxShadow: isSelected ? '0 0 20px hsl(var(--accent) / 0.15)' : 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{
                      width: 36, height: 36, borderRadius: '10px',
                      background: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--muted) / 0.5)',
                      color: isSelected ? 'white' : 'hsl(var(--foreground))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'all 0.2s ease',
                    }}>
                      <Cpu size={18} />
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'hsl(var(--foreground))' }}>
                        {model.name}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                        {model.architecture}
                      </div>
                    </div>
                  </div>

                  {isSelected && (
                    <span style={{
                      display: 'flex', alignItems: 'center', gap: '4px',
                      background: 'hsl(var(--accent) / 0.15)', color: 'hsl(var(--accent))',
                      padding: '2px 8px', borderRadius: '20px', fontSize: '0.7rem', fontWeight: 600,
                    }}>
                      <CheckCircle2 size={12} /> Selected
                    </span>
                  )}
                </div>

                {/* Specs badges */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '0.75rem' }}>
                  <span style={{
                    display: 'flex', alignItems: 'center', gap: '4px',
                    background: 'hsl(var(--muted) / 0.4)', padding: '2px 8px',
                    borderRadius: '6px', fontSize: '0.72rem', color: 'hsl(var(--foreground))',
                  }}>
                    <Layers size={11} /> {model.dimension} dimensions
                  </span>
                  <span style={{
                    display: 'flex', alignItems: 'center', gap: '4px',
                    background: 'hsl(var(--muted) / 0.4)', padding: '2px 8px',
                    borderRadius: '6px', fontSize: '0.72rem', color: 'hsl(var(--foreground))',
                  }}>
                    <HardDrive size={11} /> {model.size_mb} MB
                  </span>
                  {model.is_default && (
                    <span style={{
                      display: 'flex', alignItems: 'center', gap: '4px',
                      background: 'hsl(210 100% 50% / 0.15)', color: 'hsl(210 100% 60%)',
                      padding: '2px 8px', borderRadius: '6px', fontSize: '0.72rem', fontWeight: 600,
                    }}>
                      <Sparkles size={11} /> Default Base
                    </span>
                  )}
                </div>

                <div style={{
                  fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  fontFamily: 'monospace', opacity: 0.8,
                }} title={model.path}>
                  {model.path}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Navigation button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
        <button
          onClick={onNext}
          disabled={!selectedModelId}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: selectedModelId ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
            color: selectedModelId ? 'white' : 'hsl(var(--muted-foreground))',
            border: 'none', padding: '0.65rem 1.5rem', borderRadius: '8px',
            fontWeight: 600, fontSize: '0.875rem', cursor: selectedModelId ? 'pointer' : 'not-allowed',
            transition: 'all 0.15s ease',
          }}
        >
          Next: Upload Documents
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
