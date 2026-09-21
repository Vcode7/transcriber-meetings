import React, { useState } from 'react';
import {
  CheckCircle2, HardDrive, ShieldCheck, Sparkles,
  ArrowRight, Layers, Cpu, Zap, AlertTriangle
} from 'lucide-react';
import {
  TrainingJobStatus,
  ExtractionResult,
  embeddingTrainingApi
} from '../../../services/embeddingTrainingApi';

interface SaveVariantCardProps {
  jobStatus: TrainingJobStatus;
  extractionResult: ExtractionResult;
  onVariantSaved: (variantId: string, activated: boolean) => void;
}

export default function SaveVariantCard({
  jobStatus,
  extractionResult,
  onVariantSaved,
}: SaveVariantCardProps) {
  const [variantName, setVariantName] = useState(jobStatus.variant_name || 'ADA-Technical-v1');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async (activateImmediately: boolean) => {
    if (!variantName.trim()) {
      setError('Please provide a valid variant name.');
      return;
    }

    if (!jobStatus.staged_model_path) {
      setError('Staged model path not found. Please re-run training.');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const saved = await embeddingTrainingApi.saveVariant({
        variant_name: variantName.trim(),
        base_model_name: jobStatus.base_model_name,
        staged_model_path: jobStatus.staged_model_path,
        document_count: extractionResult.successful_documents,
        document_names: extractionResult.document_stats.map((d) => d.filename),
        chunk_count: extractionResult.total_chunks,
        metrics: {
          final_loss: jobStatus.final_loss,
          epochs: jobStatus.total_epochs,
          elapsed_seconds: jobStatus.elapsed_seconds,
        },
      });

      if (activateImmediately) {
        await embeddingTrainingApi.selectVariant(saved.id);
      }

      onVariantSaved(saved.id, activateImmediately);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to save variant.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', maxWidth: '780px', margin: '0 auto', width: '100%' }}>
      <div style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
        <div style={{
          width: 56, height: 56, borderRadius: '16px',
          background: 'hsl(150 75% 45% / 0.15)', color: 'hsl(150 75% 45%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 1rem', boxShadow: '0 0 25px hsl(150 75% 45% / 0.25)',
        }}>
          <CheckCircle2 size={32} />
        </div>
        <h2 style={{ fontSize: '1.35rem', fontWeight: 800, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
          Domain Adaptation Complete!
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
          Save the trained embedding weights as a persistent, isolated variant.
        </p>
      </div>

      {/* Structured Summary Card matching example specification */}
      <div style={{
        background: 'linear-gradient(145deg, hsl(var(--card)) 0%, hsl(var(--card) / 0.7) 100%)',
        borderRadius: '16px', border: '1px solid hsl(var(--border))',
        padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem',
        boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
      }}>
        <div style={{
          borderBottom: '1px solid hsl(var(--border))', paddingBottom: '1rem',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'hsl(var(--muted-foreground))' }}>
            Variant Specification
          </span>
          <span style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            background: 'hsl(150 75% 45% / 0.15)', color: 'hsl(150 75% 45%)',
            padding: '2px 10px', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 700,
          }}>
            Training Status: Completed
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.25rem' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}>
              Base Model:
            </div>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Cpu size={16} style={{ color: 'hsl(var(--accent))' }} />
              {jobStatus.base_model_name}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}>
              Variant Name:
            </div>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: 'hsl(var(--accent))' }}>
              {variantName}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}>
              Training Corpus:
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Layers size={15} style={{ color: 'hsl(var(--muted-foreground))' }} />
              {extractionResult.successful_documents} Documents ({extractionResult.total_chunks} Passages)
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}>
              Final Metric:
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'hsl(150 75% 45%)' }}>
              Loss: {jobStatus.final_loss ?? '0.0000'} ({jobStatus.total_epochs} epochs)
            </div>
          </div>
        </div>

        {/* Isolation assurance banner */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: '10px',
          background: 'hsl(var(--muted) / 0.4)', borderRadius: '10px',
          padding: '0.85rem 1rem', border: '1px solid hsl(var(--border) / 0.8)',
        }}>
          <ShieldCheck size={18} style={{ color: 'hsl(150 75% 45%)', flexShrink: 0, marginTop: '2px' }} />
          <div style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', lineHeight: 1.45 }}>
            <strong style={{ color: 'hsl(var(--foreground))' }}>Base Model Protected:</strong> Stored separately under{' '}
            <code style={{ background: 'hsl(var(--muted))', padding: '1px 5px', borderRadius: '4px' }}>
              embeddings/{variantName}
            </code>
            . The original base model is never overwritten and your raw ADA documents remain in the vector store for retrieval.
          </div>
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

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
        <button
          onClick={() => handleSave(false)}
          disabled={saving}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: 'hsl(var(--card))', color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))', padding: '0.75rem 1.4rem',
            borderRadius: '10px', fontWeight: 600, fontSize: '0.875rem',
            cursor: saving ? 'not-allowed' : 'pointer',
          }}
        >
          <HardDrive size={16} />
          Save Variant Only
        </button>

        <button
          onClick={() => handleSave(true)}
          disabled={saving}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: 'hsl(var(--accent))', color: 'white', border: 'none',
            padding: '0.75rem 1.6rem', borderRadius: '10px', fontWeight: 700,
            fontSize: '0.875rem', cursor: saving ? 'not-allowed' : 'pointer',
            boxShadow: '0 4px 16px hsl(var(--accent) / 0.3)',
          }}
        >
          {saving ? (
            <>
              <div className="spin" style={{
                width: 16, height: 16, border: '2px solid white',
                borderTop: '2px solid transparent', borderRadius: '50%',
              }} />
              Saving Variant...
            </>
          ) : (
            <>
              <Zap size={16} />
              Save & Activate for RAG
              <ArrowRight size={16} />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
