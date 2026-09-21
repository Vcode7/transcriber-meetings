import { useState, useEffect } from 'react';
import { Star, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Minus, RefreshCw, BarChart3 } from 'lucide-react';
import { useStage1Store } from '../../store/stage1Store';
import { listStage1Variants } from '../../api/stage1Api';
import type { Stage1Variant, Stage1EvalScores } from '../../types/stage1Types';
import { METRIC_LABELS } from '../../types/stage1Types';
import VariantPromptViewer from '../VariantPromptViewer';


const METRIC_ORDER: (keyof Stage1EvalScores)[] = [
  'overall',
  'json_validity', 'schema_validity', 'required_fields', 'action_items_empty',
  'transcript_coverage', 'speaker_preservation', 'date_preservation',
  'number_preservation', 'technical_term_preservation', 'factual_preservation',
  'duplicate_detection', 'hallucination_detection', 'missing_content',
];

function scoreColor(pct: number) {
  if (pct >= 80) return 'hsl(142 71% 45%)';
  if (pct >= 60) return 'hsl(45 93% 47%)';
  return 'hsl(0 75% 55%)';
}

function MetricRow({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value);
  const color = scoreColor(pct);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '3px 0' }}>
      <div style={{ width: '180px', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', flexShrink: 0 }}>
        {label}
      </div>
      <div style={{ flex: 1, height: '6px', borderRadius: '3px', background: 'hsl(var(--muted))', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.min(pct, 100)}%`, borderRadius: '3px',
          background: color, transition: 'width 0.5s ease',
        }} />
      </div>
      <div style={{ minWidth: '40px', textAlign: 'right', fontSize: '0.78rem', fontWeight: 700, color }}>{pct}%</div>
    </div>
  );
}

function VariantCard({ variant, allVariants }: { variant: Stage1Variant; allVariants: Stage1Variant[] }) {
  const [expanded, setExpanded] = useState(false);
  const overall = variant.scores?.overall ?? 0;
  const improvement = variant.improvement_over_baseline;

  // Resolve human-readable parent/initial base variant label & score
  let parentLabel = 'Pipeline Baseline (Default)';
  let parentScore: number | null = 0;
  if (variant.parent_variant_id && variant.parent_variant_id !== 'default') {
    const parent = allVariants.find((v) => v.variant_id === variant.parent_variant_id);
    if (parent) {
      parentLabel = parent.label;
      parentScore = parent.scores?.overall ?? null;
    } else {
      parentLabel = `Variant ${variant.parent_variant_id.slice(0, 8)}...`;
    }
  }

  return (
    <div style={{
      borderRadius: '12px',
      border: `1.5px solid ${variant.is_best ? 'hsl(45 93% 47%)' : variant.is_default ? 'hsl(var(--border))' : 'hsl(var(--accent) / 0.3)'}`,
      background: variant.is_best ? 'hsl(45 93% 47% / 0.04)' : 'hsl(var(--card))',
      overflow: 'hidden',
      transition: 'all 0.12s ease',
    }}>
      {/* Card header */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: '12px', padding: '1rem 1.15rem',
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Variant label + badges */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700, fontSize: '0.9rem', color: 'hsl(var(--foreground))' }}>
              {variant.label}
            </span>
            {variant.is_best && (
              <span style={{
                display: 'flex', alignItems: 'center', gap: '3px',
                fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '10px',
                background: 'hsl(45 93% 47% / 0.15)', color: 'hsl(45 93% 35%)',
              }}>
                <Star size={10} /> Best Variant
              </span>
            )}
            {!variant.is_default && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '4px',
                fontSize: '0.68rem', fontWeight: 600, padding: '2px 8px', borderRadius: '10px',
                background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                border: '1px solid hsl(var(--accent) / 0.25)',
              }}>
                🌱 Base: {parentLabel}
              </span>
            )}
            {variant.is_default && (
              <span style={{
                fontSize: '0.68rem', fontWeight: 600, padding: '2px 7px', borderRadius: '10px',
                background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
              }}>
                Baseline
              </span>
            )}
            {variant.status === 'error' && (
              <span style={{
                fontSize: '0.68rem', fontWeight: 600, padding: '2px 7px', borderRadius: '10px',
                background: 'hsl(0 75% 55% / 0.1)', color: 'hsl(0 75% 50%)',
              }}>
                Error
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '4px', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', flexWrap: 'wrap' }}>
            {!variant.is_default && (
              <>
                <span>Optimizer: <strong>{variant.optimizer}</strong></span>
                <span>Model: {variant.model || 'N/A'}</span>
                <span>{variant.training_window_count} train / {variant.validation_window_count} val windows</span>
                <span>Created: {new Date(variant.created_at).toLocaleDateString()}</span>
              </>
            )}
            {variant.is_default && <span>The permanent baseline — always available</span>}
          </div>
        </div>

        {/* Score + improvement */}
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexShrink: 0 }}>
          {improvement !== null && improvement !== undefined && !variant.is_default && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))' }}>IMPROVEMENT</div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '3px', justifyContent: 'center',
                fontWeight: 700, fontSize: '0.9rem',
                color: improvement > 0 ? 'hsl(142 71% 45%)' : improvement < 0 ? 'hsl(0 75% 55%)' : 'hsl(var(--muted-foreground))',
              }}>
                {improvement > 0 ? <TrendingUp size={14} /> : improvement < 0 ? <TrendingDown size={14} /> : <Minus size={14} />}
                {improvement > 0 ? '+' : ''}{improvement}%
              </div>
            </div>
          )}

          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))' }}>OVERALL</div>
            <div style={{ fontWeight: 800, fontSize: '1.4rem', color: scoreColor(overall) }}>
              {variant.is_default ? '—' : `${Math.round(overall)}%`}
            </div>
          </div>

          <div style={{ color: 'hsl(var(--muted-foreground))' }}>
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </div>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div style={{
          borderTop: '1px solid hsl(var(--border))',
          padding: '1.25rem',
          background: 'hsl(var(--background))',
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
            {/* Metrics */}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Evaluation Metrics
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {METRIC_ORDER.map((key) => (
                  <MetricRow
                    key={key}
                    label={METRIC_LABELS[key]}
                    value={variant.scores?.[key] ?? 0}
                  />
                ))}
              </div>
            </div>

            {/* Config snapshot */}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Lineage &amp; Configuration
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[
                  ['Variant ID', variant.is_default ? 'default (baseline)' : (variant.variant_id?.slice(0, 16) + '...')],
                  ['Initial Base Variant', parentLabel + (parentScore !== null ? ` (${Math.round(parentScore)}% baseline score)` : '')],
                  ['Improvement vs Base', improvement !== null ? (improvement > 0 ? `+${improvement}%` : `${improvement}%`) : '—'],
                  ['Optimizer', variant.optimizer],
                  ['Model', variant.model || 'pipeline default'],
                  ['Training Windows', String(variant.training_window_count)],
                  ['Validation Windows', String(variant.validation_window_count)],
                  ['Feedback Rounds', String(variant.feedback_count)],
                  ['Created', new Date(variant.created_at).toLocaleString()],
                ].map(([label, value]) => (
                  <div key={label} style={{ display: 'flex', gap: '8px', fontSize: '0.78rem' }}>
                    <span style={{ color: 'hsl(var(--muted-foreground))', minWidth: '140px' }}>{label}</span>
                    <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Exact Prompt & Instructions used by this variant */}
          <VariantPromptViewer
            prompt={variant.instructions}
            variantLabel={variant.label}
            isDefault={variant.is_default}
          />

        </div>
      )}

    </div>
  );
}

export default function S1VariantsTab() {
  const { variants, setVariants, variantsLoading, setVariantsLoading } = useStage1Store();
  const [refreshing, setRefreshing] = useState(false);

  const loadVariants = async () => {
    setVariantsLoading(true);
    try {
      const v = await listStage1Variants();
      setVariants(v);
    } catch (e) {
      console.error(e);
    } finally {
      setVariantsLoading(false);
    }
  };

  useEffect(() => {
    loadVariants();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadVariants();
    setRefreshing(false);
  };

  const bestVariant = variants.find((v) => v.is_best);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: 28, height: 28, borderRadius: '8px',
            background: 'hsl(var(--accent) / 0.15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <BarChart3 size={14} color="hsl(var(--accent))" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Stage 1 Variants</h3>
          <span style={{
            fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
            background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
          }}>
            {variants.length} total
          </span>
        </div>
        <button
          onClick={handleRefresh}
          disabled={variantsLoading || refreshing}
          style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '5px 12px', borderRadius: '8px', border: '1px solid hsl(var(--border))',
            background: 'transparent', cursor: 'pointer', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))',
          }}
        >
          <RefreshCw size={12} style={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      {/* Best variant callout */}
      {bestVariant && (
        <div style={{
          padding: '0.8rem 1rem', borderRadius: '10px',
          background: 'hsl(45 93% 47% / 0.08)', border: '1px solid hsl(45 93% 47% / 0.3)',
          display: 'flex', alignItems: 'center', gap: '10px',
        }}>
          <Star size={16} color="hsl(45 93% 47%)" />
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'hsl(45 93% 35%)' }}>
              Best Variant: {bestVariant.label}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
              Overall score: {Math.round(bestVariant.scores?.overall ?? 0)}%
              {bestVariant.improvement_over_baseline != null && ` (+${bestVariant.improvement_over_baseline}% over baseline)`}
            </div>
          </div>
          <div style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
            Note: activating a variant must be done manually — a higher score does not auto-activate.
          </div>
        </div>
      )}

      {variantsLoading && variants.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'hsl(var(--muted-foreground))' }}>
          <RefreshCw size={24} style={{ animation: 'spin 1s linear infinite', marginBottom: '8px' }} />
          <div>Loading variants...</div>
        </div>
      ) : variants.length === 0 ? (
        <div style={{
          padding: '3rem', textAlign: 'center', borderRadius: '12px',
          border: '1px dashed hsl(var(--border))', color: 'hsl(var(--muted-foreground))',
          background: 'hsl(var(--card))',
        }}>
          <BarChart3 size={32} style={{ marginBottom: '12px', opacity: 0.4 }} />
          <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>No Stage 1 Variants Found</div>
          <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px' }}>
            Run the training loop in the Training Loop tab to create your first DSPy variant.
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {variants.map((v) => (
            <VariantCard key={v.variant_id} variant={v} allVariants={variants} />
          ))}
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
