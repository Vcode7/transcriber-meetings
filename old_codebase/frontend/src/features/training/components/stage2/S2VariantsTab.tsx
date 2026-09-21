import { useState, useEffect } from 'react';
import { Star, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Minus, RefreshCw, BarChart3, Sparkles, AlertCircle, CheckCircle, Zap } from 'lucide-react';
import { useStage2Store } from '../../store/stage2Store';
import { listStage2Variants, activateStage2Variant } from '../../api/stage2Api';
import { STAGE2_METRIC_LABELS, STAGE2_METRIC_ORDER } from '../../types/stage2Types';
import type { Stage2Variant } from '../../types/stage2Types';
import VariantPromptViewer from '../VariantPromptViewer';


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
      <div style={{ width: '190px', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', flexShrink: 0 }}>
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

function VariantCard({
  variant,
  allVariants,
  onActivate,
  activating,
}: {
  variant: Stage2Variant;
  allVariants: Stage2Variant[];
  onActivate: (variantId: string) => Promise<void>;
  activating: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const overall = variant.scores?.overall ?? 0;
  const improvement = variant.improvement_over_baseline;
  const method = variant.method || (variant.is_default ? 'baseline' : 'dspy');

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
      border: `1.5px solid ${variant.is_active ? 'hsl(142 76% 36% / 0.6)' : variant.is_best ? 'hsl(45 93% 47%)' : variant.is_default ? 'hsl(var(--border))' : 'hsl(var(--accent) / 0.3)'}`,
      background: variant.is_active ? 'hsl(142 76% 36% / 0.03)' : variant.is_best ? 'hsl(45 93% 47% / 0.04)' : 'hsl(var(--card))',
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

            {/* Method Badge */}
            {method === 'lora' && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '3px',
                fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: '10px',
                background: 'hsl(262 80% 65% / 0.15)', color: 'hsl(262 80% 65%)',
                border: '1px solid hsl(262 80% 65% / 0.3)',
              }}>
                <Zap size={10} /> LoRA Adapter
              </span>
            )}
            {method === 'qlora' && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '3px',
                fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: '10px',
                background: 'hsl(180 80% 45% / 0.15)', color: 'hsl(180 80% 40%)',
                border: '1px solid hsl(180 80% 45% / 0.3)',
              }}>
                <Zap size={10} /> QLoRA (4-bit)
              </span>
            )}
            {method === 'dspy' && !variant.is_default && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: '3px',
                fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: '10px',
                background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                border: '1px solid hsl(var(--accent) / 0.25)',
              }}>
                <Sparkles size={10} /> DSPy Prompt
              </span>
            )}

            {/* Active Status Badge */}
            {variant.is_active && (
              <span style={{
                display: 'flex', alignItems: 'center', gap: '3px',
                fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: '10px',
                background: 'hsl(142 76% 36% / 0.15)', color: 'hsl(142 76% 36%)',
                border: '1px solid hsl(142 76% 36% / 0.3)',
              }}>
                <CheckCircle size={10} /> Active in Stage 2 Pipeline
              </span>
            )}

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
                background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
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
          </div>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '4px', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', flexWrap: 'wrap' }}>
            {!variant.is_default && (
              <>
                <span>Optimizer/Method: <strong>{variant.optimizer}</strong></span>
                <span>Model: {variant.model || 'N/A'}</span>
                <span>{variant.training_group_count} train / {variant.validation_group_count} val groups</span>
                <span>Created: {new Date(variant.created_at).toLocaleDateString()}</span>
              </>
            )}
            {variant.is_default && <span>The permanent baseline prompt — always available</span>}
          </div>
        </div>

        {/* Action button + Score + improvement */}
        <div style={{ display: 'flex', gap: '1.2rem', alignItems: 'center', flexShrink: 0 }}>
          {!variant.is_active ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onActivate(variant.variant_id);
              }}
              disabled={activating}
              style={{
                padding: '4px 10px', borderRadius: '6px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                color: 'hsl(var(--foreground))', fontSize: '0.72rem', fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Activate for Pipeline
            </button>
          ) : (
            <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'hsl(142 76% 36%)' }}>
              ✓ Active
            </span>
          )}

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
                {STAGE2_METRIC_ORDER.map((key) => (
                  <MetricRow
                    key={key}
                    label={STAGE2_METRIC_LABELS[key]}
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
                  ['Training Groups', variant.training_group_count],
                  ['Validation Groups', variant.validation_group_count],
                  ['Batch Size (Points/Group)', variant.config_snapshot?.points_per_group || '5'],
                  ['Context Retrieval', variant.config_snapshot?.context_retrieval ? 'Enabled' : 'Disabled'],
                  ['Feedback Rounds', variant.feedback_count ?? 0],
                  ['Artifact Stored', variant.artifact_path ? 'Yes (immutable)' : (variant.is_default ? 'No (Built-in)' : 'No')],
                  ['Created At', new Date(variant.created_at).toLocaleString()],
                ].map(([label, value]) => (
                  <div
                    key={String(label)}
                    style={{
                      display: 'flex', justifyContent: 'space-between',
                      padding: '3px 0', borderBottom: '1px solid hsl(var(--border) / 0.5)',
                      fontSize: '0.75rem',
                    }}
                  >
                    <span style={{ color: 'hsl(var(--muted-foreground))' }}>{label}</span>
                    <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>{String(value)}</span>
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

export default function S2VariantsTab() {
  const { variants, setVariants, variantsLoading, setVariantsLoading } = useStage2Store();
  const [refreshing, setRefreshing] = useState(false);
  const [activatingId, setActivatingId] = useState<string | null>(null);

  const loadVariants = async () => {
    setVariantsLoading(true);
    try {
      const res = await listStage2Variants();
      setVariants(res.variants || []);
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

  const handleActivate = async (variantId: string) => {
    setActivatingId(variantId);
    try {
      await activateStage2Variant(variantId);
      await loadVariants();
    } catch (e) {
      console.error('Failed to activate variant:', e);
    } finally {
      setActivatingId(null);
    }
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
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Stage 2 Variants</h3>
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
              {bestVariant.improvement_over_baseline !== null && (
                <span> ({bestVariant.improvement_over_baseline > 0 ? '+' : ''}{bestVariant.improvement_over_baseline}% vs baseline)</span>
              )}
            </div>
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
          <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>No Stage 2 Variants Found</div>
          <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px' }}>
            Run the training loop in the Training Loop tab to create your first DSPy or LoRA/QLoRA variant.
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {variants.map((v) => (
            <VariantCard
              key={v.variant_id}
              variant={v}
              allVariants={variants}
              onActivate={handleActivate}
              activating={activatingId === v.variant_id}
            />
          ))}
        </div>
      )}
    </div>
  );
}
