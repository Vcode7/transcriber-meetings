import { useState } from 'react';
import { BarChart3, Star, Code2, ChevronDown, ChevronRight } from 'lucide-react';
import { useStage3Store } from '../../store/stage3Store';
import type { Stage3Variant } from '../../types/training';
import VariantPromptViewer from '../VariantPromptViewer';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  th: {
    padding: '0.65rem 0.85rem',
    textAlign: 'left' as const,
    fontSize: '0.72rem',
    fontWeight: 700,
    color: 'hsl(var(--muted-foreground))',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
    borderBottom: '1px solid hsl(var(--border))',
  },
  td: {
    padding: '0.75rem 0.85rem',
    fontSize: '0.8rem',
    borderBottom: '1px solid hsl(var(--border) / 0.5)',
    verticalAlign: 'middle' as const,
  },
};

function formatScore(val?: number): string {
  if (val === undefined || val === null) return '0%';
  const num = val <= 1.0 ? val * 100 : val;
  return `${Math.round(num)}%`;
}

function scoreColor(pct: number) {
  if (pct >= 80) return 'hsl(142 71% 45%)';
  if (pct >= 60) return 'hsl(45 93% 47%)';
  return 'hsl(0 75% 55%)';
}

export default function S3VariantsTab() {
  const { variants, valVariantId, setValVariantId } = useStage3Store();
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: 28, height: 28, borderRadius: '8px',
          background: 'hsl(var(--accent) / .15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <BarChart3 size={14} color="hsl(var(--accent))" />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Stage 3 DSPy Variants &amp; Evaluation</h3>
          <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
            Click any variant to view its metric details and exact prompt for side-by-side comparison.
          </p>
        </div>
      </div>

      <div style={{ ...S.card, overflowX: 'auto', padding: 0 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'hsl(var(--muted) / 0.3)' }}>
              <th style={{ ...S.th, width: '30px' }} />
              <th style={S.th}>Variant</th>
              <th style={S.th}>Initial Base</th>
              <th style={S.th}>Optimizer</th>
              <th style={S.th}>Feedback Used</th>
              <th style={S.th}>Accuracy</th>
              <th style={S.th}>Cand. Match</th>
              <th style={S.th}>Schema</th>
              <th style={S.th}>Overall Score</th>
              <th style={S.th}>Improvement</th>
              <th style={{ ...S.th, textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {variants.map((v) => {
              const isSelected = valVariantId === v.variant_id;
              const isExpanded = expandedIds.has(v.variant_id);
              const scores = v.scores || { overall: 0, accuracy: 0, candidate_validity: 0, schema_validity: 0 };
              const overallNum = (scores.overall || 0) <= 1.0 ? (scores.overall || 0) * 100 : (scores.overall || 0);
              const improvement = v.improvement_over_baseline;

              const parentVariant = v.parent_variant_id === 'default'
                ? { label: 'Default Baseline' }
                : variants.find((p) => p.variant_id === v.parent_variant_id);
              const parentLabel = v.is_default ? '—' : (parentVariant?.label || 'Baseline');

              return (
                <React.Fragment key={v.variant_id}>
                  <tr
                    onClick={() => toggleExpand(v.variant_id)}
                    style={{
                      background: isSelected
                        ? 'hsl(var(--accent) / 0.08)'
                        : isExpanded
                        ? 'hsl(var(--muted) / 0.15)'
                        : 'transparent',
                      cursor: 'pointer',
                      transition: 'background 0.15s ease',
                    }}
                  >
                    <td style={{ ...S.td, textAlign: 'center', color: 'hsl(var(--muted-foreground))' }}>
                      {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                    </td>
                    <td style={S.td}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        {v.is_best && <Star size={12} fill="hsl(38 92% 50%)" color="hsl(38 92% 50%)" />}
                        <span style={{ fontWeight: 700, color: 'hsl(var(--foreground))' }}>{v.label}</span>
                        {v.is_default && (
                          <span style={{
                            fontSize: '0.65rem', padding: '1px 5px', borderRadius: '4px',
                            background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
                          }}>
                            Baseline
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={S.td}>
                      {v.is_default ? (
                        <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.75rem' }}>— (Baseline)</span>
                      ) : (
                        <span style={{
                          fontSize: '0.72rem', fontWeight: 600, padding: '2px 7px', borderRadius: '6px',
                          background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
                          border: '1px solid hsl(var(--accent) / 0.25)',
                          display: 'inline-flex', alignItems: 'center', gap: '3px',
                        }}>
                          🌱 {parentLabel}
                        </span>
                      )}
                    </td>
                    <td style={S.td}>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{v.optimizer}</span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 600 }}>{v.feedback_count || 0}</span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 700, color: 'hsl(140 70% 45%)' }}>
                        {formatScore(scores.accuracy)}
                      </span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 700, color: 'hsl(210 80% 55%)' }}>
                        {formatScore(scores.candidate_validity)}
                      </span>
                    </td>
                    <td style={S.td}>
                      <span>{formatScore(scores.schema_validity)}</span>
                    </td>
                    <td style={S.td}>
                      <span style={{
                        fontWeight: 800, fontSize: '0.85rem',
                        color: scoreColor(overallNum),
                      }}>
                        {formatScore(scores.overall)}
                      </span>
                    </td>
                    <td style={S.td}>
                      {improvement !== null && improvement !== undefined && !v.is_default ? (
                        <span style={{
                          fontSize: '0.75rem', fontWeight: 700,
                          color: improvement > 0 ? 'hsl(140 70% 45%)' : improvement < 0 ? 'hsl(0 75% 55%)' : 'hsl(var(--muted-foreground))',
                        }}>
                          {improvement > 0 ? `+${improvement}%` : `${improvement}%`}
                        </span>
                      ) : (
                        <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.75rem' }}>—</span>
                      )}
                    </td>
                    <td style={{ ...S.td, textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setValVariantId(v.variant_id);
                          }}
                          style={{
                            padding: '3px 9px', borderRadius: '6px', border: 'none',
                            background: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                            color: isSelected ? 'white' : 'hsl(var(--foreground))',
                            cursor: 'pointer', fontSize: '0.72rem', fontWeight: 600,
                          }}
                        >
                          {isSelected ? 'Active' : 'Select'}
                        </button>
                      </div>
                    </td>
                  </tr>

                  {/* Expanded detail row showing prompt alongside metrics */}
                  {isExpanded && (
                    <tr>
                      <td colSpan={11} style={{ padding: '1.25rem', background: 'hsl(var(--background))', borderBottom: '1px solid hsl(var(--border))' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.5rem' }}>
                          {/* Metrics breakdown */}
                          <div>
                            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                              Evaluation Metrics Breakdown
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                              {[
                                { label: 'Overall Quality Score', val: scores.overall },
                                { label: 'Agenda Assignment Accuracy', val: scores.accuracy },
                                { label: 'Candidate Match Validity', val: scores.candidate_validity },
                                { label: 'Schema Validity', val: scores.schema_validity },
                              ].map(({ label, val }) => {
                                const pct = Math.round((val <= 1.0 ? val * 100 : val) || 0);
                                const col = scoreColor(pct);
                                return (
                                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '2px 0' }}>
                                    <div style={{ width: '190px', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', flexShrink: 0 }}>
                                      {label}
                                    </div>
                                    <div style={{ flex: 1, height: '6px', borderRadius: '3px', background: 'hsl(var(--muted))', overflow: 'hidden' }}>
                                      <div style={{
                                        height: '100%', width: `${Math.min(pct, 100)}%`, borderRadius: '3px',
                                        background: col, transition: 'width 0.5s ease',
                                      }} />
                                    </div>
                                    <div style={{ minWidth: '40px', textAlign: 'right', fontSize: '0.78rem', fontWeight: 700, color: col }}>
                                      {pct}%
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* Lineage & Config */}
                          <div>
                            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                              Lineage &amp; Configuration
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                              {[
                                ['Variant ID', v.is_default ? 'default (baseline)' : v.variant_id],
                                ['Initial Base Variant', parentLabel],
                                ['Improvement vs Base', improvement !== null && improvement !== undefined ? (improvement > 0 ? `+${improvement}%` : `${improvement}%`) : '—'],
                                ['Optimizer', v.optimizer || '—'],
                                ['Model', v.model || 'local default'],
                                ['Training Batches', String(v.training_batch_count ?? 0)],
                                ['Validation Batches', String(v.validation_batch_count ?? 0)],
                                ['Feedback Count', String(v.feedback_count ?? 0)],
                                ['Created', v.created_at ? new Date(v.created_at).toLocaleString() : '—'],
                              ].map(([label, value]) => (
                                <div key={label} style={{ display: 'flex', gap: '8px', fontSize: '0.78rem' }}>
                                  <span style={{ color: 'hsl(var(--muted-foreground))', minWidth: '140px' }}>{label}</span>
                                  <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>{value}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Exact Variant Prompt & Instructions */}
                        <VariantPromptViewer
                          prompt={v.instructions}
                          variantLabel={v.label}
                          isDefault={v.is_default}
                        />

                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

