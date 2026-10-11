import React, { useState, useEffect } from 'react';
import {
  FlaskConical, CheckCircle2, AlertCircle, Loader, RotateCcw,
  Maximize2, Sparkles, Filter, CheckSquare, Square, Search,
  Sliders, FileText, ChevronRight, Check
} from 'lucide-react';
import { listVariants, listTrainingData, rerunVariantTest } from '../api/romTrainingApi';
import type { RomVariant, TrainingDataEntry, TestResult } from '../types/romTrainingTypes';
import TestComparisonModal from './TestComparisonModal';

export default function TestTab() {
  const [variants, setVariants] = useState<RomVariant[]>([]);
  const [loadingVariants, setLoadingVariants] = useState(true);
  const [selectedVariantId, setSelectedVariantId] = useState<string | null>(null);

  const [dataset, setDataset] = useState<TrainingDataEntry[]>([]);
  const [loadingDataset, setLoadingDataset] = useState(true);

  // Selection mode: 'random' or 'specific'
  const [sampleMode, setSampleMode] = useState<'random' | 'specific'>('random');
  const [randomCount, setRandomCount] = useState<number>(3);
  const [selectedSampleIds, setSelectedSampleIds] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  // Execution options & state
  const [disableThinking, setDisableThinking] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Evaluation results
  const [testResults, setTestResults] = useState<TestResult[] | null>(null);
  const [accuracy, setAccuracy] = useState<string | null>(null);

  // Full comparison modal state
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    results: TestResult[];
    initialIndex: number;
    variantName?: string;
    accuracy?: string | null;
  }>({
    isOpen: false,
    results: [],
    initialIndex: 0,
    variantName: '',
    accuracy: null,
  });

  // Load variants and training dataset on mount
  useEffect(() => {
    setLoadingVariants(true);
    listVariants()
      .then(res => {
        const list = res.variants || [];
        setVariants(list);
        // Preselect first completed variant with checkpoint
        const readyVariant = list.find((v: RomVariant) => v.status === 'done' && v.has_checkpoint);
        if (readyVariant) {
          setSelectedVariantId(readyVariant.id);
          // If variant already has previous test results, display them as starting preview
          if (readyVariant.test_results?.length) {
            setTestResults(readyVariant.test_results);
            setAccuracy(readyVariant.accuracy || null);
          }
        } else if (list.length > 0) {
          setSelectedVariantId(list[0].id);
        }
      })
      .catch(() => {})
      .finally(() => setLoadingVariants(false));

    setLoadingDataset(true);
    listTrainingData()
      .then(res => {
        const data = res.data || [];
        setDataset(data);
        if (data.length > 0) {
          setRandomCount(Math.min(3, data.length));
          setSelectedSampleIds(data.slice(0, Math.min(3, data.length)).map((d: TrainingDataEntry) => d.id));
        }
      })
      .catch(() => {})
      .finally(() => setLoadingDataset(false));
  }, []);

  const selectedVariant = variants.find(v => v.id === selectedVariantId);
  const readyVariants = variants.filter(v => v.status === 'done' && v.has_checkpoint);

  // Filtered dataset for specific mode search
  const filteredDataset = dataset.filter(item => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      (item.agenda_title && item.agenda_title.toLowerCase().includes(q)) ||
      (item.recording_title && item.recording_title.toLowerCase().includes(q))
    );
  });

  // Toggle specific sample checkbox
  const toggleSample = (id: string) => {
    setSelectedSampleIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  // Select all / Deselect all
  const selectAllFiltered = () => {
    const allFilteredIds = filteredDataset.map(d => d.id);
    setSelectedSampleIds(Array.from(new Set([...selectedSampleIds, ...allFilteredIds])));
  };

  const deselectAllFiltered = () => {
    const filteredIdSet = new Set(filteredDataset.map(d => d.id));
    setSelectedSampleIds(prev => prev.filter(id => !filteredIdSet.has(id)));
  };

  // When variant changes, update results preview if available
  const handleVariantChange = (id: string) => {
    setSelectedVariantId(id);
    const chosen = variants.find(v => v.id === id);
    if (chosen?.test_results?.length) {
      setTestResults(chosen.test_results);
      setAccuracy(chosen.accuracy || null);
    } else {
      setTestResults(null);
      setAccuracy(null);
    }
  };

  // Run Test Evaluation
  const handleRunEvaluation = async () => {
    if (!selectedVariantId) {
      setError('Please select a trained variant to test.');
      return;
    }
    if (sampleMode === 'specific' && selectedSampleIds.length === 0) {
      setError('Please select at least one specific sample to evaluate.');
      return;
    }
    if (sampleMode === 'random' && randomCount <= 0) {
      setError('Please specify a valid sample count for random testing.');
      return;
    }

    setRunning(true);
    setError(null);

    try {
      const res = await rerunVariantTest(
        selectedVariantId,
        sampleMode === 'random' ? randomCount : undefined,
        disableThinking,
        sampleMode === 'specific' ? selectedSampleIds : undefined
      );

      if (res && res.test_results) {
        setTestResults(res.test_results);
        setAccuracy(res.accuracy || null);

        // Update local variant record in state
        setVariants(prev => prev.map(v => v.id === selectedVariantId ? {
          ...v,
          test_results: res.test_results,
          accuracy: res.accuracy,
          has_checkpoint: true,
          status: 'done',
        } : v));
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Failed to run test evaluation.';
      setError(msg);
    } finally {
      setRunning(false);
    }
  };

  const effectiveSampleCount = sampleMode === 'random' ? randomCount : selectedSampleIds.length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 1200, margin: '0 auto' }}>
      {/* Tab Header Banner */}
      <div style={{
        background: 'linear-gradient(135deg, hsl(280 80% 60% / .08) 0%, hsl(var(--accent) / .05) 100%)',
        border: '1.5px solid hsl(280 80% 60% / .2)',
        borderRadius: 12,
        padding: '1.2rem 1.4rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10,
            background: 'hsl(280 80% 60% / .18)',
            color: 'hsl(280 80% 55%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <FlaskConical size={20} />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800 }}>Model Test & Comparison</h2>
            <p style={{ margin: 0, fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))', marginTop: 2 }}>
              Evaluate fine-tuned LoRA variants against the Base model and real human MoM ground-truth.
            </p>
          </div>
        </div>

        {readyVariants.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', fontWeight: 600, color: 'hsl(142 70% 40%)' }}>
            <CheckCircle2 size={14} />
            <span>{readyVariants.length} Trained Variant{readyVariants.length !== 1 ? 's' : ''} Ready</span>
          </div>
        )}
      </div>

      {/* Error alert if any */}
      {error && (
        <div style={{
          padding: '0.75rem 1rem', borderRadius: 8,
          background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .3)',
          color: 'hsl(var(--destructive))', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Top Grid: Variant Selection & Sample Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1fr) minmax(360px, 1.25fr)', gap: '1.25rem' }}>
        {/* Step 1: Select Trained Variant */}
        <div style={{
          background: 'hsl(var(--card))', borderRadius: 12, border: '1px solid hsl(var(--border))',
          padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '1rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '0.9rem' }}>
              <span style={{
                width: 22, height: 22, borderRadius: '50%', background: 'hsl(var(--accent) / .15)',
                color: 'hsl(var(--accent))', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '0.75rem', fontWeight: 800
              }}>1</span>
              <span>Select Model Variant</span>
            </div>
            {selectedVariant && (
              <span style={{
                fontSize: '0.68rem', padding: '2px 8px', borderRadius: 6,
                background: selectedVariant.has_checkpoint ? 'hsl(142 70% 45% / .15)' : 'hsl(var(--destructive) / .15)',
                color: selectedVariant.has_checkpoint ? 'hsl(142 70% 35%)' : 'hsl(var(--destructive))',
                fontWeight: 700,
              }}>
                {selectedVariant.has_checkpoint ? 'CHECKPOINT LOADABLE' : 'NO CHECKPOINT'}
              </span>
            )}
          </div>

          {loadingVariants ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem', padding: '1rem' }}>
              <Loader size={14} className="spin" /> Loading variants…
            </div>
          ) : variants.length === 0 ? (
            <div style={{ padding: '1.5rem', textAlign: 'center', background: 'hsl(var(--muted) / .2)', borderRadius: 8, color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem' }}>
              No trained variants found yet. Train a variant in the Training tab first.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <select
                value={selectedVariantId || ''}
                onChange={e => handleVariantChange(e.target.value)}
                style={{
                  width: '100%', padding: '0.6rem 0.85rem', borderRadius: 8,
                  border: '1.5px solid hsl(var(--border))', background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))', fontSize: '0.85rem', fontWeight: 600,
                  fontFamily: 'Inter, sans-serif', cursor: 'pointer',
                }}
              >
                {variants.map(v => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.base_model?.replace(/\\/g, '/').split('/').filter(Boolean).pop() || 'LoRA'}) {v.has_checkpoint ? '✓' : '(missing weights)'}
                  </option>
                ))}
              </select>

              {/* Variant Details Card */}
              {selectedVariant && (
                <div style={{
                  padding: '0.85rem', borderRadius: 8,
                  background: 'hsl(var(--muted) / .25)', border: '1px solid hsl(var(--border) / .7)',
                  fontSize: '0.74rem', display: 'flex', flexDirection: 'column', gap: 5,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'hsl(var(--muted-foreground))' }}>
                    <span>Base Model:</span>
                    <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                      {selectedVariant.base_model?.replace(/\\/g, '/').split('/').filter(Boolean).pop()}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'hsl(var(--muted-foreground))' }}>
                    <span>Training Type:</span>
                    <span style={{ fontWeight: 600, textTransform: 'capitalize', color: 'hsl(var(--foreground))' }}>
                      {selectedVariant.training_type}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: 'hsl(var(--muted-foreground))' }}>
                    <span>Created:</span>
                    <span style={{ color: 'hsl(var(--foreground))' }}>
                      {selectedVariant.created_at ? new Date(selectedVariant.created_at).toLocaleDateString() : 'N/A'}
                    </span>
                  </div>
                  {selectedVariant.accuracy && (
                    <div style={{
                      marginTop: 4, paddingTop: 6, borderTop: '1px dashed hsl(var(--border))',
                      fontSize: '0.72rem', color: 'hsl(142 70% 35%)', fontWeight: 600,
                    }}>
                      Current Score: {selectedVariant.accuracy}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Test Option: Thinking suppression toggle */}
          <div style={{
            marginTop: 'auto', paddingTop: '0.85rem', borderTop: '1px solid hsl(var(--border) / .6)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
          }}>
            <div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700 }}>Suppress Thinking Tokens</div>
              <div style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))' }}>
                Skips deep thought tags to output direct structured points faster
              </div>
            </div>
            <input
              type="checkbox"
              checked={disableThinking}
              onChange={e => setDisableThinking(e.target.checked)}
              style={{ width: 16, height: 16, cursor: 'pointer', accentColor: 'hsl(var(--accent))' }}
            />
          </div>
        </div>

        {/* Step 2: Sample Selection Mode */}
        <div style={{
          background: 'hsl(var(--card))', borderRadius: 12, border: '1px solid hsl(var(--border))',
          padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.9rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '0.9rem' }}>
              <span style={{
                width: 22, height: 22, borderRadius: '50%', background: 'hsl(var(--accent) / .15)',
                color: 'hsl(var(--accent))', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '0.75rem', fontWeight: 800
              }}>2</span>
              <span>Dataset Sample Selection</span>
            </div>
            <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
              {dataset.length} total dataset pairs
            </span>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              onClick={() => setSampleMode('random')}
              style={{
                flex: 1, padding: '0.5rem 0.75rem', borderRadius: 8,
                border: sampleMode === 'random' ? '2px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                background: sampleMode === 'random' ? 'hsl(var(--accent) / .12)' : 'hsl(var(--muted) / .2)',
                color: sampleMode === 'random' ? 'hsl(var(--accent))' : 'hsl(var(--foreground))',
                fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}
            >
              <span>🎲 Random Sampling</span>
            </button>
            <button
              type="button"
              onClick={() => setSampleMode('specific')}
              style={{
                flex: 1, padding: '0.5rem 0.75rem', borderRadius: 8,
                border: sampleMode === 'specific' ? '2px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                background: sampleMode === 'specific' ? 'hsl(var(--accent) / .12)' : 'hsl(var(--muted) / .2)',
                color: sampleMode === 'specific' ? 'hsl(var(--accent))' : 'hsl(var(--foreground))',
                fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}
            >
              <span>🎯 Specific Samples</span>
            </button>
          </div>

          {/* Mode View: Random */}
          {sampleMode === 'random' && (
            <div style={{
              padding: '1rem', borderRadius: 8, background: 'hsl(var(--muted) / .2)',
              border: '1px solid hsl(var(--border) / .6)', display: 'flex', flexDirection: 'column', gap: '0.75rem',
            }}>
              <div style={{ fontSize: '0.78rem', fontWeight: 600 }}>Choose number of random samples:</div>
              <div style={{ display: 'flex', gap: 8 }}>
                {[1, 3, 5, 10].map(count => (
                  <button
                    key={count}
                    type="button"
                    disabled={count > dataset.length && dataset.length > 0}
                    onClick={() => setRandomCount(Math.min(count, dataset.length || count))}
                    style={{
                      flex: 1, padding: '0.45rem', borderRadius: 6,
                      border: randomCount === count ? '1.5px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                      background: randomCount === count ? 'hsl(var(--accent))' : 'hsl(var(--background))',
                      color: randomCount === count ? 'white' : 'hsl(var(--foreground))',
                      fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer',
                    }}
                  >
                    {count} Sample{count > 1 ? 's' : ''}
                  </button>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>Custom count:</span>
                <input
                  type="number"
                  min={1}
                  max={Math.max(1, dataset.length)}
                  value={randomCount}
                  onChange={e => setRandomCount(Math.max(1, Math.min(dataset.length || 1, parseInt(e.target.value) || 1)))}
                  style={{
                    width: 70, padding: '0.3rem 0.5rem', borderRadius: 6,
                    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                    color: 'hsl(var(--foreground))', fontSize: '0.78rem', fontWeight: 700,
                  }}
                />
                <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                  (max: {dataset.length})
                </span>
              </div>
            </div>
          )}

          {/* Mode View: Specific Samples */}
          {sampleMode === 'specific' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ position: 'relative', flex: 1 }}>
                  <Search size={13} style={{ position: 'absolute', left: 9, top: 9, color: 'hsl(var(--muted-foreground))' }} />
                  <input
                    type="text"
                    placeholder="Search by agenda or meeting title…"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    style={{
                      width: '100%', padding: '0.45rem 0.75rem 0.45rem 2rem', borderRadius: 6,
                      border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                      color: 'hsl(var(--foreground))', fontSize: '0.75rem',
                    }}
                  />
                </div>
                <button
                  type="button"
                  onClick={selectAllFiltered}
                  style={{
                    padding: '0.45rem 0.65rem', borderRadius: 6, border: '1px solid hsl(var(--border))',
                    background: 'hsl(var(--muted) / .4)', fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={deselectAllFiltered}
                  style={{
                    padding: '0.45rem 0.65rem', borderRadius: 6, border: '1px solid hsl(var(--border))',
                    background: 'hsl(var(--muted) / .4)', fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  Clear
                </button>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                <span>Selected: <strong style={{ color: 'hsl(var(--foreground))' }}>{selectedSampleIds.length}</strong> of {dataset.length}</span>
                <span>Showing {filteredDataset.length}</span>
              </div>

              {/* Scrollable Sample Checkbox List */}
              <div style={{
                maxHeight: 180, overflowY: 'auto', border: '1px solid hsl(var(--border))',
                borderRadius: 8, background: 'hsl(var(--background))', display: 'flex', flexDirection: 'column',
              }}>
                {loadingDataset ? (
                  <div style={{ padding: '1rem', textAlign: 'center', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                    <Loader size={12} className="spin" /> Loading dataset…
                  </div>
                ) : filteredDataset.length === 0 ? (
                  <div style={{ padding: '1rem', textAlign: 'center', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                    No dataset items match your search.
                  </div>
                ) : (
                  filteredDataset.map(item => {
                    const isSelected = selectedSampleIds.includes(item.id);
                    return (
                      <div
                        key={item.id}
                        onClick={() => toggleSample(item.id)}
                        style={{
                          padding: '0.5rem 0.75rem',
                          borderBottom: '1px solid hsl(var(--border) / .5)',
                          background: isSelected ? 'hsl(var(--accent) / .08)' : 'transparent',
                          display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer',
                          transition: 'background 0.12s ease',
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {}}
                          style={{ cursor: 'pointer', accentColor: 'hsl(var(--accent))' }}
                        />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {item.agenda_title || 'General Discussion'}
                          </div>
                          <div style={{ fontSize: '0.67rem', color: 'hsl(var(--muted-foreground))', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {item.recording_title || 'Meeting'} · {item.long_rom_points?.length || 0} Long pts → {item.manual_mom_points?.length || 0} MoM pts
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}

          {/* Run Button */}
          <button
            type="button"
            onClick={handleRunEvaluation}
            disabled={running || !selectedVariant?.has_checkpoint || effectiveSampleCount <= 0}
            style={{
              marginTop: 'auto',
              padding: '0.75rem 1.25rem',
              borderRadius: 8,
              border: 'none',
              background: running
                ? 'hsl(var(--muted))'
                : 'linear-gradient(135deg, hsl(280 80% 55%), hsl(var(--accent)))',
              color: 'white',
              fontSize: '0.85rem',
              fontWeight: 800,
              cursor: running || !selectedVariant?.has_checkpoint || effectiveSampleCount <= 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            }}
          >
            {running ? <Loader size={15} className="spin" /> : <RotateCcw size={15} />}
            {running
              ? `Running Test Evaluation (${effectiveSampleCount} sample${effectiveSampleCount > 1 ? 's' : ''})…`
              : `Run Evaluation (${effectiveSampleCount} Sample${effectiveSampleCount !== 1 ? 's' : ''})`}
          </button>
        </div>
      </div>

      {/* Evaluation Results Section */}
      {testResults && testResults.length > 0 && (
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: 12,
          border: '1.5px solid hsl(var(--border))',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}>
          {/* Header Bar */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexWrap: 'wrap', gap: 10, paddingBottom: '0.85rem', borderBottom: '1px solid hsl(var(--border))',
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '0.95rem', fontWeight: 800 }}>Evaluation Results</span>
                <span style={{
                  fontSize: '0.7rem', padding: '2px 8px', borderRadius: 6,
                  background: 'hsl(var(--accent) / .15)', color: 'hsl(var(--accent))', fontWeight: 700,
                }}>
                  {testResults.length} Sample{testResults.length !== 1 ? 's' : ''} Evaluated
                </span>
              </div>
              {accuracy && (
                <div style={{ fontSize: '0.76rem', color: 'hsl(142 70% 35%)', fontWeight: 600, marginTop: 3 }}>
                  {accuracy}
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() => setModalState({
                isOpen: true,
                results: testResults,
                initialIndex: 0,
                variantName: selectedVariant?.name || 'Trained Variant',
                accuracy: accuracy,
              })}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '0.45rem 1rem', borderRadius: 8,
                background: 'linear-gradient(135deg, hsl(280 80% 55%), hsl(var(--accent)))',
                color: 'white', border: 'none',
                fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              }}
            >
              <Maximize2 size={13} /> View Full Comparison Modal
            </button>
          </div>

          {/* Cards for each test result sample */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {testResults.map((r, i) => (
              <div
                key={i}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 10,
                  border: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--background))',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                {/* Sample Header */}
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  paddingBottom: 6, borderBottom: '1px solid hsl(var(--border) / .6)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      padding: '2px 7px', borderRadius: 4, background: 'hsl(var(--muted))',
                      fontSize: '0.7rem', fontWeight: 700, fontFamily: 'monospace',
                    }}>
                      #{i + 1}
                    </span>
                    <span style={{ fontSize: '0.82rem', fontWeight: 700 }}>
                      {r.agenda_title || `Sample ${i + 1}`}
                    </span>
                    {r.recording_title && (
                      <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                        ({r.recording_title})
                      </span>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() => setModalState({
                      isOpen: true,
                      results: testResults,
                      initialIndex: i,
                      variantName: selectedVariant?.name || 'Trained Variant',
                      accuracy: accuracy,
                    })}
                    style={{
                      border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                      color: 'hsl(var(--foreground))', borderRadius: 6, padding: '3px 8px',
                      fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 4,
                    }}
                  >
                    <Maximize2 size={11} /> Full View
                  </button>
                </div>

                {/* 3 Columns Display */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
                  {/* Column 1: Base Model */}
                  <div style={{
                    background: 'hsl(var(--muted) / .2)',
                    padding: '0.6rem 0.75rem',
                    borderRadius: 8,
                    border: '1px solid hsl(var(--border) / .7)',
                    display: 'flex', flexDirection: 'column', gap: 4,
                  }}>
                    <div style={{ fontSize: '0.68rem', fontWeight: 800, color: 'hsl(var(--muted-foreground))' }}>
                      ⚡ BEFORE (Base Model)
                    </div>
                    <div style={{
                      fontSize: '0.73rem', lineHeight: 1.45, whiteSpace: 'pre-wrap',
                      color: 'hsl(var(--muted-foreground))', maxHeight: 150, overflowY: 'auto',
                    }}>
                      {r.before ? (r.before.slice(0, 320) + (r.before.length > 320 ? '…' : '')) : 'N/A'}
                    </div>
                  </div>

                  {/* Column 2: Trained Variant */}
                  <div style={{
                    background: 'hsl(142 70% 45% / .05)',
                    padding: '0.6rem 0.75rem',
                    borderRadius: 8,
                    border: '1.5px solid hsl(142 70% 45% / .25)',
                    display: 'flex', flexDirection: 'column', gap: 4,
                  }}>
                    <div style={{ fontSize: '0.68rem', fontWeight: 800, color: 'hsl(142 70% 35%)' }}>
                      🧠 AFTER (Trained LoRA Variant)
                    </div>
                    <div style={{
                      fontSize: '0.73rem', lineHeight: 1.45, whiteSpace: 'pre-wrap',
                      color: 'hsl(var(--foreground))', maxHeight: 150, overflowY: 'auto',
                    }}>
                      {r.after ? (r.after.slice(0, 320) + (r.after.length > 320 ? '…' : '')) : 'N/A'}
                    </div>
                  </div>

                  {/* Column 3: Manual Ground Truth */}
                  <div style={{
                    background: 'hsl(280 80% 55% / .05)',
                    padding: '0.6rem 0.75rem',
                    borderRadius: 8,
                    border: '1.5px solid hsl(280 80% 55% / .25)',
                    display: 'flex', flexDirection: 'column', gap: 4,
                  }}>
                    <div style={{ fontSize: '0.68rem', fontWeight: 800, color: 'hsl(280 80% 55%)' }}>
                      📝 MANUAL (Human Target Ground-Truth)
                    </div>
                    <div style={{
                      fontSize: '0.73rem', lineHeight: 1.45, whiteSpace: 'pre-wrap',
                      color: 'hsl(var(--foreground))', maxHeight: 150, overflowY: 'auto',
                    }}>
                      {r.manual ? (r.manual.slice(0, 320) + (r.manual.length > 320 ? '…' : '')) : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Embedded Full Comparison Modal */}
      <TestComparisonModal
        isOpen={modalState.isOpen}
        onClose={() => setModalState(prev => ({ ...prev, isOpen: false }))}
        results={modalState.results}
        initialIndex={modalState.initialIndex}
        variantName={modalState.variantName}
        accuracySummary={modalState.accuracy}
      />
    </div>
  );
}
