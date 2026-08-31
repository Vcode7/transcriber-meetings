import { useState, useEffect } from 'react';
import {
  Settings,
  Save,
  RefreshCw,
  Check,
  Cpu,
  Sliders,
  Brain,
  Layers,
  Sparkles,
  Database,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import { useStage1Store } from '../store/stage1Store';
import { useStage2Store } from '../store/stage2Store';
import { useStage3Store } from '../store/stage3Store';
import { loadStage1Settings, saveStage1Settings, listStage1Variants } from '../api/stage1Api';
import { loadStage2Settings, saveStage2Settings, listStage2Variants } from '../api/stage2Api';
import { loadStage3Settings, saveStage3Settings, listStage3Variants } from '../api/stage3Api';
import { fetchOllamaModels } from '../api/trainingApi';
import type { Stage1Settings, Stage1Variant } from '../types/stage1Types';
import type { Stage2Settings, Stage2Variant } from '../types/stage2Types';
import type { Stage3Settings, Stage3Variant } from '../types/training';
import { DEFAULT_STAGE1_SETTINGS } from '../types/stage1Types';
import { DEFAULT_STAGE2_SETTINGS } from '../types/stage2Types';

type SettingsCategory = 'all' | 'global' | 'stage1' | 'stage2' | 'stage3';

const S = {
  section: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.35rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.15rem',
    boxShadow: '0 2px 8px -2px hsl(var(--foreground) / 0.04)',
  } as React.CSSProperties,
  sectionTitle: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '0.2rem',
  } as React.CSSProperties,
  iconBadge: (bg = 'hsl(var(--accent) / .15)') => ({
    width: 32,
    height: 32,
    borderRadius: '8px',
    background: bg,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  } as React.CSSProperties),
  label: {
    display: 'block',
    fontSize: '0.78rem',
    fontWeight: 700,
    color: 'hsl(var(--foreground))',
    marginBottom: '0.35rem',
    letterSpacing: '0.01em',
  },
  desc: {
    fontSize: '0.72rem',
    color: 'hsl(var(--muted-foreground))',
    marginTop: '5px',
    lineHeight: 1.4,
  },
  input: {
    width: '100%',
    padding: '0.6rem 0.85rem',
    borderRadius: '8px',
    border: '1px solid hsl(var(--border))',
    background: 'hsl(var(--background))',
    color: 'hsl(var(--foreground))',
    fontSize: '0.85rem',
    outline: 'none',
    boxSizing: 'border-box' as const,
    transition: 'border-color 0.15s ease',
  },
  select: {
    width: '100%',
    padding: '0.6rem 0.85rem',
    borderRadius: '8px',
    border: '1px solid hsl(var(--border))',
    background: 'hsl(var(--background))',
    color: 'hsl(var(--foreground))',
    fontSize: '0.85rem',
    outline: 'none',
    boxSizing: 'border-box' as const,
    cursor: 'pointer',
  },
  rangeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  } as React.CSSProperties,
  rangeValBadge: {
    minWidth: '54px',
    textAlign: 'center' as const,
    padding: '3px 8px',
    borderRadius: '6px',
    background: 'hsl(var(--accent) / 0.12)',
    color: 'hsl(var(--accent))',
    fontWeight: 700,
    fontSize: '0.82rem',
    border: '1px solid hsl(var(--accent) / 0.25)',
  },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' } as React.CSSProperties,
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' } as React.CSSProperties,
};

export default function SettingsTab() {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('all');

  // Stage 1, 2, 3 Settings State
  const [s1Settings, setS1Settings] = useState<Stage1Settings>(DEFAULT_STAGE1_SETTINGS);
  const [s2Settings, setS2Settings] = useState<Stage2Settings>(DEFAULT_STAGE2_SETTINGS);
  const [s3Settings, setS3Settings] = useState<Stage3Settings>({
    model_name: '',
    optimizer: 'BootstrapFewShot',
    num_trials: 10,
    eval_split: 0.2,
    bootstrap_examples: 3,
    max_demonstrations: 4,
    temperature: 0.0,
    max_tokens: 2048,
    points_per_batch: 15,
  });

  // Variants preview
  const [s1Variants, setS1Variants] = useState<Stage1Variant[]>([]);
  const [s2Variants, setS2Variants] = useState<Stage2Variant[]>([]);
  const [s3Variants, setS3Variants] = useState<Stage3Variant[]>([]);

  // Models
  const [models, setModels] = useState<{ name: string }[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  // Status
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const loadAllSettings = async () => {
    setLoading(true);
    try {
      const [s1, s2, s3, v1, v2, v3, m] = await Promise.all([
        loadStage1Settings().catch(() => DEFAULT_STAGE1_SETTINGS),
        loadStage2Settings().catch(() => DEFAULT_STAGE2_SETTINGS),
        loadStage3Settings().catch(() => ({
          model_name: '',
          optimizer: 'BootstrapFewShot',
          num_trials: 10,
          eval_split: 0.2,
          bootstrap_examples: 3,
          max_demonstrations: 4,
          temperature: 0.0,
          max_tokens: 2048,
          points_per_batch: 15,
        })),
        listStage1Variants().catch(() => []),
        listStage2Variants().then((r) => r.variants || []).catch(() => []),
        listStage3Variants().then((r) => r.variants || []).catch(() => []),
        fetchOllamaModels().catch(() => []),
      ]);

      setS1Settings(s1);
      setS2Settings(s2);
      setS3Settings(s3);
      setS1Variants(v1);
      setS2Variants(v2);
      setS3Variants(v3);
      setModels(m);
    } catch (err) {
      console.error('Failed to load training settings:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllSettings();
  }, []);

  const updateGlobalModel = (modelName: string) => {
    setS1Settings((prev) => ({ ...prev, model_name: modelName }));
    setS2Settings((prev) => ({ ...prev, model_name: modelName }));
    setS3Settings((prev) => ({ ...prev, model_name: modelName }));
  };

  const updateGlobalOptimizer = (optimizer: 'BootstrapFewShot' | 'MIPROv2') => {
    setS1Settings((prev) => ({ ...prev, optimizer }));
    setS2Settings((prev) => ({ ...prev, optimizer }));
    setS3Settings((prev) => ({ ...prev, optimizer }));
  };

  const updateGlobalHyperparam = (key: 'temperature' | 'max_tokens' | 'eval_split' | 'bootstrap_examples' | 'max_demonstrations' | 'num_trials', value: number) => {
    setS1Settings((prev) => ({ ...prev, [key]: value }));
    setS2Settings((prev) => ({ ...prev, [key]: value }));
    setS3Settings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSaveAll = async () => {
    setSaving(true);
    setSaveSuccess(false);
    try {
      await Promise.all([
        saveStage1Settings(s1Settings),
        saveStage2Settings(s2Settings),
        saveStage3Settings(s3Settings),
      ]);

      // Sync Zustand stores
      useStage1Store.getState().setSettings(s1Settings);
      useStage2Store.getState().setSettings(s2Settings);
      useStage3Store.getState().setSettings(s3Settings);

      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 4000);
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleResetDefaults = () => {
    if (!window.confirm('Reset all Stage 1, Stage 2, and Stage 3 training settings to factory defaults?')) return;
    setS1Settings(DEFAULT_STAGE1_SETTINGS);
    setS2Settings(DEFAULT_STAGE2_SETTINGS);
    setS3Settings({
      model_name: '',
      optimizer: 'BootstrapFewShot',
      num_trials: 10,
      eval_split: 0.2,
      bootstrap_examples: 3,
      max_demonstrations: 4,
      temperature: 0.0,
      max_tokens: 2048,
      points_per_batch: 15,
    });
    setSaveSuccess(false);
  };

  const handleRefreshModels = async () => {
    setModelsLoading(true);
    try {
      const m = await fetchOllamaModels();
      setModels(m);
    } catch (err) {
      console.error(err);
    } finally {
      setModelsLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem 2rem', color: 'hsl(var(--muted-foreground))' }}>
        <RefreshCw size={24} style={{ animation: 'spin 1s linear infinite', display: 'block', margin: '0 auto 12px' }} />
        <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Loading Training Settings...</div>
        <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* ── Top Header & Sub-Navigation ─────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={18} color="hsl(var(--accent))" />
            DSPy Training &amp; Pipeline Settings
          </h2>
          <p style={{ margin: '3px 0 0', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
            Configure LLM models, DSPy optimizers, windowing hyperparameters, and stage thresholds.
          </p>
        </div>

        {/* Category Pill Filters */}
        <div style={{
          display: 'flex',
          gap: '4px',
          background: 'hsl(var(--card))',
          padding: '3px',
          borderRadius: '10px',
          border: '1px solid hsl(var(--border))',
        }}>
          {[
            { id: 'all', label: 'All Settings' },
            { id: 'global', label: 'Global & LLM' },
            { id: 'stage1', label: 'Stage 1' },
            { id: 'stage2', label: 'Stage 2' },
            { id: 'stage3', label: 'Stage 3' },
          ].map((cat) => {
            const isActive = activeCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id as SettingsCategory)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '7px',
                  border: 'none',
                  background: isActive ? 'hsl(var(--accent))' : 'transparent',
                  color: isActive ? 'white' : 'hsl(var(--muted-foreground))',
                  fontSize: '0.76rem',
                  fontWeight: isActive ? 700 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {cat.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Section 1: Global LLM & DSPy Optimizers ────────────────────────── */}
      {(activeCategory === 'all' || activeCategory === 'global') && (
        <div style={S.section}>
          <div style={S.sectionTitle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={S.iconBadge()}>
                <Cpu size={16} color="hsl(var(--accent))" />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>Global Model &amp; DSPy Optimizer</h3>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                  Shared default LLM and optimizer applied across all training stages
                </div>
              </div>
            </div>
            <button
              onClick={handleRefreshModels}
              disabled={modelsLoading}
              title="Refresh Ollama Models"
              style={{
                display: 'flex', alignItems: 'center', gap: '4px',
                padding: '4px 9px', borderRadius: '6px', border: '1px solid hsl(var(--border))',
                background: 'hsl(var(--background))', fontSize: '0.72rem', cursor: 'pointer',
                color: 'hsl(var(--muted-foreground))',
              }}
            >
              <RefreshCw size={11} style={{ animation: modelsLoading ? 'spin 1s linear infinite' : 'none' }} />
              {modelsLoading ? 'Loading...' : 'Refresh Models'}
            </button>
          </div>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Ollama Model</label>
              <select
                value={s1Settings.model_name || s2Settings.model_name || s3Settings.model_name}
                onChange={(e) => updateGlobalModel(e.target.value)}
                style={S.select}
              >
                <option value="">— Select Ollama Model —</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </select>
              <div style={S.desc}>Primary model used for DSPy prompt compilation and metric evaluation.</div>
            </div>

            <div>
              <label style={S.label}>DSPy Optimizer</label>
              <select
                value={s1Settings.optimizer}
                onChange={(e) => updateGlobalOptimizer(e.target.value as 'BootstrapFewShot' | 'MIPROv2')}
                style={S.select}
              >
                <option value="BootstrapFewShot">BootstrapFewShot (Fast &amp; Recommended)</option>
                <option value="MIPROv2">MIPROv2 (Deep Multi-Prompt Instruction Search)</option>
              </select>
              <div style={S.desc}>Algorithm used to synthesize and refine few-shot demonstrations.</div>
            </div>
          </div>

          {/* Hyperparameters Grid */}
          <div style={{ ...S.grid3, paddingTop: '0.5rem', borderTop: '1px solid hsl(var(--border) / 0.5)' }}>
            <div>
              <label style={S.label}>Temperature</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={s1Settings.temperature}
                  onChange={(e) => updateGlobalHyperparam('temperature', parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.temperature}</span>
              </div>
              <div style={S.desc}>0.0 = Deterministic output (recommended for extraction).</div>
            </div>

            <div>
              <label style={S.label}>Max Tokens</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={512} max={8096} step={256}
                  value={s1Settings.max_tokens}
                  onChange={(e) => updateGlobalHyperparam('max_tokens', parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.max_tokens}</span>
              </div>
              <div style={S.desc}>Maximum generation output length per LLM invocation.</div>
            </div>

            <div>
              <label style={S.label}>Train / Validation Split</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0.1} max={0.4} step={0.05}
                  value={s1Settings.eval_split}
                  onChange={(e) => updateGlobalHyperparam('eval_split', parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{Math.round(s1Settings.eval_split * 100)}%</span>
              </div>
              <div style={S.desc}>Dataset split reserved for variant scoring evaluation.</div>
            </div>

            <div>
              <label style={S.label}>Bootstrap Examples</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={1} max={10} step={1}
                  value={s1Settings.bootstrap_examples}
                  onChange={(e) => updateGlobalHyperparam('bootstrap_examples', parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.bootstrap_examples}</span>
              </div>
              <div style={S.desc}>Max bootstrapped demonstrations synthesized by DSPy.</div>
            </div>

            <div>
              <label style={S.label}>Max Demonstrations</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={1} max={10} step={1}
                  value={s1Settings.max_demonstrations}
                  onChange={(e) => updateGlobalHyperparam('max_demonstrations', parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.max_demonstrations}</span>
              </div>
              <div style={S.desc}>Max labeled ground-truth examples included in optimized prompt.</div>
            </div>

            <div>
              <label style={S.label}>MIPROv2 Number of Trials</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={5} max={50} step={5}
                  value={s1Settings.num_trials}
                  onChange={(e) => updateGlobalHyperparam('num_trials', parseInt(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.num_trials}</span>
              </div>
              <div style={S.desc}>Number of candidate instruction variants evaluated by MIPROv2.</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Section 2: Stage 1 Settings ─────────────────────────────────────── */}
      {(activeCategory === 'all' || activeCategory === 'stage1') && (
        <div style={S.section}>
          <div style={S.sectionTitle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={S.iconBadge('hsl(262 80% 65% / 0.15)')}>
                <Brain size={16} color="hsl(262 80% 65%)" />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>Stage 1 — Discussion Point Extraction</h3>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                  Transcript window duration, overlap offsets, and validation threshold gates
                </div>
              </div>
            </div>
            <span style={{
              fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
              background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
            }}>
              {s1Variants.length} Variant(s)
            </span>
          </div>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Default Window Duration (minutes)</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0.5} max={10} step={0.5}
                  value={s1Settings.default_window_size_minutes}
                  onChange={(e) => setS1Settings({ ...s1Settings, default_window_size_minutes: parseFloat(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.default_window_size_minutes} min</span>
              </div>
              <div style={S.desc}>Length of each overlapping transcript segment sliced for Stage 1 extraction.</div>
            </div>

            <div>
              <label style={S.label}>Default Window Overlap (seconds)</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0} max={120} step={5}
                  value={s1Settings.default_window_overlap_seconds}
                  onChange={(e) => setS1Settings({ ...s1Settings, default_window_overlap_seconds: parseFloat(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s1Settings.default_window_overlap_seconds}s</span>
              </div>
              <div style={S.desc}>Overlap buffer to prevent cutting mid-sentence or losing conversational context.</div>
            </div>

            <div>
              <label style={S.label}>Minimum Transcript Coverage Threshold</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0.1} max={0.9} step={0.05}
                  value={s1Settings.min_coverage_threshold}
                  onChange={(e) => setS1Settings({ ...s1Settings, min_coverage_threshold: parseFloat(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{Math.round(s1Settings.min_coverage_threshold * 100)}%</span>
              </div>
              <div style={S.desc}>Minimum acceptable vocabulary overlap between transcript and points.</div>
            </div>

            <div>
              <label style={S.label}>Minimum JSON Validity Gate</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={0.5} max={1.0} step={0.05}
                  value={s1Settings.min_json_validity_threshold}
                  onChange={(e) => setS1Settings({ ...s1Settings, min_json_validity_threshold: parseFloat(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{Math.round(s1Settings.min_json_validity_threshold * 100)}%</span>
              </div>
              <div style={S.desc}>Strictness required for valid JSON schema compliance before finalizing a variant.</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Section 3: Stage 2 Settings ─────────────────────────────────────── */}
      {(activeCategory === 'all' || activeCategory === 'stage2') && (
        <div style={S.section}>
          <div style={S.sectionTitle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={S.iconBadge('hsl(140 70% 45% / 0.15)')}>
                <Sparkles size={16} color="hsl(140 70% 45%)" />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>Stage 2 — Point Consolidation &amp; Enhancement</h3>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                  Grouping batch sizes, context retrieval RAG parameters, and reference point alignment
                </div>
              </div>
            </div>
            <span style={{
              fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
              background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
            }}>
              {s2Variants.length} Variant(s)
            </span>
          </div>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Stage 2 Training Mode</label>
              <select
                value={s2Settings.training_mode || 'dspy'}
                onChange={(e) => setS2Settings({ ...s2Settings, training_mode: e.target.value as any })}
                style={S.select}
              >
                <option value="dspy">DSPy Prompt Optimization (Ollama)</option>
                <option value="lora">LoRA Fine-Tuning (Local Hugging Face Model)</option>
                <option value="qlora">QLoRA 4-bit Fine-Tuning (Local Hugging Face Model)</option>
              </select>
              <div style={S.desc}>Select whether Stage 2 optimizes prompt instructions or fine-tunes local PEFT adapters.</div>
            </div>

            {(s2Settings.training_mode === 'lora' || s2Settings.training_mode === 'qlora') && (
              <div>
                <label style={S.label}>Local Hugging Face Model Path</label>
                <input
                  type="text"
                  placeholder="e.g. C:\models\Qwen2.5-7B-Instruct"
                  value={s2Settings.hf_model_path || ''}
                  onChange={(e) => setS2Settings({ ...s2Settings, hf_model_path: e.target.value })}
                  style={S.input}
                />
                <div style={S.desc}>Absolute directory path containing Hugging Face config.json and weights (Ollama not used).</div>
              </div>
            )}
          </div>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Points per Group (Batch Size)</label>
              <div style={S.rangeRow}>
                <input
                  type="range" min={3} max={10} step={1}
                  value={s2Settings.points_per_group}
                  onChange={(e) => setS2Settings({ ...s2Settings, points_per_group: parseInt(e.target.value) })}
                  style={{ flex: 1 }}
                />
                <span style={S.rangeValBadge}>{s2Settings.points_per_group} pts</span>
              </div>
              <div style={S.desc}>Number of raw Stage 1 points bundled together into each Stage 2 consolidation step.</div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <label style={S.label}>Context Retrieval (Global &amp; Meeting RAG)</label>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '0.65rem 0.85rem', borderRadius: '8px',
                background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))',
              }}>
                <input
                  type="checkbox"
                  id="s2_context_toggle"
                  checked={s2Settings.context_retrieval}
                  onChange={(e) => setS2Settings({ ...s2Settings, context_retrieval: e.target.checked })}
                  style={{ cursor: 'pointer', width: '16px', height: '16px' }}
                />
                <label htmlFor="s2_context_toggle" style={{ fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer' }}>
                  Enable Global &amp; Meeting Context Retrieval
                </label>
              </div>
              <div style={S.desc}>Enriches consolidated points with organizational acronyms, projects, and meeting context.</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Section 4: Stage 3 Settings ─────────────────────────────────────── */}
      {(activeCategory === 'all' || activeCategory === 'stage3') && (
        <div style={S.section}>
          <div style={S.sectionTitle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={S.iconBadge('hsl(210 80% 55% / 0.15)')}>
                <Layers size={16} color="hsl(210 80% 55%)" />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>Stage 3 — Point → Agenda Assignment</h3>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                  Batch sizing for assignment training and cosine similarity candidate matching
                </div>
              </div>
            </div>
            <span style={{
              fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
              background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
            }}>
              {s3Variants.length} Variant(s)
            </span>
          </div>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Points per Training Batch</label>
              <select
                value={s3Settings.points_per_batch || 15}
                onChange={(e) => setS3Settings({ ...s3Settings, points_per_batch: parseInt(e.target.value) })}
                style={S.select}
              >
                <option value={10}>10 Discussion Points per Batch</option>
                <option value={15}>15 Discussion Points per Batch (Recommended)</option>
                <option value={20}>20 Discussion Points per Batch</option>
              </select>
              <div style={S.desc}>Stage 2 points are sent in batches with Top-3 candidate agendas to the LLM.</div>
            </div>

            <div>
              <label style={S.label}>Top Candidate Agendas</label>
              <input
                type="text"
                disabled
                value="Top 3 Matching Agendas (Cosine Similarity)"
                style={{ ...S.input, opacity: 0.8, background: 'hsl(var(--muted) / 0.3)' }}
              />
              <div style={S.desc}>Automatically calculates embedding similarity and passes the top 3 matches to the optimizer.</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Sticky Save / Reset Action Bar ──────────────────────────────────── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
        padding: '1rem 1.35rem',
        borderRadius: '12px',
        background: 'hsl(var(--card))',
        border: '1px solid hsl(var(--border))',
        boxShadow: '0 4px 16px -4px hsl(var(--foreground) / 0.08)',
        position: 'sticky',
        bottom: 0,
        zIndex: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={handleSaveAll}
            disabled={saving}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0.65rem 1.4rem',
              borderRadius: '8px',
              border: 'none',
              background: 'hsl(var(--accent))',
              color: 'white',
              cursor: saving ? 'wait' : 'pointer',
              fontSize: '0.86rem',
              fontWeight: 700,
              opacity: saving ? 0.7 : 1,
              transition: 'all 0.15s ease',
            }}
          >
            {saving ? (
              <>
                <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} />
                Saving All Settings...
              </>
            ) : saveSuccess ? (
              <>
                <Check size={14} />
                Settings Saved!
              </>
            ) : (
              <>
                <Save size={14} />
                Save All Settings
              </>
            )}
          </button>

          <button
            onClick={handleResetDefaults}
            type="button"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '5px',
              padding: '0.65rem 1rem',
              borderRadius: '8px',
              border: '1px solid hsl(var(--border))',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: '0.82rem',
              color: 'hsl(var(--muted-foreground))',
              fontWeight: 600,
            }}
          >
            <RotateCcw size={13} />
            Reset to Defaults
          </button>
        </div>

        {saveSuccess && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            fontSize: '0.8rem', fontWeight: 600, color: 'hsl(140 70% 45%)',
          }}>
            <CheckCircle2 size={15} />
            Settings saved and will apply to all subsequent training runs.
          </div>
        )}

        <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
          All modifications are persisted across Stage 1, 2, and 3 DSPy pipelines.
        </div>
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
