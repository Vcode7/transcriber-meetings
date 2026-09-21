import { useState, useEffect, useRef } from 'react';
import {
  Brain, Play, Loader, ChevronDown, ChevronUp, Check,
  AlertCircle, RefreshCw, Eye,
} from 'lucide-react';
import { startTraining, getVariant, listTrainingData, listVariants } from '../api/romTrainingApi';
import type { LoRAConfig, RomVariant, TrainingDataEntry, TestResult } from '../types/romTrainingTypes';

const DEFAULT_LORA: LoRAConfig = {
  method: 'qlora',
  r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  target_modules: 'q_proj,v_proj',
  learning_rate: 0.0002,
  num_epochs: 3,
  per_device_train_batch_size: 2,
  gradient_accumulation_steps: 4,
  use_4bit: true,
  bnb_4bit_compute_dtype: 'float16',
  max_seq_length: 2048,
};

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  label: {
    display: 'block', fontSize: '0.78rem', fontWeight: 700,
    color: 'hsl(var(--foreground))', marginBottom: '0.35rem',
  } as React.CSSProperties,
  input: {
    width: '100%', padding: '0.55rem 0.8rem', borderRadius: '8px',
    border: '1px solid hsl(var(--border))',
    background: 'hsl(var(--background))', color: 'hsl(var(--foreground))',
    fontSize: '0.82rem', fontFamily: 'Inter, sans-serif',
    boxSizing: 'border-box',
  } as React.CSSProperties,
};

export default function TrainingTab() {
  const [dbSize, setDbSize] = useState(0);
  const [variants, setVariants] = useState<RomVariant[]>([]);

  // Config
  const [modelPath, setModelPath] = useState('');
  const [trainingType, setTrainingType] = useState<'new' | 'upgrade'>('new');
  const [parentVariantId, setParentVariantId] = useState('');
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [datasetSize, setDatasetSize] = useState(50);
  const [testSize, setTestSize] = useState(10);
  const [loraConfig, setLoraConfig] = useState<LoRAConfig>(DEFAULT_LORA);
  const [showLoRA, setShowLoRA] = useState(false);

  // Training state
  const [training, setTraining] = useState(false);
  const [currentVariantId, setCurrentVariantId] = useState<string | null>(null);
  const [currentVariant, setCurrentVariant] = useState<RomVariant | null>(null);
  const [error, setError] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listTrainingData().then(r => setDbSize(r.total || 0)).catch(() => {});
    listVariants().then(r => setVariants(r.variants || [])).catch(() => {});
  }, []);

  useEffect(() => {
    if (!currentVariantId) return;
    pollRef.current = setInterval(async () => {
      try {
        const v = await getVariant(currentVariantId);
        setCurrentVariant(v);
        if (v.status === 'done' || v.status === 'error') {
          clearInterval(pollRef.current!);
          setTraining(false);
          listVariants().then(r => setVariants(r.variants || [])).catch(() => {});
        }
      } catch {
        clearInterval(pollRef.current!);
        setTraining(false);
      }
    }, 3000);
    return () => clearInterval(pollRef.current!);
  }, [currentVariantId]);

  const handleTrain = async () => {
    if (!modelPath.trim() || !name.trim()) {
      setError('Model Path and Name are required.');
      return;
    }
    if (datasetSize + testSize > dbSize) {
      setError(`Not enough data. Need ${datasetSize + testSize}, have ${dbSize}.`);
      return;
    }
    setError('');
    setTraining(true);
    setCurrentVariant(null);
    try {
      const res = await startTraining({
        name, description: desc,
        model_path: modelPath,
        training_type: trainingType,
        parent_variant_id: trainingType === 'upgrade' ? parentVariantId : null,
        lora_config: loraConfig,
        dataset_size: datasetSize,
        test_size: testSize,
      });
      setCurrentVariantId(res.variant_id);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to start training.');
      setTraining(false);
    }
  };

  const patchLora = (key: keyof LoRAConfig, val: any) =>
    setLoraConfig(prev => ({ ...prev, [key]: val }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Config panel */}
      <div style={S.card}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '1rem' }}>
          <Brain size={16} style={{ color: 'hsl(var(--accent))' }} />
          <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Training Configuration</span>
          <span style={{
            fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px',
            borderRadius: 6, background: 'hsl(var(--accent) / .12)', color: 'hsl(var(--accent))',
          }}>
            {dbSize} entries in DB
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          {/* Name */}
          <div>
            <label style={S.label}>Variant Name *</label>
            <input style={S.input} value={name} onChange={e => setName(e.target.value)} placeholder="e.g. ROM-v1" />
          </div>
          {/* Model Path */}
          <div>
            <label style={S.label}>Model Path *</label>
            <input style={S.input} value={modelPath} onChange={e => setModelPath(e.target.value)} placeholder="e.g. Qwen/Qwen2.5-7B-Instruct" />
          </div>
          {/* Training Type */}
          <div>
            <label style={S.label}>Training Type</label>
            <div style={{ display: 'flex', gap: 8 }}>
              {(['new', 'upgrade'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setTrainingType(t)}
                  style={{
                    flex: 1, padding: '0.5rem', borderRadius: 8, border: 'none',
                    background: trainingType === t ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                    color: trainingType === t ? 'white' : 'hsl(var(--foreground))',
                    fontWeight: 600, fontSize: '0.8rem', cursor: 'pointer',
                    fontFamily: 'Inter, sans-serif',
                  }}
                >
                  {t === 'new' ? 'New Training' : 'Upgrade Training'}
                </button>
              ))}
            </div>
          </div>
          {/* Parent Variant (upgrade only) */}
          {trainingType === 'upgrade' && (
            <div>
              <label style={S.label}>Base Variant</label>
              <select
                value={parentVariantId}
                onChange={e => setParentVariantId(e.target.value)}
                style={S.input}
              >
                <option value="">— Select existing variant —</option>
                {variants.filter(v => v.status === 'done').map(v => (
                  <option key={v.id} value={v.id}>{v.name}</option>
                ))}
              </select>
            </div>
          )}
          {/* Dataset + Test sizes */}
          <div>
            <label style={S.label}>Dataset Size (train samples)</label>
            <input type="number" style={S.input} value={datasetSize}
              onChange={e => setDatasetSize(Number(e.target.value))} min={1} max={dbSize} />
          </div>
          <div>
            <label style={S.label}>Test Case Size</label>
            <input type="number" style={S.input} value={testSize}
              onChange={e => setTestSize(Number(e.target.value))} min={1} max={dbSize} />
          </div>
          {/* Description */}
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={S.label}>Description (optional)</label>
            <input style={S.input} value={desc} onChange={e => setDesc(e.target.value)} placeholder="Brief description" />
          </div>
        </div>

        {/* LoRA Parameters */}
        <div style={{ marginTop: '1rem' }}>
          <button
            onClick={() => setShowLoRA(!showLoRA)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: '0.82rem', fontWeight: 700, color: 'hsl(var(--foreground))',
              fontFamily: 'Inter, sans-serif', padding: '0.3rem 0',
            }}
          >
            {showLoRA ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            LoRA / QLoRA Parameters
          </button>
          {showLoRA && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginTop: '0.75rem' }}>
              {/* Method */}
              <div>
                <label style={S.label}>Method</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  {(['lora', 'qlora'] as const).map(m => (
                    <button key={m} onClick={() => patchLora('method', m)}
                      style={{
                        flex: 1, padding: '0.4rem', borderRadius: 8, border: 'none',
                        background: loraConfig.method === m ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                        color: loraConfig.method === m ? 'white' : 'hsl(var(--foreground))',
                        fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer',
                        fontFamily: 'Inter, sans-serif',
                      }}
                    >{m.toUpperCase()}</button>
                  ))}
                </div>
              </div>
              {/* Numeric fields */}
              {([
                ['r', 'Rank (r)', 'number'],
                ['lora_alpha', 'LoRA Alpha', 'number'],
                ['lora_dropout', 'Dropout', 'number'],
                ['learning_rate', 'Learning Rate', 'number'],
                ['num_epochs', 'Epochs', 'number'],
                ['per_device_train_batch_size', 'Batch Size', 'number'],
                ['gradient_accumulation_steps', 'Grad Accum Steps', 'number'],
                ['max_seq_length', 'Max Seq Length', 'number'],
              ] as [keyof LoRAConfig, string, string][]).map(([key, label]) => (
                <div key={key}>
                  <label style={S.label}>{label}</label>
                  <input
                    type="number"
                    style={S.input}
                    value={loraConfig[key] as number}
                    onChange={e => patchLora(key, key === 'lora_dropout' || key === 'learning_rate'
                      ? parseFloat(e.target.value) : parseInt(e.target.value))}
                  />
                </div>
              ))}
              <div>
                <label style={S.label}>Target Modules</label>
                <input style={S.input} value={loraConfig.target_modules}
                  onChange={e => patchLora('target_modules', e.target.value)}
                  placeholder="q_proj,v_proj" />
              </div>
              {loraConfig.method === 'qlora' && (
                <div>
                  <label style={S.label}>BnB Compute Dtype</label>
                  <select value={loraConfig.bnb_4bit_compute_dtype}
                    onChange={e => patchLora('bnb_4bit_compute_dtype', e.target.value)}
                    style={S.input}
                  >
                    <option value="float16">float16</option>
                    <option value="bfloat16">bfloat16</option>
                    <option value="float32">float32</option>
                  </select>
                </div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div style={{ marginTop: 10, fontSize: '0.78rem', color: 'hsl(var(--destructive))', display: 'flex', alignItems: 'center', gap: 6 }}>
            <AlertCircle size={13} /> {error}
          </div>
        )}

        <div style={{ marginTop: '1.25rem' }}>
          <button
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '0.6rem 1.5rem', borderRadius: '10px', border: 'none',
              background: training ? 'hsl(var(--muted))' : 'hsl(var(--accent))',
              color: training ? 'hsl(var(--muted-foreground))' : 'white',
              fontSize: '0.9rem', fontWeight: 700, cursor: training ? 'not-allowed' : 'pointer',
              fontFamily: 'Inter, sans-serif',
            }}
            onClick={handleTrain}
            disabled={training}
          >
            {training ? <Loader size={16} className="spin" /> : <Play size={16} />}
            {training ? 'Training in progress…' : 'Train'}
          </button>
        </div>
      </div>

      {/* Progress / status */}
      {currentVariant && (
        <div style={S.card}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.75rem' }}>
            {currentVariant.status === 'training' && <Loader size={16} className="spin" style={{ color: 'hsl(var(--accent))' }} />}
            {currentVariant.status === 'done' && <Check size={16} style={{ color: 'hsl(142 70% 45%)' }} />}
            {currentVariant.status === 'error' && <AlertCircle size={16} style={{ color: 'hsl(var(--destructive))' }} />}
            <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>
              {currentVariant.name} — {currentVariant.status.toUpperCase()}
            </span>
          </div>
          {currentVariant.status === 'training' && (
            <p style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
              Training is running in the background. This may take a while depending on dataset size and hardware.
              This page will update automatically.
            </p>
          )}
          {currentVariant.status === 'error' && currentVariant.error && (
            <div style={{
              padding: '0.75rem', borderRadius: 8,
              background: 'hsl(var(--destructive) / .08)',
              border: '1px solid hsl(var(--destructive) / .2)',
              fontSize: '0.78rem', color: 'hsl(var(--destructive))',
            }}>
              {currentVariant.error}
            </div>
          )}
          {currentVariant.status === 'done' && currentVariant.accuracy && (
            <div style={{
              padding: '0.75rem', borderRadius: 8,
              background: 'hsl(142 70% 45% / .08)',
              border: '1px solid hsl(142 70% 45% / .2)',
              fontSize: '0.82rem', fontWeight: 600, color: 'hsl(142 70% 35%)',
              marginBottom: '1rem',
            }}>
              <Check size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />
              {currentVariant.accuracy}
            </div>
          )}

          {/* Test Results */}
          {currentVariant.status === 'done' && currentVariant.test_results.length > 0 && (
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.85rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Eye size={14} /> Test Case Comparison ({currentVariant.test_results.length} samples)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {currentVariant.test_results.map((result, idx) => (
                  <TestResultCard key={idx} result={result} index={idx} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TestResultCard({ result, index }: { result: TestResult; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);
  return (
    <div style={{ borderRadius: 10, border: '1px solid hsl(var(--border))', overflow: 'hidden' }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          padding: '0.6rem 0.85rem', cursor: 'pointer',
          background: 'hsl(var(--muted) / .3)',
          borderBottom: expanded ? '1px solid hsl(var(--border))' : 'none',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}
      >
        <span style={{ fontWeight: 700, fontSize: '0.8rem' }}>
          Sample {index + 1}: {result.agenda_title || result.recording_title}
        </span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {expanded && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 0 }}>
          {([
            { label: 'BEFORE (Base Model)', text: result.before, color: 'hsl(0 0% 50%)' },
            { label: 'AFTER (Trained Model)', text: result.after, color: 'hsl(142 70% 35%)' },
            { label: 'MANUAL (Ground Truth)', text: result.manual, color: 'hsl(280 80% 55%)' },
          ] as { label: string; text: string; color: string }[]).map((col, i) => (
            <div
              key={i}
              style={{
                padding: '0.75rem',
                borderRight: i < 2 ? '1px solid hsl(var(--border))' : 'none',
              }}
            >
              <div style={{ fontSize: '0.68rem', fontWeight: 700, color: col.color, marginBottom: 8 }}>{col.label}</div>
              <div style={{ fontSize: '0.73rem', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{col.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
