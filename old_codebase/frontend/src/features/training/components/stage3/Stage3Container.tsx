import { useEffect } from 'react';
import { Database, Zap, BarChart3, History } from 'lucide-react';
import { useStage3Store } from '../../store/stage3Store';
import { fetchMeetingsForTraining } from '../../api/trainingApi';
import { listStage3Variants, loadStage3Settings } from '../../api/stage3Api';
import type { Stage3Tab } from '../../store/stage3Store';
import S3DatasetTab from './S3DatasetTab';
import S3TrainingLoopTab from './S3TrainingLoopTab';
import S3VariantsTab from './S3VariantsTab';
import S3HistoryTab from './S3HistoryTab';

const STAGE3_TABS: { id: Stage3Tab; label: string; icon: React.ElementType; desc: string }[] = [
  { id: 'dataset', label: 'Dataset', icon: Database, desc: 'Load Stage 2 points & verify agenda' },
  { id: 'training_loop', label: 'Training Loop', icon: Zap, desc: 'Optimize, validate & move points' },
  { id: 'variants', label: 'Evaluation / Variants', icon: BarChart3, desc: 'Compare all Stage 3 DSPy variants' },
  { id: 'history', label: 'History', icon: History, desc: 'Chronological run history' },
];

export default function Stage3Container() {
  const {
    activeTab,
    setActiveTab,
    setMeetings,
    setSettings,
    setVariants,
    setPointsPerBatch,
  } = useStage3Store();

  useEffect(() => {
    // Load meetings
    fetchMeetingsForTraining()
      .then(setMeetings)
      .catch(console.warn);

    // Load persisted settings
    loadStage3Settings()
      .then((s) => {
        setSettings(s);
        if (s.points_per_batch) setPointsPerBatch(s.points_per_batch);
      })
      .catch(console.warn);

    // Load variants
    listStage3Variants()
      .then((res) => setVariants(res.variants || []))
      .catch(console.warn);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minHeight: 0 }}>
      {/* Stage 3 sub-nav */}
      <div style={{
        display: 'flex',
        gap: '2px',
        background: 'hsl(var(--muted) / 0.4)',
        borderRadius: '10px',
        padding: '4px',
        width: 'fit-content',
        border: '1px solid hsl(var(--border) / 0.5)',
      }}>
        {STAGE3_TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '7px',
                padding: '0.45rem 1rem',
                borderRadius: '7px',
                border: 'none',
                background: isActive ? 'hsl(var(--card))' : 'transparent',
                color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                fontWeight: isActive ? 700 : 500,
                fontSize: '0.82rem',
                cursor: 'pointer',
                transition: 'all 0.12s ease',
                boxShadow: isActive ? '0 1px 4px rgba(0,0,0,0.08)' : 'none',
                fontFamily: 'Inter, sans-serif',
              }}
            >
              <Icon size={14} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Panels */}
      <div>
        {activeTab === 'dataset' && <S3DatasetTab />}
        {activeTab === 'training_loop' && <S3TrainingLoopTab />}
        {activeTab === 'variants' && <S3VariantsTab />}
        {activeTab === 'history' && <S3HistoryTab />}
      </div>
    </div>
  );
}
