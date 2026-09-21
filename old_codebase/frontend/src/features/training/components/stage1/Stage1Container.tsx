import { useEffect } from 'react';
import { Database, Zap, BarChart3, History } from 'lucide-react';
import { useStage1Store } from '../../store/stage1Store';
import { fetchMeetingsForTraining } from '../../api/trainingApi';
import { listStage1Variants, loadStage1Settings } from '../../api/stage1Api';
import type { Stage1Tab } from '../../store/stage1Store';
import S1DatasetTab from './S1DatasetTab';
import S1TrainingLoopTab from './S1TrainingLoopTab';
import S1VariantsTab from './S1VariantsTab';
import S1HistoryTab from './S1HistoryTab';

const STAGE1_TABS: { id: Stage1Tab; label: string; icon: React.ElementType; desc: string }[] = [
  { id: 'dataset', label: 'Dataset', icon: Database, desc: 'Select meeting & prepare windows' },
  { id: 'training_loop', label: 'Training Loop', icon: Zap, desc: 'Optimize, validate & retrain' },
  { id: 'variants', label: 'Evaluation / Variants', icon: BarChart3, desc: 'Compare all DSPy variants' },
  { id: 'history', label: 'History', icon: History, desc: 'Chronological run history' },
];

export default function Stage1Container() {
  const {
    activeTab, setActiveTab,
    setMeetings, setSettings, setVariants, setWindowSizeMinutes, setWindowOverlapSeconds,
  } = useStage1Store();

  useEffect(() => {
    // Load meetings
    fetchMeetingsForTraining()
      .then(setMeetings)
      .catch(console.warn);

    // Load persisted settings
    loadStage1Settings()
      .then((s) => {
        setSettings(s);
        setWindowSizeMinutes(s.default_window_size_minutes ?? 2);
        setWindowOverlapSeconds(s.default_window_overlap_seconds ?? 30);
      })
      .catch(console.warn);

    // Load variants
    listStage1Variants()
      .then(setVariants)
      .catch(console.warn);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minHeight: 0 }}>
      {/* Stage 1 sub-nav */}
      <div style={{
        display: 'flex',
        gap: '2px',
        background: 'hsl(var(--muted) / 0.4)',
        borderRadius: '10px',
        padding: '4px',
        width: 'fit-content',
      }}>
        {STAGE1_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              title={tab.desc}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '0.45rem 0.9rem',
                borderRadius: '8px',
                border: 'none',
                background: isActive ? 'hsl(var(--card))' : 'transparent',
                color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                cursor: 'pointer',
                fontSize: '0.82rem',
                fontWeight: isActive ? 700 : 500,
                boxShadow: isActive ? '0 1px 4px hsl(0 0% 0% / 0.08)' : 'none',
                transition: 'all 0.15s ease',
                whiteSpace: 'nowrap',
                fontFamily: 'Inter, sans-serif',
              }}
            >
              <Icon size={13} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, minHeight: 0 }}>
        {activeTab === 'dataset' && <S1DatasetTab />}
        {activeTab === 'training_loop' && <S1TrainingLoopTab />}
        {activeTab === 'variants' && <S1VariantsTab />}
        {activeTab === 'history' && <S1HistoryTab />}
      </div>
    </div>
  );
}
