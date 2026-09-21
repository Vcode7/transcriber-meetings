import { useEffect } from 'react';
import { Database, Zap, BarChart3, History } from 'lucide-react';
import { useStage2Store } from '../../store/stage2Store';
import { fetchMeetingsForTraining } from '../../api/trainingApi';
import { listStage2Variants, loadStage2Settings } from '../../api/stage2Api';
import type { Stage2Tab } from '../../store/stage2Store';
import S2DatasetTab from './S2DatasetTab';
import S2TrainingLoopTab from './S2TrainingLoopTab';
import S2VariantsTab from './S2VariantsTab';
import S2HistoryTab from './S2HistoryTab';

const STAGE2_TABS: { id: Stage2Tab; label: string; icon: React.ElementType; desc: string }[] = [
  { id: 'dataset', label: 'Dataset', icon: Database, desc: 'Load Stage 1 points & configure context' },
  { id: 'training_loop', label: 'Training Loop', icon: Zap, desc: 'Optimize, validate & retrain' },
  { id: 'variants', label: 'Evaluation / Variants', icon: BarChart3, desc: 'Compare all DSPy variants' },
  { id: 'history', label: 'History', icon: History, desc: 'Chronological run history' },
];

export default function Stage2Container() {
  const {
    activeTab, setActiveTab,
    setMeetings, setSettings, setVariants, setPointsPerGroup, setContextRetrieval,
  } = useStage2Store();

  useEffect(() => {
    // Load meetings
    fetchMeetingsForTraining()
      .then(setMeetings)
      .catch(console.warn);

    // Load persisted settings
    loadStage2Settings()
      .then((s) => {
        setSettings(s);
        if (s.points_per_group) setPointsPerGroup(s.points_per_group);
        if (s.context_retrieval !== undefined) setContextRetrieval(s.context_retrieval);
      })
      .catch(console.warn);

    // Load variants
    listStage2Variants()
      .then((res) => setVariants(res.variants || []))
      .catch(console.warn);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minHeight: 0 }}>
      {/* Stage 2 sub-nav */}
      <div style={{
        display: 'flex',
        gap: '2px',
        background: 'hsl(var(--muted) / 0.4)',
        borderRadius: '10px',
        padding: '4px',
        width: 'fit-content',
      }}>
        {STAGE2_TABS.map((tab) => {
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
        {activeTab === 'dataset' && <S2DatasetTab />}
        {activeTab === 'training_loop' && <S2TrainingLoopTab />}
        {activeTab === 'variants' && <S2VariantsTab />}
        {activeTab === 'history' && <S2HistoryTab />}
      </div>
    </div>
  );
}
