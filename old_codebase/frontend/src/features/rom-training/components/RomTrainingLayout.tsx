import { useState } from 'react';
import { BookMarked, Database, Brain, History, Sparkles } from 'lucide-react';
import DataCreationTab from './DataCreationTab';
import DatabaseTab from './DatabaseTab';
import TrainingTab from './TrainingTab';
import HistoryTab from './HistoryTab';

type ActiveTab = 'data-creation' | 'database' | 'training' | 'history';

const TABS: { id: ActiveTab; label: string; icon: React.ElementType }[] = [
  { id: 'data-creation', label: 'Data Creation', icon: BookMarked },
  { id: 'database', label: 'Database', icon: Database },
  { id: 'training', label: 'Training', icon: Brain },
  { id: 'history', label: 'History', icon: History },
];

export default function RomTrainingLayout() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('data-creation');

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minHeight: 0,
      width: '100%',
      background: 'hsl(var(--background))',
      color: 'hsl(var(--foreground))',
      fontFamily: 'Inter, sans-serif',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--background)) 100%)',
        borderBottom: '1px solid hsl(var(--border))',
        padding: '1.5rem 2rem 0',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
          <div style={{
            width: 40, height: 40, borderRadius: '12px',
            background: 'linear-gradient(135deg, hsl(280 80% 60%) 0%, hsl(var(--accent)) 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px hsl(280 80% 60% / .3)', flexShrink: 0,
          }}>
            <BookMarked size={20} color="white" />
          </div>
          <div>
            <h1 style={{
              margin: 0, fontSize: '1.4rem', fontWeight: 800,
              letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8,
            }}>
              ROM Training
              <Sparkles size={16} style={{ color: 'hsl(var(--accent))' }} />
            </h1>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: 2 }}>
              Train models on manually written MoM data for improved Short ROM generation
            </p>
          </div>
        </div>

        {/* Tab bar */}
        <div style={{ display: 'flex', gap: '2px' }}>
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '7px',
                  padding: '0.55rem 1.25rem',
                  background: isActive ? 'hsl(var(--accent) / .12)' : 'transparent',
                  border: 'none',
                  borderBottom: isActive ? '2px solid hsl(var(--accent))' : '2px solid transparent',
                  color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                  cursor: 'pointer',
                  borderRadius: '8px 8px 0 0',
                  fontSize: '0.875rem',
                  fontWeight: isActive ? 700 : 500,
                  whiteSpace: 'nowrap',
                  transition: 'all 0.15s ease',
                  fontFamily: 'Inter, sans-serif',
                }}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '1.5rem 2rem 1.5rem 1.5rem' }}>
        {activeTab === 'data-creation' && <DataCreationTab />}
        {activeTab === 'database' && <DatabaseTab />}
        {activeTab === 'training' && <TrainingTab />}
        {activeTab === 'history' && <HistoryTab />}
      </div>
    </div>
  );
}
