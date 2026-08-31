import { Brain, Layers, Settings, Sparkles } from 'lucide-react';
import { useState } from 'react';
import Stage1Container from './stage1/Stage1Container';
import Stage2Container from './stage2/Stage2Container';
import Stage3Container from './stage3/Stage3Container';
import SettingsTab from './SettingsTab';

type TopTab = 'stage1' | 'stage2' | 'stage3' | 'settings';

const TOP_TABS: { id: TopTab; label: string; icon: React.ElementType }[] = [
  { id: 'stage1', label: 'Stage 1', icon: Layers },
  { id: 'stage2', label: 'Stage 2', icon: Layers },
  { id: 'stage3', label: 'Stage 3', icon: Layers },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function TrainingLayout() {
  const [activeTab, setActiveTab] = useState<TopTab>('stage1');

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
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div style={{
        background: 'linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--background)) 100%)',
        borderBottom: '1px solid hsl(var(--border))',
        padding: '1.5rem 2rem 0',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
          <div style={{
            width: 40, height: 40, borderRadius: '12px',
            background: 'linear-gradient(135deg, hsl(var(--accent)) 0%, hsl(262 80% 65%) 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px hsl(var(--accent) / .3)', flexShrink: 0,
          }}>
            <Brain size={20} color="white" />
          </div>
          <div>
            <h1 style={{
              margin: 0, fontSize: '1.4rem', fontWeight: 800,
              letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8,
            }}>
              Training &amp; Optimization
              <Sparkles size={16} style={{ color: 'hsl(var(--accent))' }} />
            </h1>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: 2 }}>
              Stage-oriented DSPy training workflow for the MoM pipeline
            </p>
          </div>
        </div>

        {/* Top-level tab bar */}
        <div style={{ display: 'flex', gap: '2px' }}>
          {TOP_TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            const Icon = tab.icon;
            const isStage = tab.id.startsWith('stage');
            const stageNum = isStage ? tab.id.replace('stage', '') : null;

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
                {isStage ? (
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    width: 20, height: 20, borderRadius: '6px', fontSize: '0.72rem', fontWeight: 800,
                    background: isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                    color: isActive ? 'white' : 'hsl(var(--muted-foreground))',
                    flexShrink: 0,
                  }}>
                    {stageNum}
                  </span>
                ) : (
                  <Icon size={14} />
                )}
                {isStage ? `Stage ${stageNum}` : 'Settings'}
                {(tab.id === 'stage1' || tab.id === 'stage2' || tab.id === 'stage3') && (
                  <span style={{
                    fontSize: '0.6rem', fontWeight: 700, padding: '1px 5px', borderRadius: '6px',
                    background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                  }}>
                    ACTIVE
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Tab Content ──────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '1.5rem 2rem 0rem 1.5rem' }}>
        {activeTab === 'stage1' && <Stage1Container />}
        {activeTab === 'stage2' && <Stage2Container />}
        {activeTab === 'stage3' && <Stage3Container />}
        {activeTab === 'settings' && <SettingsTab />}
      </div>
    </div>
  );
}
