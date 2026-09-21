import { CheckCircle, Loader } from 'lucide-react'

interface Step {
  label: string
  status: 'pending' | 'active' | 'complete'
}

interface ProcessingStepsProps {
  steps: Step[]
}

export default function ProcessingSteps({ steps }: ProcessingStepsProps) {
  return (
    <div
      className="glass fade-in"
      style={{
        padding: '1.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        maxWidth: '500px',
        margin: '0 auto',
      }}
    >
      <h3 style={{
        fontSize: '1rem', fontWeight: 700, textAlign: 'center',
        color: 'hsl(var(--ink))',
        fontFamily: 'Inter, sans-serif'
      }}>
        Processing Recording
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {steps.map((step, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px',
              background: step.status === 'active' ? 'hsl(var(--accent) / .08)' : 'transparent',
              borderRadius: '10px',
              transition: 'all 0.3s ease',
            }}
          >
            {step.status === 'complete' && (
              <CheckCircle size={20} style={{ color: 'hsl(130, 60%, 45%)', flexShrink: 0 }} />
            )}
            {step.status === 'active' && (
              <Loader size={20} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} className="spin" />
            )}
            {step.status === 'pending' && (
              <div
                style={{
                  width: '20px',
                  height: '20px',
                  borderRadius: '50%',
                  border: '2px solid hsl(var(--ink) / .2)',
                  flexShrink: 0,
                }}
              />
            )}
            <span
              style={{
                fontSize: '0.875rem',
                fontWeight: step.status === 'active' ? 600 : 400,
                color: step.status === 'pending' ? 'hsl(var(--pencil))' : 'hsl(var(--ink))',
                fontFamily: 'Inter, sans-serif'
              }}
            >
              {step.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
