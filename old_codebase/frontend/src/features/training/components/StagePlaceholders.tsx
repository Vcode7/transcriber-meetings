import { FlaskConical } from 'lucide-react';

function PlaceholderContent({ stage, description }: { stage: string; description: string }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      minHeight: '320px', gap: '1rem', color: 'hsl(var(--muted-foreground))',
      background: 'hsl(var(--card))', borderRadius: '16px',
      border: '1px dashed hsl(var(--border))',
      padding: '3rem',
    }}>
      <div style={{
        width: 56, height: 56, borderRadius: '16px',
        background: 'hsl(var(--muted))',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <FlaskConical size={26} style={{ opacity: 0.4 }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
          {stage}
        </h2>
        <p style={{ margin: 0, fontSize: '0.88rem', maxWidth: '360px', lineHeight: '1.6' }}>
          {description}
        </p>
      </div>
    </div>
  );
}

export function Stage2Placeholder() {
  return (
    <PlaceholderContent
      stage="Stage 2"
      description="Training workflow will be added later."
    />
  );
}

export function Stage3Placeholder() {
  return (
    <PlaceholderContent
      stage="Stage 3"
      description="Training workflow will be added later."
    />
  );
}
