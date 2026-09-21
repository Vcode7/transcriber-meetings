export default function LoadingSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="skeleton"
          style={{
            padding: '1.25rem',
            animationDelay: `${i * 0.1}s`,
          }}
        >
          <div
            style={{
              height: '16px',
              background: 'hsl(var(--muted))',
              borderRadius: '4px',
              marginBottom: '8px',
              width: '60%',
            }}
          />
          <div
            style={{
              height: '12px',
              background: 'hsl(var(--muted))',
              borderRadius: '4px',
              width: '40%',
            }}
          />
        </div>
      ))}
    </div>
  )
}
