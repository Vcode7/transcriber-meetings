import { ShieldOff, Calendar, Mail } from 'lucide-react'

interface LicenseExpiredProps {
  message?: string
}

export default function LicenseExpired({ message }: LicenseExpiredProps) {
  const displayMessage =
    message ||
    'This application license expired on 30 July 2026. Please contact the administrator for a renewed version.'

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 99999,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background:
          'radial-gradient(ellipse at 50% 0%, hsl(220 30% 10%) 0%, hsl(220 20% 6%) 100%)',
        padding: '2rem',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {/* Ambient glow */}
      <div
        style={{
          position: 'absolute',
          top: '15%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '500px',
          height: '300px',
          background:
            'radial-gradient(ellipse, hsl(0 70% 45% / .12) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Card */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          maxWidth: '520px',
          background: 'hsl(220 20% 10% / .95)',
          border: '1.5px solid hsl(0 60% 40% / .3)',
          borderRadius: '20px',
          padding: '2.5rem 2.5rem 2.25rem',
          boxShadow:
            '0 0 0 1px hsl(0 0% 100% / .04), 0 32px 64px hsl(0 0% 0% / .5)',
          backdropFilter: 'blur(24px)',
          textAlign: 'center',
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: '72px',
            height: '72px',
            borderRadius: '18px',
            background: 'hsl(0 60% 40% / .12)',
            border: '2px solid hsl(0 60% 40% / .3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.75rem',
          }}
        >
          <ShieldOff size={32} style={{ color: 'hsl(0 70% 60%)' }} />
        </div>

        {/* Title */}
        <h1
          style={{
            fontSize: '1.35rem',
            fontWeight: 700,
            color: 'hsl(0 5% 95%)',
            marginBottom: '0.65rem',
            letterSpacing: '-0.02em',
            lineHeight: 1.3,
          }}
        >
          Application License Expired
        </h1>

        {/* Divider */}
        <div
          style={{
            width: '48px',
            height: '3px',
            borderRadius: '99px',
            background:
              'linear-gradient(90deg, hsl(0 70% 55%), hsl(20 90% 55%))',
            margin: '0 auto 1.5rem',
          }}
        />

        {/* Message */}
        <p
          style={{
            fontSize: '0.92rem',
            color: 'hsl(220 10% 65%)',
            lineHeight: 1.7,
            marginBottom: '2rem',
          }}
        >
          {displayMessage}
        </p>

        {/* Info pills */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            marginBottom: '2rem',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '0.7rem 1rem',
              background: 'hsl(220 20% 7%)',
              border: '1px solid hsl(220 15% 18%)',
              borderRadius: '10px',
            }}
          >
            <Calendar size={15} style={{ color: 'hsl(0 70% 60%)', flexShrink: 0 }} />
            <span style={{ fontSize: '0.83rem', color: 'hsl(220 10% 55%)' }}>
              License expired on{' '}
              <strong style={{ color: 'hsl(0 5% 85%)' }}>30 July 2026</strong>
            </span>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '0.7rem 1rem',
              background: 'hsl(220 20% 7%)',
              border: '1px solid hsl(220 15% 18%)',
              borderRadius: '10px',
            }}
          >
            <Mail size={15} style={{ color: 'hsl(var(--accent, 210 90% 55%))', flexShrink: 0 }} />
            <span style={{ fontSize: '0.83rem', color: 'hsl(220 10% 55%)' }}>
              Contact your administrator to obtain a renewed version
            </span>
          </div>
        </div>

        {/* Footer note */}
        <p
          style={{
            fontSize: '0.72rem',
            color: 'hsl(220 10% 35%)',
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}
        >
          All features are disabled until the license is renewed
        </p>
      </div>

      {/* App name */}
      <p
        style={{
          marginTop: '1.75rem',
          fontSize: '0.75rem',
          color: 'hsl(220 10% 30%)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}
      >
        VoiceSum
      </p>
    </div>
  )
}
