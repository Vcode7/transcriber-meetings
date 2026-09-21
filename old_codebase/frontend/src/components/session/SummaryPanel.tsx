import { FileText, Lightbulb, CheckSquare, Users, Sparkles } from 'lucide-react'

interface SummaryPanelProps {
  summary?: string
  keyPoints?: string[]
  actionItems?: string[]
  speakers?: Array<{ name: string; percentage: number }>
}

export default function SummaryPanel({ summary, keyPoints, actionItems, speakers }: SummaryPanelProps) {
  if (!summary && !keyPoints && !actionItems && !speakers) {
    return (
      <div
        className="glass"
        style={{
          padding: '3rem 2rem',
          textAlign: 'center',
          color: 'hsl(var(--pencil))',
        }}
      >
        <FileText 
          size={48} 
          style={{ 
            margin: '0 auto 1.5rem', 
            display: 'block', 
            opacity: 0.3,
            color: 'hsl(var(--accent))'
          }} 
          className="animate-float"
        />
        <p style={{ 
          fontSize: '.95rem',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500
        }}>
          No summary available yet
        </p>
        <p style={{ 
          fontSize: '.82rem',
          marginTop: '.5rem',
          opacity: .7
        }}>
          Complete a recording to see AI insights
        </p>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {summary && (
        <div className="glass animate-slide-up" style={{ 
          padding: '1.4rem',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: '-20px',
            right: '-20px',
            width: '80px',
            height: '80px',
            background: 'hsl(var(--accent) / .05)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            marginBottom: '14px',
            position: 'relative'
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'hsl(var(--accent) / .15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px solid hsl(var(--accent) / .3)'
            }}>
              <FileText size={16} style={{ color: 'hsl(var(--accent))' }} />
            </div>
            <h3 style={{ 
              fontSize: '1.05rem', 
              fontWeight: 700,
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              Summary
            </h3>
            <Sparkles size={14} style={{ color: 'hsl(var(--accent))', opacity: .6 }} />
          </div>
          <p style={{ 
            fontSize: '0.9rem', 
            lineHeight: 1.75, 
            color: 'hsl(var(--ink-soft))',
            fontFamily: 'Inter, sans-serif',
            position: 'relative'
          }}>
            {summary}
          </p>
        </div>
      )}

      {keyPoints && keyPoints.length > 0 && (
        <div className="glass animate-slide-up" style={{ 
          padding: '1.4rem',
          animationDelay: '0.1s',
          animationFillMode: 'both'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            marginBottom: '14px' 
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'hsl(var(--sticky-yellow) / .4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px solid hsl(var(--ink) / .2)'
            }}>
              <Lightbulb size={16} style={{ color: 'hsl(var(--ink))' }} />
            </div>
            <h3 style={{ 
              fontSize: '1.05rem', 
              fontWeight: 700,
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              Key Points
            </h3>
          </div>
          <ul style={{ 
            paddingLeft: '1.5rem', 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '10px' 
          }}>
            {keyPoints.map((point, idx) => (
              <li 
                key={idx} 
                className="animate-slide-up"
                style={{ 
                  fontSize: '0.88rem', 
                  lineHeight: 1.7, 
                  color: 'hsl(var(--ink-soft))',
                  fontFamily: 'Inter, sans-serif',
                  animationDelay: `${0.1 + idx * 0.05}s`,
                  animationFillMode: 'both'
                }}
              >
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {actionItems && actionItems.length > 0 && (
        <div className="glass animate-slide-up" style={{ 
          padding: '1.4rem',
          animationDelay: '0.2s',
          animationFillMode: 'both'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            marginBottom: '14px' 
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'hsl(var(--sticky-green) / .4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px solid hsl(var(--ink) / .2)'
            }}>
              <CheckSquare size={16} style={{ color: 'hsl(var(--ink))' }} />
            </div>
            <h3 style={{ 
              fontSize: '1.05rem', 
              fontWeight: 700,
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              Action Items
            </h3>
          </div>
          <ul style={{ 
            paddingLeft: '1.5rem', 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '10px' 
          }}>
            {actionItems.map((item, idx) => (
              <li 
                key={idx}
                className="animate-slide-up"
                style={{ 
                  fontSize: '0.88rem', 
                  lineHeight: 1.7, 
                  color: 'hsl(var(--ink-soft))',
                  fontFamily: 'Inter, sans-serif',
                  animationDelay: `${0.2 + idx * 0.05}s`,
                  animationFillMode: 'both'
                }}
              >
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}

      {speakers && speakers.length > 0 && (
        <div className="glass animate-slide-up" style={{ 
          padding: '1.4rem',
          animationDelay: '0.3s',
          animationFillMode: 'both'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            marginBottom: '14px' 
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'hsl(var(--sticky-blue) / .4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '2px solid hsl(var(--ink) / .2)'
            }}>
              <Users size={16} style={{ color: 'hsl(var(--ink))' }} />
            </div>
            <h3 style={{ 
              fontSize: '1.05rem', 
              fontWeight: 700,
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              Speakers Overview
            </h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {speakers.map((speaker, idx) => (
              <div 
                key={idx}
                className="animate-slide-up"
                style={{
                  animationDelay: `${0.3 + idx * 0.05}s`,
                  animationFillMode: 'both'
                }}
              >
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  marginBottom: '8px' 
                }}>
                  <span style={{ 
                    fontSize: '0.88rem', 
                    fontWeight: 600, 
                    color: 'hsl(var(--ink))',
                    fontFamily: 'Inter, sans-serif'
                  }}>
                    {speaker.name}
                  </span>
                  <span style={{ 
                    fontSize: '0.88rem', 
                    color: 'hsl(var(--pencil))',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontWeight: 600
                  }}>
                    {speaker.percentage}%
                  </span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{
                      width: `${speaker.percentage}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
