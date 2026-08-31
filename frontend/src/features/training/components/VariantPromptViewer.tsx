import { useState } from 'react';
import { Copy, Check, Code2, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';

interface Props {
  prompt?: string | null;
  variantLabel?: string;
  isDefault?: boolean;
}

export default function VariantPromptViewer({
  prompt,
  variantLabel,
  isDefault,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [expandedText, setExpandedText] = useState(false);

  const hasPrompt = Boolean(prompt && prompt.trim());
  const textToDisplay = prompt?.trim() || '';
  const charCount = textToDisplay.length;
  const wordCount = textToDisplay ? textToDisplay.split(/\s+/).length : 0;
  const isLong = charCount > 500;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!hasPrompt) return;
    navigator.clipboard.writeText(textToDisplay);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      marginTop: '1.25rem',
      background: 'hsl(var(--card))',
      border: '1px solid hsl(var(--border))',
      borderRadius: '10px',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.65rem 1rem',
        background: 'hsl(var(--muted) / 0.35)',
        borderBottom: '1px solid hsl(var(--border) / 0.6)',
        flexWrap: 'wrap',
        gap: '0.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: 24,
            height: 24,
            borderRadius: '6px',
            background: 'hsl(var(--accent) / 0.12)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'hsl(var(--accent))',
          }}>
            <Code2 size={13} />
          </div>
          <div>
            <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
              Exact Variant Prompt &amp; Instructions
            </span>
            {isDefault && (
              <span style={{
                marginLeft: '6px',
                fontSize: '0.65rem',
                fontWeight: 600,
                padding: '1px 6px',
                borderRadius: '4px',
                background: 'hsl(var(--muted))',
                color: 'hsl(var(--muted-foreground))',
              }}>
                Baseline
              </span>
            )}
          </div>
        </div>

        {hasPrompt ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{
              fontSize: '0.68rem',
              color: 'hsl(var(--muted-foreground))',
              fontFamily: 'monospace',
            }}>
              {charCount.toLocaleString()} chars • {wordCount} words
            </span>

            <button
              onClick={handleCopy}
              title="Copy prompt text"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '3px 8px',
                borderRadius: '6px',
                border: '1px solid hsl(var(--border))',
                background: copied ? 'hsl(142 71% 45% / 0.1)' : 'hsl(var(--background))',
                color: copied ? 'hsl(142 71% 45%)' : 'hsl(var(--foreground))',
                fontSize: '0.72rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {copied ? (
                <>
                  <Check size={12} />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        ) : (
          <span style={{
            fontSize: '0.7rem',
            color: 'hsl(var(--muted-foreground))',
            fontStyle: 'italic',
          }}>
            Not saved
          </span>
        )}
      </div>

      {/* Body */}
      {hasPrompt ? (
        <>
          <div style={{
            position: 'relative',
            maxHeight: isLong && !expandedText ? '220px' : 'none',
            overflowY: isLong && !expandedText ? 'hidden' : 'auto',
            background: 'hsl(var(--background) / 0.6)',
          }}>
            <pre style={{
              margin: 0,
              padding: '1rem',
              fontSize: '0.75rem',
              lineHeight: '1.55',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              color: 'hsl(var(--foreground))',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {textToDisplay}
            </pre>

            {isLong && !expandedText && (
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '60px',
                background: 'linear-gradient(to bottom, transparent, hsl(var(--card)))',
                pointerEvents: 'none',
              }} />
            )}
          </div>

          {/* Expand / Collapse toggle for long prompts */}
          {isLong && (
            <div style={{
              padding: '0.4rem',
              textAlign: 'center',
              borderTop: '1px solid hsl(var(--border) / 0.5)',
              background: 'hsl(var(--muted) / 0.15)',
            }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedText(!expandedText);
                }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'hsl(var(--accent))',
                  cursor: 'pointer',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                {expandedText ? (
                  <>
                    <ChevronUp size={12} /> Show Less
                  </>
                ) : (
                  <>
                    <ChevronDown size={12} /> Show Full Prompt ({charCount.toLocaleString()} chars)
                  </>
                )}
              </button>
            </div>
          )}
        </>
      ) : (
        <div style={{
          padding: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: 'hsl(var(--muted-foreground))',
          fontSize: '0.78rem',
          background: 'hsl(var(--background) / 0.4)',
        }}>
          <AlertCircle size={14} style={{ color: 'hsl(var(--muted-foreground))', flexShrink: 0 }} />
          <span>No prompt instructions were saved in the artifact for this variant.</span>
        </div>
      )}
    </div>
  );
}
