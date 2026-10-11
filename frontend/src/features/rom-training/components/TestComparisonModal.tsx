import React, { useState, useEffect, useRef } from 'react';
import {
  X, ChevronLeft, ChevronRight, Copy, Check,
  FileText, Sparkles, UserCheck, Layers, Code, Link2, Unlink
} from 'lucide-react';
import type { TestResult } from '../types/romTrainingTypes';

interface TestComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  results: TestResult[];
  initialIndex?: number;
  variantName?: string;
  accuracySummary?: string | null;
}

export default function TestComparisonModal({
  isOpen,
  onClose,
  results,
  initialIndex = 0,
  variantName = 'Variant',
  accuracySummary,
}: TestComparisonModalProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [copiedCol, setCopiedCol] = useState<'before' | 'after' | 'manual' | null>(null);
  const [viewMode, setViewMode] = useState<'formatted' | 'raw'>('formatted');
  const [layoutMode, setLayoutMode] = useState<'columns' | 'tabs'>('columns');
  const [activeTab, setActiveTab] = useState<'after' | 'before' | 'manual'>('after');
  const [syncScroll, setSyncScroll] = useState(false);

  // References for column scrolling
  const beforeColRef = useRef<HTMLDivElement>(null);
  const afterColRef = useRef<HTMLDivElement>(null);
  const manualColRef = useRef<HTMLDivElement>(null);
  const isSyncingRef = useRef(false);

  useEffect(() => {
    setCurrentIndex(initialIndex);
  }, [initialIndex, isOpen]);

  // Reset scroll to top when sample changes
  useEffect(() => {
    if (beforeColRef.current) beforeColRef.current.scrollTop = 0;
    if (afterColRef.current) afterColRef.current.scrollTop = 0;
    if (manualColRef.current) manualColRef.current.scrollTop = 0;
  }, [currentIndex]);

  // Handle synchronized scrolling across the 3 columns
  const handleScroll = (source: 'before' | 'after' | 'manual') => (e: React.UIEvent<HTMLDivElement>) => {
    if (!syncScroll || isSyncingRef.current || layoutMode !== 'columns') return;
    const target = e.currentTarget;
    const maxScroll = target.scrollHeight - target.clientHeight;
    if (maxScroll <= 0) return;
    const scrollRatio = target.scrollTop / maxScroll;

    isSyncingRef.current = true;
    requestAnimationFrame(() => {
      if (source !== 'before' && beforeColRef.current) {
        const col = beforeColRef.current;
        col.scrollTop = scrollRatio * (col.scrollHeight - col.clientHeight);
      }
      if (source !== 'after' && afterColRef.current) {
        const col = afterColRef.current;
        col.scrollTop = scrollRatio * (col.scrollHeight - col.clientHeight);
      }
      if (source !== 'manual' && manualColRef.current) {
        const col = manualColRef.current;
        col.scrollTop = scrollRatio * (col.scrollHeight - col.clientHeight);
      }
      isSyncingRef.current = false;
    });
  };

  // Handle keyboard shortcuts (Esc to close, Arrow keys to navigate)
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowLeft') {
        setCurrentIndex(prev => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowRight') {
        setCurrentIndex(prev => Math.min(results.length - 1, prev + 1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results.length, onClose]);

  if (!isOpen || !results || results.length === 0) return null;

  const currentSample = results[currentIndex] || results[0];

  const handleCopy = (text: string, col: 'before' | 'after' | 'manual') => {
    navigator.clipboard.writeText(text);
    setCopiedCol(col);
    setTimeout(() => setCopiedCol(null), 1800);
  };

  const getWordCount = (text: string) => {
    if (!text) return 0;
    return text.trim().split(/\s+/).filter(Boolean).length;
  };

  // Helper to parse JSON array of strings if present
  const parsePoints = (text: string): string[] | null => {
    if (!text) return null;
    const trimmed = text.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed) && parsed.every(p => typeof p === 'string')) {
          return parsed;
        }
      } catch {
        // Not valid JSON
      }
    }
    // Check for lines starting with "- " or numbered bullets
    const lines = trimmed.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length > 1 && lines.every(l => /^[-*•\d+.]\s+/.test(l))) {
      return lines.map(l => l.replace(/^[-*•\d+.]\s+/, ''));
    }
    return null;
  };

  const renderContent = (text: string, accentColor: string) => {
    if (!text) {
      return (
        <div style={{ color: 'hsl(var(--muted-foreground))', fontStyle: 'italic', padding: '1rem' }}>
          No content generated or sample empty.
        </div>
      );
    }

    if (viewMode === 'formatted') {
      const points = parsePoints(text);
      if (points && points.length > 0) {
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {points.map((pt, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  gap: '0.65rem',
                  alignItems: 'flex-start',
                  padding: '0.65rem 0.85rem',
                  borderRadius: '8px',
                  background: 'hsl(var(--background) / 0.75)',
                  border: '1px solid hsl(var(--border) / 0.85)',
                  fontSize: '0.82rem',
                  lineHeight: '1.5',
                }}
              >
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '22px',
                    height: '22px',
                    borderRadius: '6px',
                    background: `${accentColor}18`,
                    color: accentColor,
                    fontWeight: 700,
                    fontSize: '0.72rem',
                    flexShrink: 0,
                    marginTop: '2px',
                  }}
                >
                  {idx + 1}
                </span>
                <span style={{ color: 'hsl(var(--foreground))', flex: 1, wordBreak: 'break-word' }}>
                  {pt}
                </span>
              </div>
            ))}
          </div>
        );
      }
    }

    return (
      <div
        style={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontFamily: viewMode === 'raw' ? 'monospace' : 'inherit',
          fontSize: viewMode === 'raw' ? '0.78rem' : '0.82rem',
          lineHeight: '1.55',
          color: 'hsl(var(--foreground))',
          padding: '0.4rem',
        }}
      >
        {text}
      </div>
    );
  };

  return (
    <>
      <style>{`
        .comparison-scrollable {
          scrollbar-width: thin;
          scrollbar-color: hsl(var(--muted-foreground) / 0.4) transparent;
        }
        .comparison-scrollable::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .comparison-scrollable::-webkit-scrollbar-track {
          background: hsl(var(--muted) / 0.15);
          border-radius: 4px;
        }
        .comparison-scrollable::-webkit-scrollbar-thumb {
          background: hsl(var(--muted-foreground) / 0.35);
          border-radius: 4px;
        }
        .comparison-scrollable::-webkit-scrollbar-thumb:hover {
          background: hsl(var(--muted-foreground) / 0.6);
        }
      `}</style>

      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 9999,
          background: 'rgba(0, 0, 0, 0.78)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.25rem',
          overflow: 'auto',
        }}
        onClick={onClose}
      >
        <div
          style={{
            width: '95vw',
            maxWidth: '1600px',
            height: '92vh',
            maxHeight: '1000px',
            background: 'hsl(var(--card))',
            borderRadius: '16px',
            border: '1px solid hsl(var(--border))',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
          }}
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div
            style={{
              padding: '0.85rem 1.25rem',
              borderBottom: '1px solid hsl(var(--border))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'hsl(var(--muted) / 0.3)',
              flexShrink: 0,
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', minWidth: 0, flex: 1 }}>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  padding: '3px 9px',
                  borderRadius: '6px',
                  background: 'hsl(var(--accent) / 0.15)',
                  color: 'hsl(var(--accent))',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  flexShrink: 0,
                }}
              >
                {variantName}
              </span>
              <div style={{ minWidth: 0, overflow: 'hidden' }}>
                <h2
                  style={{
                    fontSize: '1rem',
                    fontWeight: 700,
                    margin: 0,
                    color: 'hsl(var(--foreground))',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={currentSample.agenda_title || `Sample ${currentIndex + 1}`}
                >
                  <span>{currentSample.agenda_title || `Sample ${currentIndex + 1}`}</span>
                </h2>
                {currentSample.recording_title && (
                  <div style={{ fontSize: '0.74rem', color: 'hsl(var(--muted-foreground))', marginTop: 1, textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }}>
                    Meeting: {currentSample.recording_title}
                  </div>
                )}
              </div>
            </div>

            {/* Controls: Navigator + Sync + Toggles + Close */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
              {/* Sample Navigation */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                  background: 'hsl(var(--background))',
                  borderRadius: '8px',
                  border: '1px solid hsl(var(--border))',
                  padding: '2px',
                }}
              >
                <button
                  onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))}
                  disabled={currentIndex === 0}
                  style={{
                    padding: '4px 7px',
                    borderRadius: '5px',
                    border: 'none',
                    background: 'transparent',
                    color: currentIndex === 0 ? 'hsl(var(--muted-foreground))' : 'hsl(var(--foreground))',
                    cursor: currentIndex === 0 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  title="Previous sample (Left Arrow)"
                >
                  <ChevronLeft size={16} />
                </button>
                <span
                  style={{
                    fontSize: '0.74rem',
                    fontWeight: 600,
                    padding: '0 6px',
                    color: 'hsl(var(--foreground))',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {currentIndex + 1} / {results.length}
                </span>
                <button
                  onClick={() => setCurrentIndex(prev => Math.min(results.length - 1, prev + 1))}
                  disabled={currentIndex === results.length - 1}
                  style={{
                    padding: '4px 7px',
                    borderRadius: '5px',
                    border: 'none',
                    background: 'transparent',
                    color: currentIndex === results.length - 1 ? 'hsl(var(--muted-foreground))' : 'hsl(var(--foreground))',
                    cursor: currentIndex === results.length - 1 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  title="Next sample (Right Arrow)"
                >
                  <ChevronRight size={16} />
                </button>
              </div>

              {/* Layout switch: Columns vs Tabs */}
              <div
                style={{
                  display: 'flex',
                  background: 'hsl(var(--background))',
                  borderRadius: '8px',
                  border: '1px solid hsl(var(--border))',
                  padding: '2px',
                }}
              >
                <button
                  onClick={() => setLayoutMode('columns')}
                  style={{
                    padding: '4px 9px',
                    borderRadius: '6px',
                    border: 'none',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: layoutMode === 'columns' ? 'hsl(var(--accent) / 0.15)' : 'transparent',
                    color: layoutMode === 'columns' ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                  }}
                >
                  Side-by-Side
                </button>
                <button
                  onClick={() => setLayoutMode('tabs')}
                  style={{
                    padding: '4px 9px',
                    borderRadius: '6px',
                    border: 'none',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: layoutMode === 'tabs' ? 'hsl(var(--accent) / 0.15)' : 'transparent',
                    color: layoutMode === 'tabs' ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                  }}
                >
                  Single Tab
                </button>
              </div>

              {/* Synchronized Scrolling Toggle (only relevant in columns mode) */}
              {layoutMode === 'columns' && (
                <button
                  onClick={() => setSyncScroll(prev => !prev)}
                  style={{
                    padding: '4px 9px',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: syncScroll ? 'hsl(var(--accent))' : 'hsl(var(--border))',
                    background: syncScroll ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--background))',
                    color: syncScroll ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                  }}
                  title={syncScroll ? 'Sync Scroll is ON: Scrolling one column scrolls all 3 simultaneously' : 'Sync Scroll is OFF: Click to scroll all 3 columns together'}
                >
                  {syncScroll ? <Link2 size={13} /> : <Unlink size={13} />}
                  <span>{syncScroll ? 'Sync Scroll: ON' : 'Sync Scroll'}</span>
                </button>
              )}

              {/* Formatted vs Raw switch */}
              <button
                onClick={() => setViewMode(prev => prev === 'formatted' ? 'raw' : 'formatted')}
                style={{
                  padding: '4px 9px',
                  borderRadius: '8px',
                  border: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}
                title="Toggle between structured point rendering and raw JSON"
              >
                {viewMode === 'formatted' ? <Code size={13} /> : <FileText size={13} />}
                {viewMode === 'formatted' ? 'Raw Text' : 'Formatted'}
              </button>

              {/* Close Button */}
              <button
                onClick={onClose}
                style={{
                  padding: '6px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'hsl(var(--muted) / 0.5)',
                  color: 'hsl(var(--foreground))',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                title="Close modal (Esc)"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Tab selector for Tabs mode */}
          {layoutMode === 'tabs' && (
            <div
              style={{
                display: 'flex',
                gap: 8,
                padding: '0.55rem 1.25rem',
                borderBottom: '1px solid hsl(var(--border))',
                background: 'hsl(var(--background) / 0.5)',
                flexShrink: 0,
              }}
            >
              <button
                onClick={() => setActiveTab('after')}
                style={{
                  padding: '0.38rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: activeTab === 'after' ? '#22c55e' : 'hsl(var(--border))',
                  background: activeTab === 'after' ? 'rgba(34, 197, 94, 0.12)' : 'transparent',
                  color: activeTab === 'after' ? '#22c55e' : 'hsl(var(--muted-foreground))',
                  fontWeight: 700,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}
              >
                <Sparkles size={13} />
                AFTER (Trained Variant)
              </button>
              <button
                onClick={() => setActiveTab('before')}
                style={{
                  padding: '0.38rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: activeTab === 'before' ? '#94a3b8' : 'hsl(var(--border))',
                  background: activeTab === 'before' ? 'rgba(148, 163, 184, 0.15)' : 'transparent',
                  color: activeTab === 'before' ? '#94a3b8' : 'hsl(var(--muted-foreground))',
                  fontWeight: 700,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}
              >
                <Layers size={13} />
                BEFORE (Base Model)
              </button>
              <button
                onClick={() => setActiveTab('manual')}
                style={{
                  padding: '0.38rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid',
                  borderColor: activeTab === 'manual' ? '#a855f7' : 'hsl(var(--border))',
                  background: activeTab === 'manual' ? 'rgba(168, 85, 247, 0.12)' : 'transparent',
                  color: activeTab === 'manual' ? '#a855f7' : 'hsl(var(--muted-foreground))',
                  fontWeight: 700,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}
              >
                <UserCheck size={13} />
                MANUAL (Ground Truth Target)
              </button>
            </div>
          )}

          {/* Content Area - Scrollable columns container with strict minHeight: 0 constraint */}
          <div
            style={{
              flex: '1 1 0%',
              minHeight: 0,
              minWidth: 0,
              overflow: 'hidden',
              display: 'grid',
              gridTemplateColumns: layoutMode === 'columns' ? 'repeat(3, minmax(0, 1fr))' : 'minmax(0, 1fr)',
              background: 'hsl(var(--card))',
            }}
          >
            {/* Column 1: Base Model (Before) */}
            {(layoutMode === 'columns' || activeTab === 'before') && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  minHeight: 0,
                  minWidth: 0,
                  overflow: 'hidden',
                  borderRight: layoutMode === 'columns' ? '1px solid hsl(var(--border))' : 'none',
                  background: 'hsl(var(--card))',
                }}
              >
                {/* Column Header */}
                <div
                  style={{
                    padding: '0.7rem 1rem',
                    borderBottom: '1px solid hsl(var(--border))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'hsl(var(--muted) / 0.15)',
                    flexShrink: 0,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Layers size={15} style={{ color: '#94a3b8' }} />
                    <span style={{ fontWeight: 700, fontSize: '0.8rem', color: '#94a3b8' }}>
                      BEFORE (Base Model)
                    </span>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        padding: '1px 6px',
                        borderRadius: 4,
                        background: 'rgba(148, 163, 184, 0.15)',
                        color: '#94a3b8',
                      }}
                    >
                      {getWordCount(currentSample.before)} words
                    </span>
                  </div>
                  <button
                    onClick={() => handleCopy(currentSample.before, 'before')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '3px 8px',
                      borderRadius: 6,
                      border: '1px solid hsl(var(--border))',
                      background: 'hsl(var(--background))',
                      color: 'hsl(var(--foreground))',
                      fontSize: '0.7rem',
                      cursor: 'pointer',
                    }}
                    title="Copy to clipboard"
                  >
                    {copiedCol === 'before' ? <Check size={12} color="#22c55e" /> : <Copy size={12} />}
                    {copiedCol === 'before' ? 'Copied!' : 'Copy'}
                  </button>
                </div>

                {/* Column Body - Vertically Scrollable with thin scrollbar */}
                <div
                  ref={beforeColRef}
                  onScroll={handleScroll('before')}
                  className="comparison-scrollable"
                  style={{
                    flex: '1 1 0%',
                    minHeight: 0,
                    minWidth: 0,
                    padding: '1.1rem',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                  }}
                >
                  {renderContent(currentSample.before, '#94a3b8')}
                </div>
              </div>
            )}

            {/* Column 2: Trained Variant (After) */}
            {(layoutMode === 'columns' || activeTab === 'after') && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  minHeight: 0,
                  minWidth: 0,
                  overflow: 'hidden',
                  borderRight: layoutMode === 'columns' ? '1px solid hsl(var(--border))' : 'none',
                  background: 'hsl(var(--card))',
                }}
              >
                {/* Column Header */}
                <div
                  style={{
                    padding: '0.7rem 1rem',
                    borderBottom: '1px solid hsl(var(--border))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'rgba(34, 197, 94, 0.08)',
                    flexShrink: 0,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Sparkles size={15} style={{ color: '#22c55e' }} />
                    <span style={{ fontWeight: 700, fontSize: '0.8rem', color: '#22c55e' }}>
                      AFTER (Trained Variant)
                    </span>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        padding: '1px 6px',
                        borderRadius: 4,
                        background: 'rgba(34, 197, 94, 0.15)',
                        color: '#22c55e',
                      }}
                    >
                      {getWordCount(currentSample.after)} words
                    </span>
                  </div>
                  <button
                    onClick={() => handleCopy(currentSample.after, 'after')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '3px 8px',
                      borderRadius: 6,
                      border: '1px solid hsl(var(--border))',
                      background: 'hsl(var(--background))',
                      color: 'hsl(var(--foreground))',
                      fontSize: '0.7rem',
                      cursor: 'pointer',
                    }}
                    title="Copy to clipboard"
                  >
                    {copiedCol === 'after' ? <Check size={12} color="#22c55e" /> : <Copy size={12} />}
                    {copiedCol === 'after' ? 'Copied!' : 'Copy'}
                  </button>
                </div>

                {/* Column Body - Vertically Scrollable with thin scrollbar */}
                <div
                  ref={afterColRef}
                  onScroll={handleScroll('after')}
                  className="comparison-scrollable"
                  style={{
                    flex: '1 1 0%',
                    minHeight: 0,
                    minWidth: 0,
                    padding: '1.1rem',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                  }}
                >
                  {renderContent(currentSample.after, '#22c55e')}
                </div>
              </div>
            )}

            {/* Column 3: Manual Target */}
            {(layoutMode === 'columns' || activeTab === 'manual') && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  minHeight: 0,
                  minWidth: 0,
                  overflow: 'hidden',
                  background: 'hsl(var(--card))',
                }}
              >
                {/* Column Header */}
                <div
                  style={{
                    padding: '0.7rem 1rem',
                    borderBottom: '1px solid hsl(var(--border))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'rgba(168, 85, 247, 0.08)',
                    flexShrink: 0,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <UserCheck size={15} style={{ color: '#a855f7' }} />
                    <span style={{ fontWeight: 700, fontSize: '0.8rem', color: '#a855f7' }}>
                      MANUAL (Ground Truth Target)
                    </span>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        padding: '1px 6px',
                        borderRadius: 4,
                        background: 'rgba(168, 85, 247, 0.15)',
                        color: '#a855f7',
                      }}
                    >
                      {getWordCount(currentSample.manual)} words
                    </span>
                  </div>
                  <button
                    onClick={() => handleCopy(currentSample.manual, 'manual')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '3px 8px',
                      borderRadius: 6,
                      border: '1px solid hsl(var(--border))',
                      background: 'hsl(var(--background))',
                      color: 'hsl(var(--foreground))',
                      fontSize: '0.7rem',
                      cursor: 'pointer',
                    }}
                    title="Copy to clipboard"
                  >
                    {copiedCol === 'manual' ? <Check size={12} color="#22c55e" /> : <Copy size={12} />}
                    {copiedCol === 'manual' ? 'Copied!' : 'Copy'}
                  </button>
                </div>

                {/* Column Body - Vertically Scrollable with thin scrollbar */}
                <div
                  ref={manualColRef}
                  onScroll={handleScroll('manual')}
                  className="comparison-scrollable"
                  style={{
                    flex: '1 1 0%',
                    minHeight: 0,
                    minWidth: 0,
                    padding: '1.1rem',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                  }}
                >
                  {renderContent(currentSample.manual, '#a855f7')}
                </div>
              </div>
            )}
          </div>

          {/* Footer with summary & sample jump pills */}
          <div
            style={{
              padding: '0.65rem 1.25rem',
              borderTop: '1px solid hsl(var(--border))',
              background: 'hsl(var(--muted) / 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 10,
              flexShrink: 0,
            }}
          >
            <div style={{ fontSize: '0.74rem', color: 'hsl(var(--muted-foreground))' }}>
              {accuracySummary ? (
                <span><strong>Metric:</strong> {accuracySummary}</span>
              ) : (
                <span>Showing sample <strong>{currentIndex + 1}</strong> of <strong>{results.length}</strong></span>
              )}
            </div>

            {/* Sample quick selector pills */}
            <div
              className="comparison-scrollable"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                overflowX: 'auto',
                maxWidth: '65%',
                paddingBottom: 2,
              }}
            >
              {results.map((r, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentIndex(idx)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: '6px',
                    border: '1px solid',
                    borderColor: idx === currentIndex ? 'hsl(var(--accent))' : 'hsl(var(--border))',
                    background: idx === currentIndex ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--background))',
                    color: idx === currentIndex ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                    fontSize: '0.7rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    flexShrink: 0,
                  }}
                >
                  {idx + 1}. {r.agenda_title ? (r.agenda_title.slice(0, 16) + (r.agenda_title.length > 16 ? '…' : '')) : `S${idx + 1}`}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
