import React, { useEffect, useState } from 'react';
import {
  FileText, Sparkles, CheckCircle2, AlertTriangle,
  ArrowLeft, ArrowRight, RefreshCw, Hash, Layers,
  ChevronDown, ChevronUp, Tag
} from 'lucide-react';
import {
  ExtractionResult,
  embeddingTrainingApi
} from '../../../services/embeddingTrainingApi';

interface TextExtractionViewProps {
  files: File[];
  extractionResult: ExtractionResult | null;
  onExtractionComplete: (result: ExtractionResult) => void;
  onBack: () => void;
  onNext: () => void;
}

export default function TextExtractionView({
  files,
  extractionResult,
  onExtractionComplete,
  onBack,
  onNext,
}: TextExtractionViewProps) {
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSampleChunks, setShowSampleChunks] = useState(false);

  const runExtraction = async () => {
    if (files.length === 0) return;
    setExtracting(true);
    setError(null);
    try {
      const result = await embeddingTrainingApi.extractDocuments(files);
      onExtractionComplete(result);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to extract text from documents.');
    } finally {
      setExtracting(false);
    }
  };

  useEffect(() => {
    if (!extractionResult && files.length > 0) {
      runExtraction();
    }
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
            Step 3: Extract & Normalize Text
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
            Extracting text from all uploaded documents, normalizing whitespace, removing duplicate passages, and mining ADA domain terminology.
          </p>
        </div>

        {extractionResult && !extracting && (
          <button
            onClick={runExtraction}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: 'hsl(var(--muted) / 0.5)', color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))', padding: '0.45rem 0.9rem',
              borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
            }}
          >
            <RefreshCw size={13} /> Re-extract
          </button>
        )}
      </div>

      {extracting && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: '3.5rem 2rem', background: 'hsl(var(--card))', borderRadius: '16px',
          border: '1px solid hsl(var(--border))', gap: '1rem', textAlign: 'center',
        }}>
          <div className="spin" style={{
            width: 36, height: 36, border: '3px solid hsl(var(--accent) / 0.2)',
            borderTop: '3px solid hsl(var(--accent))', borderRadius: '50%',
          }} />
          <div>
            <h4 style={{ margin: '0 0 0.4rem', fontSize: '1.05rem', fontWeight: 700 }}>
              Extracting & Normalizing Documents...
            </h4>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>
              Parsing {files.length} document{files.length > 1 ? 's' : ''}, cleaning text, and deduplicating chunks.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: 'hsl(0 80% 50% / 0.1)', color: 'hsl(0 80% 65%)',
          border: '1px solid hsl(0 80% 50% / 0.25)', padding: '1rem',
          borderRadius: '12px', fontSize: '0.875rem',
        }}>
          <AlertTriangle size={18} />
          <div>
            <strong>Extraction Error:</strong> {error}
          </div>
        </div>
      )}

      {extractionResult && !extracting && (
        <>
          {/* Statistics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{
              background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
              borderRadius: '12px', padding: '1.15rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
                <FileText size={15} /> Documents Processed
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                {extractionResult.successful_documents}
                <span style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', fontWeight: 500 }}>
                  /{extractionResult.total_documents}
                </span>
              </div>
            </div>

            <div style={{
              background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
              borderRadius: '12px', padding: '1.15rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
                <Hash size={15} /> Cleaned Words
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'hsl(var(--accent))' }}>
                {extractionResult.total_cleaned_words.toLocaleString()}
              </div>
            </div>

            <div style={{
              background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
              borderRadius: '12px', padding: '1.15rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
                <Layers size={15} /> Training Passages
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'hsl(150 75% 45%)' }}>
                {extractionResult.total_chunks}
              </div>
            </div>

            <div style={{
              background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
              borderRadius: '12px', padding: '1.15rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'hsl(var(--muted-foreground))', fontSize: '0.8rem', marginBottom: '0.4rem' }}>
                <CheckCircle2 size={15} /> Duplicates Filtered
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                {extractionResult.deduplicated_chunks}
              </div>
            </div>
          </div>

          {/* ADA Domain Terminology & Abbreviations Preview */}
          {extractionResult.domain_terms && extractionResult.domain_terms.length > 0 && (
            <div style={{
              background: 'hsl(var(--card))', borderRadius: '12px',
              border: '1px solid hsl(var(--border))', padding: '1.25rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.85rem' }}>
                <Tag size={16} style={{ color: 'hsl(var(--accent))' }} />
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                  Identified ADA Domain Terminology & Technical Abbreviations
                </h4>
              </div>
              <p style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
                These domain keywords, acronyms, and recurring project names will be directly adapted into the embedding model space:
              </p>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {extractionResult.domain_terms.map((item, i) => (
                  <span
                    key={`${item.term}-${i}`}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: '6px',
                      background: 'linear-gradient(135deg, hsl(var(--accent) / 0.15) 0%, hsl(var(--accent) / 0.05) 100%)',
                      border: '1px solid hsl(var(--accent) / 0.3)',
                      color: 'hsl(var(--foreground))',
                      padding: '4px 10px', borderRadius: '20px', fontSize: '0.78rem', fontWeight: 600,
                    }}
                  >
                    {item.term}
                    <span style={{
                      background: 'hsl(var(--accent))', color: 'white',
                      borderRadius: '10px', padding: '1px 5px', fontSize: '0.68rem', fontWeight: 700,
                    }}>
                      {item.count}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Document Status Table */}
          <div style={{
            background: 'hsl(var(--card))', borderRadius: '12px',
            border: '1px solid hsl(var(--border))', overflow: 'hidden',
          }}>
            <div style={{
              padding: '0.85rem 1.25rem', borderBottom: '1px solid hsl(var(--border))',
              fontSize: '0.875rem', fontWeight: 700, color: 'hsl(var(--foreground))',
            }}>
              Extraction Breakdown ({extractionResult.document_stats.length} Files)
            </div>
            <div style={{ maxHeight: '220px', overflowY: 'auto' }}>
              {extractionResult.document_stats.map((doc, idx) => (
                <div
                  key={`${doc.filename}-${idx}`}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '0.7rem 1.25rem', borderBottom: '1px solid hsl(var(--border) / 0.5)',
                    fontSize: '0.82rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <FileText size={15} style={{ color: 'hsl(var(--muted-foreground))' }} />
                    <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>{doc.filename}</span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <span style={{ color: 'hsl(var(--muted-foreground))' }}>
                      {doc.word_count.toLocaleString()} words
                    </span>
                    {doc.status === 'extracted' ? (
                      <span style={{
                        display: 'flex', alignItems: 'center', gap: '4px',
                        color: 'hsl(150 75% 45%)', fontWeight: 600, fontSize: '0.75rem',
                      }}>
                        <CheckCircle2 size={13} /> Extracted
                      </span>
                    ) : (
                      <span style={{
                        display: 'flex', alignItems: 'center', gap: '4px',
                        color: 'hsl(0 80% 65%)', fontWeight: 600, fontSize: '0.75rem',
                      }} title={doc.error || 'Failed'}>
                        <AlertTriangle size={13} /> Error
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Collapsible Sample Chunks */}
          {extractionResult.sample_chunks && extractionResult.sample_chunks.length > 0 && (
            <div style={{
              background: 'hsl(var(--card))', borderRadius: '12px',
              border: '1px solid hsl(var(--border))', overflow: 'hidden',
            }}>
              <button
                onClick={() => setShowSampleChunks(!showSampleChunks)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '0.85rem 1.25rem', background: 'transparent', border: 'none',
                  cursor: 'pointer', color: 'hsl(var(--foreground))', fontWeight: 600, fontSize: '0.875rem',
                }}
              >
                <span>Preview Sample Normalized Chunks ({extractionResult.sample_chunks.length})</span>
                {showSampleChunks ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showSampleChunks && (
                <div style={{
                  padding: '1rem 1.25rem', borderTop: '1px solid hsl(var(--border))',
                  display: 'flex', flexDirection: 'column', gap: '0.75rem',
                }}>
                  {extractionResult.sample_chunks.map((chunk, i) => (
                    <div
                      key={i}
                      style={{
                        padding: '0.75rem 1rem', background: 'hsl(var(--muted) / 0.3)',
                        borderRadius: '8px', fontSize: '0.8rem', color: 'hsl(var(--foreground))',
                        lineHeight: 1.5, borderLeft: '3px solid hsl(var(--accent))',
                      }}
                    >
                      {chunk}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Navigation buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
        <button
          onClick={onBack}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: 'hsl(var(--muted) / 0.5)', color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))', padding: '0.65rem 1.25rem',
            borderRadius: '8px', fontWeight: 600, fontSize: '0.875rem', cursor: 'pointer',
          }}
        >
          <ArrowLeft size={16} />
          Back: Documents
        </button>

        <button
          onClick={onNext}
          disabled={!extractionResult || extractionResult.total_chunks < 2 || extracting}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: extractionResult && extractionResult.total_chunks >= 2 && !extracting
              ? 'hsl(var(--accent))'
              : 'hsl(var(--muted))',
            color: extractionResult && extractionResult.total_chunks >= 2 && !extracting
              ? 'white'
              : 'hsl(var(--muted-foreground))',
            border: 'none', padding: '0.65rem 1.5rem', borderRadius: '8px',
            fontWeight: 600, fontSize: '0.875rem',
            cursor: extractionResult && extractionResult.total_chunks >= 2 && !extracting
              ? 'pointer'
              : 'not-allowed',
          }}
        >
          Next: Embedding Training
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
