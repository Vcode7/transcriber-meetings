import { useState } from 'react';
import { UploadCloud, Edit3, Sparkles, Loader2, FileText, CheckCircle2, ListFilter, AlertCircle } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { extractFileForTraining, extractMomPoints } from '../api/trainingApi';
import { toast } from 'sonner';

export default function ManualMomEditor() {
  const {
    manualMom,
    setManualMom,
    manualMomFilename,
    setManualMomFilename,
    extractedMomPoints,
    setExtractedMomPoints,
    isExtractingMom,
    setIsExtractingMom,
    uploadedContext,
    uploadedTranscript,
  } = useTrainingStore();

  const [isExtractingFile, setIsExtractingFile] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsExtractingFile(true);
    setManualMomFilename(file.name);
    try {
      const res = await extractFileForTraining(file);
      setManualMom(res.extracted_text);
      toast.success(`Extracted text from ${file.name} using document extraction & OCR`);
    } catch (err: any) {
      toast.error(`Failed to extract text from ${file.name}`);
    } finally {
      setIsExtractingFile(false);
    }
  };

  const handleExtractPointsLLM = async () => {
    if (!manualMom || !manualMom.strip ? false : manualMom.trim().length === 0) {
      toast.error('Please upload a file or enter Manual MoM text first.');
      return;
    }

    setIsExtractingMom(true);
    try {
      const combinedContext = [uploadedContext, uploadedTranscript].filter(Boolean).join('\n\n');
      const res = await extractMomPoints(manualMom, combinedContext);
      setExtractedMomPoints(res.points);
      toast.success(`Extracted ${res.count} raw MoM points preserving exact wording!`);
    } catch (err: any) {
      toast.error('Failed to extract MoM points via LLM.');
    } finally {
      setIsExtractingMom(false);
    }
  };

  return (
    <div style={{
      background: 'hsl(var(--card))',
      borderRadius: '12px',
      padding: '1.25rem 1.5rem',
      border: '1.5px solid hsl(var(--border) / 0.8)',
      display: 'flex',
      flexDirection: 'column',
      gap: '1.25rem',
      height: '100%',
    }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <Edit3 size={18} style={{ color: 'hsl(var(--accent))' }} />
          <h3 style={{ margin: 0, fontWeight: 700, fontSize: '1.05rem', color: 'hsl(var(--foreground))' }}>
            Manual MoM Upload
          </h3>
        </div>
        <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
          Upload PDF, DOC/DOCX, scanned documents (OCR), images, or enter manual MoM text.
        </p>
      </div>

      {/* File Upload Dropzone */}
      <div style={{
        border: `2px dashed ${manualMomFilename ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
        borderRadius: '10px',
        padding: '1rem',
        textAlign: 'center',
        background: manualMomFilename ? 'hsl(var(--accent) / 0.04)' : 'hsl(var(--background))',
        position: 'relative',
        cursor: 'pointer',
        transition: 'all 0.15s ease',
      }}>
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.csv"
          onChange={handleFileUpload}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }}
        />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
          {isExtractingFile ? (
            <Loader2 size={20} className="spin" style={{ color: 'hsl(var(--accent))' }} />
          ) : manualMomFilename ? (
            <CheckCircle2 size={20} style={{ color: 'hsl(140 65% 45%)' }} />
          ) : (
            <UploadCloud size={20} style={{ color: 'hsl(var(--muted-foreground))' }} />
          )}
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
            {isExtractingFile ? 'Running Document Extraction & OCR...' : manualMomFilename ? `Loaded: ${manualMomFilename}` : 'Drop PDF / DOC / DOCX / Image here or click to upload'}
          </span>
        </div>
        <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px' }}>
          Supports scanned PDFs and image-based documents via RapidOCR pipeline
        </div>
      </div>

      {/* Text Editor */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem' }}>
          <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>Extracted / Raw MoM Content</span>
          <span style={{ color: 'hsl(var(--muted-foreground))' }}>{manualMom.length} chars</span>
        </div>

        <textarea
          value={manualMom}
          onChange={(e) => setManualMom(e.target.value)}
          placeholder="Extracted MoM text will appear here. You can also paste or edit ground-truth MoM text directly..."
          style={{
            width: '100%',
            minHeight: '160px',
            maxHeight: '260px',
            background: 'hsl(var(--background))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            color: 'hsl(var(--foreground))',
            fontFamily: 'Inter, sans-serif',
            fontSize: '0.85rem',
            lineHeight: '1.45',
            resize: 'vertical',
          }}
        />
      </div>

      {/* LLM JSON Point Extraction Trigger */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <button
          onClick={handleExtractPointsLLM}
          disabled={isExtractingMom || !manualMom || manualMom.trim().length === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '0.65rem 1.25rem',
            borderRadius: '8px',
            background: manualMom && manualMom.trim().length > 0
              ? 'linear-gradient(135deg, hsl(var(--accent)), hsl(262 80% 65%))'
              : 'hsl(var(--muted))',
            color: manualMom && manualMom.trim().length > 0 ? 'white' : 'hsl(var(--muted-foreground))',
            border: 'none',
            fontWeight: 700,
            fontSize: '0.85rem',
            cursor: manualMom && manualMom.trim().length > 0 ? 'pointer' : 'not-allowed',
            transition: 'all 0.15s ease',
            boxShadow: manualMom && manualMom.trim().length > 0 ? '0 0 15px hsl(var(--accent) / 0.3)' : 'none',
          }}
        >
          {isExtractingMom ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />}
          {isExtractingMom ? 'Extracting Points via LLM...' : 'Extract Raw MoM Points (Preserve Wording)'}
        </button>

        {/* LLM Extraction Rules Notice */}
        <div style={{
          fontSize: '0.72rem',
          color: 'hsl(var(--muted-foreground))',
          background: 'hsl(var(--background))',
          padding: '0.5rem 0.75rem',
          borderRadius: '6px',
          border: '1px solid hsl(var(--border) / 0.5)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '6px',
        }}>
          <AlertCircle size={14} style={{ color: 'hsl(var(--accent))', flexShrink: 0, marginTop: '1px' }} />
          <span>
            The LLM returns JSON containing <strong>only extracted points</strong> with <strong>exact original wording preserved</strong>. Zero intros, agenda headings, extra explanations, summaries, or paraphrasing.
          </span>
        </div>
      </div>

      {/* Extracted Points Preview (JSON) */}
      {extractedMomPoints.length > 0 && (
        <div style={{
          background: 'hsl(var(--background))',
          borderRadius: '8px',
          border: '1.5px solid hsl(140 65% 45% / 0.4)',
          padding: '0.85rem 1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          maxHeight: '200px',
          overflowY: 'auto',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(140 65% 40%)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ListFilter size={14} />
              Extracted Raw Points ({extractedMomPoints.length})
            </span>
            <span style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>
              Exact Wording Preserved
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            {extractedMomPoints.map((pt, i) => (
              <div
                key={i}
                style={{
                  fontSize: '0.78rem',
                  color: 'hsl(var(--foreground))',
                  padding: '4px 8px',
                  background: 'hsl(var(--card))',
                  borderRadius: '4px',
                  border: '1px solid hsl(var(--border) / 0.5)',
                }}
              >
                <strong>Point {i + 1}:</strong> {pt}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
