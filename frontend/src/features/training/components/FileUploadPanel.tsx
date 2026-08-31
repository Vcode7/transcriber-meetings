import { useState } from 'react';
import { UploadCloud, FileText, Loader2, CheckCircle2 } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { extractFileForTraining } from '../api/trainingApi';
import { toast } from 'sonner';

export default function FileUploadPanel() {
  const {
    uploadedTranscript,
    setUploadedTranscript,
    uploadedContext,
    setUploadedContext,
    uploadedAgenda,
    setUploadedAgenda,
  } = useTrainingStore();

  const [loadingSlot, setLoadingSlot] = useState<number | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>, slotIdx: number, setter: (t: string) => void) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoadingSlot(slotIdx);
    try {
      const res = await extractFileForTraining(file);
      setter(res.extracted_text || file.name);
      toast.success(`Extracted ${res.char_count} chars from ${file.name}`);
    } catch (err: any) {
      toast.error(`Failed to extract text from ${file.name}`);
      setter(file.name);
    } finally {
      setLoadingSlot(null);
    }
  };

  const slots = [
    { label: 'Transcription Context File', val: uploadedTranscript, setter: setUploadedTranscript, accept: '.txt,.pdf,.doc,.docx,.md,.png,.jpg,.jpeg' },
    { label: 'Meeting Context File', val: uploadedContext, setter: setUploadedContext, accept: '.txt,.pdf,.doc,.docx,.md,.png,.jpg,.jpeg' },
    { label: 'Agenda File', val: uploadedAgenda, setter: setUploadedAgenda, accept: '.txt,.pdf,.doc,.docx,.md,.png,.jpg,.jpeg' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h3 style={{ margin: '0 0 0.25rem 0', fontWeight: 700, fontSize: '1rem', color: 'hsl(var(--foreground))' }}>
          Upload Context & Agenda Files
        </h3>
        <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
          Upload transcription context files, agenda documents, meeting context, or reference files. Uses document extraction pipeline with OCR for scanned files and PDFs.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        {slots.map((slot, idx) => {
          const isLoading = loadingSlot === idx;
          const hasVal = Boolean(slot.val);

          return (
            <div
              key={idx}
              style={{
                border: `2px dashed ${hasVal ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                borderRadius: '10px',
                padding: '1.25rem 1rem',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.5rem',
                position: 'relative',
                background: hasVal ? 'hsl(var(--accent) / 0.04)' : 'hsl(var(--background))',
                transition: 'all 0.15s ease',
              }}
            >
              <input
                type="file"
                accept={slot.accept}
                onChange={(e) => handleFileChange(e, idx, slot.setter)}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }}
              />

              {isLoading ? (
                <Loader2 size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              ) : hasVal ? (
                <CheckCircle2 size={24} style={{ color: 'hsl(140 65% 45%)' }} />
              ) : (
                <UploadCloud size={24} style={{ color: 'hsl(var(--muted-foreground))' }} />
              )}

              <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'hsl(var(--foreground))' }}>{slot.label}</div>
              <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>PDF, DOCX, Images & Text</div>

              {hasVal && (
                <div style={{
                  marginTop: '0.4rem',
                  fontSize: '0.75rem',
                  color: 'hsl(var(--accent))',
                  fontWeight: 600,
                  wordBreak: 'break-all',
                  maxWidth: '100%',
                  lineHeight: '1.3',
                }}>
                  Loaded ({slot.val.length} chars)
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
