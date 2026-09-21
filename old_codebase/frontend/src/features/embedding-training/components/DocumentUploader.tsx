import React, { useRef, useState } from 'react';
import {
  UploadCloud, FileText, FileSpreadsheet, Presentation,
  Trash2, ArrowLeft, ArrowRight, AlertCircle, FileCode,
  FileCheck
} from 'lucide-react';

interface DocumentUploaderProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  onBack: () => void;
  onNext: () => void;
}

const SUPPORTED_EXTENSIONS = [
  { ext: '.pdf', label: 'PDF', color: 'hsl(0 75% 55%)' },
  { ext: '.docx', label: 'DOCX', color: 'hsl(215 85% 55%)' },
  { ext: '.doc', label: 'DOC', color: 'hsl(215 85% 55%)' },
  { ext: '.pptx', label: 'PPTX', color: 'hsl(25 90% 55%)' },
  { ext: '.ppt', label: 'PPT', color: 'hsl(25 90% 55%)' },
  { ext: '.txt', label: 'TXT', color: 'hsl(150 70% 45%)' },
  { ext: '.md', label: 'MD', color: 'hsl(280 75% 60%)' },
];

export default function DocumentUploader({
  files,
  onFilesChange,
  onBack,
  onNext,
}: DocumentUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleFilesAdded = (newFiles: FileList | File[]) => {
    setValidationError(null);
    const validFiles: File[] = [];
    const validExts = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md'];

    Array.from(newFiles).forEach((f) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      if (validExts.includes(ext)) {
        // Prevent duplicate by filename and size
        const exists = files.some((existing) => existing.name === f.name && existing.size === f.size);
        if (!exists) {
          validFiles.push(f);
        }
      } else {
        setValidationError(`Ignored unsupported file format: ${f.name}`);
      }
    });

    if (validFiles.length > 0) {
      onFilesChange([...files, ...validFiles]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesAdded(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleRemoveFile = (index: number) => {
    const updated = [...files];
    updated.splice(index, 1);
    onFilesChange(updated);
  };

  const handleClearAll = () => {
    onFilesChange([]);
    setValidationError(null);
  };

  const getFormatBadge = (filename: string) => {
    const ext = '.' + filename.split('.').pop()?.toLowerCase();
    const match = SUPPORTED_EXTENSIONS.find((s) => s.ext === ext);
    if (!match) return null;
    return (
      <span style={{
        background: `${match.color}1f`,
        color: match.color,
        border: `1px solid ${match.color}40`,
        borderRadius: '4px',
        padding: '1px 6px',
        fontSize: '0.7rem',
        fontWeight: 700,
        textTransform: 'uppercase',
      }}>
        {match.label}
      </span>
    );
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
          Step 2: Upload ADA Documents
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
          Upload raw domain documents (technical specifications, system manuals, meeting notes, guidelines) to adapt the embedding model.
        </p>
      </div>

      {/* Supported format chips */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', fontWeight: 500 }}>
          Supported formats:
        </span>
        {SUPPORTED_EXTENSIONS.map((item) => (
          <span
            key={item.ext}
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '6px',
              background: 'hsl(var(--muted) / 0.4)',
              color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))',
            }}
          >
            {item.label}
          </span>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: isDragging
            ? '2px dashed hsl(var(--accent))'
            : '2px dashed hsl(var(--border))',
          borderRadius: '16px',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          background: isDragging
            ? 'hsl(var(--accent) / 0.05)'
            : 'hsl(var(--card))',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.md"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFilesAdded(e.target.files);
            }
          }}
        />
        <div style={{
          width: 52, height: 52, borderRadius: '14px',
          background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 1rem',
        }}>
          <UploadCloud size={28} />
        </div>
        <h4 style={{ margin: '0 0 0.4rem', fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
          Drag and drop documents here
        </h4>
        <p style={{ margin: '0 0 1rem', fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>
          or click to browse from your computer
        </p>
        <span style={{
          fontSize: '0.75rem', color: 'hsl(var(--accent))', fontWeight: 600,
          background: 'hsl(var(--accent) / 0.1)', padding: '4px 12px', borderRadius: '20px',
        }}>
          Upload multiple documents at once
        </span>
      </div>

      {validationError && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          background: 'hsl(0 80% 50% / 0.1)', color: 'hsl(0 80% 65%)',
          padding: '0.65rem 1rem', borderRadius: '8px', fontSize: '0.82rem',
        }}>
          <AlertCircle size={15} />
          {validationError}
        </div>
      )}

      {/* Uploaded Documents List */}
      {files.length > 0 && (
        <div style={{
          background: 'hsl(var(--card))', borderRadius: '12px',
          border: '1px solid hsl(var(--border))', overflow: 'hidden',
        }}>
          <div style={{
            padding: '0.85rem 1.25rem', borderBottom: '1px solid hsl(var(--border))',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileCheck size={16} style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                Uploaded Documents ({files.length})
              </span>
            </div>
            <button
              onClick={handleClearAll}
              style={{
                background: 'transparent', border: 'none', color: 'hsl(var(--muted-foreground))',
                fontSize: '0.78rem', cursor: 'pointer', padding: '2px 8px', borderRadius: '4px',
              }}
              title="Clear all uploaded documents"
            >
              Clear All
            </button>
          </div>

          <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
            {files.map((file, idx) => (
              <div
                key={`${file.name}-${idx}`}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '0.7rem 1.25rem', borderBottom: '1px solid hsl(var(--border) / 0.5)',
                  fontSize: '0.85rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
                  {getFormatBadge(file.name)}
                  <span style={{
                    fontWeight: 500, color: 'hsl(var(--foreground))',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {file.name}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', flexShrink: 0 }}>
                    ({formatFileSize(file.size)})
                  </span>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveFile(idx);
                  }}
                  style={{
                    background: 'transparent', border: 'none', color: 'hsl(var(--muted-foreground))',
                    cursor: 'pointer', padding: '4px', borderRadius: '4px', display: 'flex', alignItems: 'center',
                  }}
                  title="Remove document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
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
          Back: Base Model
        </button>

        <button
          onClick={onNext}
          disabled={files.length === 0}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: files.length > 0 ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
            color: files.length > 0 ? 'white' : 'hsl(var(--muted-foreground))',
            border: 'none', padding: '0.65rem 1.5rem', borderRadius: '8px',
            fontWeight: 600, fontSize: '0.875rem', cursor: files.length > 0 ? 'pointer' : 'not-allowed',
          }}
        >
          Next: Extract Text ({files.length})
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
