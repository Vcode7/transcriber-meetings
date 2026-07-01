/**
 * PDFButton.tsx
 * A button that opens the ExportPDFModal for PDF report generation.
 */
import { useState } from 'react'
import { FileDown } from 'lucide-react'
import ExportPDFModal from './ExportPDFModal'

interface Props {
  recordingId: string | null | undefined
  filename?: string
  variant?: 'ghost' | 'primary' | 'outline'
  className?: string
  style?: React.CSSProperties
}

export default function PDFButton({
  recordingId,
  filename,
  variant = 'ghost',
  className = '',
  style,
}: Props) {
  const [modalOpen, setModalOpen] = useState(false)

  const isDisabled = !recordingId

  const btnClass =
    variant === 'primary'
      ? 'btn btn-primary'
      : variant === 'outline'
      ? 'btn'
      : 'btn btn-ghost'

  return (
    <>
      <button
        id="pdf-export-btn"
        className={`${btnClass} ${className}`}
        onClick={() => setModalOpen(true)}
        disabled={isDisabled}
        title={!recordingId ? 'Process a recording first' : 'Export professional PDF report'}
        style={{
          flexShrink: 0,
          fontSize: '.82rem',
          padding: '.4rem .85rem',
          height: '36px',
          gap: '6px',
          ...style,
        }}
      >
        <FileDown size={14} />
        <span>Export PDF</span>
      </button>

      <ExportPDFModal
        recordingId={recordingId}
        filename={filename}
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      />
    </>
  )
}
