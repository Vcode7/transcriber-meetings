import { useState, useRef } from 'react'
import { CheckCircle, UserPlus, Loader, AlertTriangle, Sparkles, Mic, ArrowRight, FolderUp, Folder, FileAudio, Check, X, RefreshCw } from 'lucide-react'
import VoiceRecorder from '../components/VoiceRecorder'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'
import { toast } from 'sonner'

const STEPS = [
  { title: 'Sample 1', instruction: 'Record a clear voice sample (10-30 seconds). Speak slowly and pronounce each word clearly.', num: '1' },
  { title: 'Sample 2', instruction: 'Record another sample in a normal conversational tone, as if talking in a real meeting.', num: '2' },
  { title: 'Sample 3', instruction: 'Optional but recommended -- speak with a slightly different pitch (excited, serious, or calm).', num: '3', optional: true },
]

const SAMPLE_SCRIPTS = [
  "Now I am speaking slowly and clearly. Each word is pronounced properly. This helps the system understand my voice in a clean way.",
  "Now I will speak normally, like in a real conversation. There will be some background noise and I will not be perfectly clear. This helps the system understand my voice in a real-world environment.",
  "Now I will speak with a slightly different pitch. This helps the system recognise my voice when I am excited, happy, calm or serious.",
]

export default function AddVoicePage() {
  const [activeTab, setActiveTab] = useState<'single' | 'folder'>('single')

  // Single profile recorder state
  const [label, setLabel] = useState('')
  const [savedPaths, setSavedPaths] = useState<(string | null)[]>([null, null, null])
  const [submitting, setSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState('')
  const [step, setStep] = useState(0)

  // Folder upload state
  const folderInputRef = useRef<HTMLInputElement>(null)
  const [folderFiles, setFolderFiles] = useState<File[]>([])
  const [parsedSpeakers, setParsedSpeakers] = useState<{ [speaker: string]: { file: File; relPath: string }[] }>({})
  const [skippedFolderFiles, setSkippedFolderFiles] = useState<{ relPath: string; reason: string }[]>([])
  const [folderImporting, setFolderImporting] = useState(false)
  const [folderProgress, setFolderProgress] = useState(0)
  const [folderSummary, setFolderSummary] = useState<any | null>(null)

  const handleSampleSaved = (filePath: string, sampleIndex: number) => {
    setSavedPaths((prev) => {
      const next = [...prev]
      next[sampleIndex] = filePath
      return next
    })
  }

  const goNext = () => {
    if (!savedPaths[step]) {
      setError('Please record and save this sample before continuing.')
      return
    }
    setError('')
    setStep((s) => Math.min(s + 1, STEPS.length - 1))
  }

  const handleSave = async () => {
    if (!label.trim()) { setError('Please enter a name for this voice.'); return }
    const paths = savedPaths.filter(Boolean) as string[]
    if (paths.length < 2) { setError('Please record at least 2 samples for better accuracy.'); return }
    setSubmitting(true); setError('')
    try {
      await api.post('/voice/add-profile', { file_paths: paths, label: label.trim() })
      setSuccess(true)
      setLabel(''); setSavedPaths([null, null, null]); setStep(0)
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Failed to save profile.'))
    } finally {
      setSubmitting(false)
    }
  }

  // Folder selection logic
  const handleFolderSelect = (filesList: FileList | null) => {
    if (!filesList || filesList.length === 0) return
    const filesArray = Array.from(filesList)
    setFolderFiles(filesArray)

    const supportedExts = new Set(['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.webm'])
    const speakersMap: { [speaker: string]: { file: File; relPath: string }[] } = {}
    const skipped: { relPath: string; reason: string }[] = []

    filesArray.forEach((file) => {
      const relPath = file.webkitRelativePath || file.name
      const parts = relPath.replace(/\\/g, '/').split('/').filter(Boolean)
      const ext = '.' + (file.name.split('.').pop() || '').toLowerCase()

      if (parts.length < 2) {
        skipped.push({ relPath, reason: 'File directly in root folder without a speaker subfolder' })
        return
      }

      if (!supportedExts.has(ext)) {
        skipped.push({ relPath, reason: `Unsupported audio format '${ext}'` })
        return
      }

      const speakerName = parts[parts.length - 2]
      if (!speakersMap[speakerName]) {
        speakersMap[speakerName] = []
      }
      speakersMap[speakerName].push({ file, relPath })
    })

    setParsedSpeakers(speakersMap)
    setSkippedFolderFiles(skipped)
    setFolderSummary(null)
    setError('')
  }

  const handleFolderImportSubmit = async () => {
    const speakerEntries = Object.entries(parsedSpeakers)
    if (speakerEntries.length === 0) {
      setError('No valid speaker subfolders or audio samples found in selected folder.')
      return
    }

    setFolderImporting(true)
    setError('')
    setFolderProgress(0)

    const allFiles: File[] = []
    const relativePaths: string[] = []

    speakerEntries.forEach(([_, items]) => {
      items.forEach(({ file, relPath }) => {
        allFiles.push(file)
        relativePaths.push(relPath)
      })
    })

    const form = new FormData()
    allFiles.forEach((f) => form.append('files', f))
    form.append('relative_paths', JSON.stringify(relativePaths))

    try {
      const res = await api.post('/voice/bulk-folder-import', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (evt) => {
          if (evt.total) {
            setFolderProgress(Math.round((evt.loaded / evt.total) * 100))
          }
        },
      })
      setFolderSummary(res.data)
      toast.success(`Processed ${res.data.successful_speakers} speaker voice profiles!`)
    } catch (e: unknown) {
      const msg = getApiErrorDetail(e, 'Failed to import folder.')
      setError(msg)
      toast.error(msg)
    } finally {
      setFolderImporting(false)
    }
  }

  if (success) return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: '2rem', padding: '3rem',
      background: 'hsl(var(--paper) / .4)'
    }}>
      <div className="animate-bounce-in" style={{
        width: '100px', height: '100px', borderRadius: '50%',
        background: 'hsl(var(--success) / .12)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '3px solid hsl(var(--success) / .35)',
        boxShadow: '0 0 0 14px hsl(var(--success) / .06)',
      }}>
        <CheckCircle size={52} style={{ color: 'hsl(var(--success))' }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontWeight: 800, fontSize: '1.85rem', fontFamily: 'Caveat, cursive', marginBottom: '.65rem', color: 'hsl(var(--ink))', letterSpacing: '-.02em' }}>
          Voice profile saved!
        </h2>
        <p style={{ color: 'hsl(var(--pencil))', fontSize: '.95rem', fontFamily: 'Inter, sans-serif', maxWidth: '380px', lineHeight: 1.65 }}>
          VoiceSum will now recognise this person in future recordings
        </p>
      </div>
      <button
        className="btn btn-success animate-slide-up"
        onClick={() => setSuccess(false)}
        style={{ animationDelay: '0.2s', animationFillMode: 'both', padding: '.75rem 1.75rem', fontSize: '.95rem' }}
      >
        <UserPlus size={16} /> Add Another Voice
      </button>
    </div>
  )

  const doneCount = savedPaths.filter(Boolean).length
  const totalSpeakersFound = Object.keys(parsedSpeakers).length

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>

      <div className="panel-header">
        <div style={{
          width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
          background: 'hsl(var(--accent) / .12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '2px solid hsl(var(--accent) / .3)',
        }}>
          <UserPlus size={16} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1>Add Voice Profile</h1>
          <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
            Train speaker identification interactively or bulk import speaker subfolders
          </p>
        </div>
      </div>

      <div className="page-wrapper">

        {/* Top Mode Tabs */}
        <div style={{
          display: 'flex', gap: '8px', marginBottom: '1.5rem',
          padding: '4px', background: 'hsl(var(--card))',
          borderRadius: '12px', border: '1.5px solid hsl(var(--ink) / .1)',
        }}>
          <button
            type="button"
            onClick={() => { setActiveTab('single'); setError('') }}
            style={{
              flex: 1, padding: '.7rem 1rem', borderRadius: '9px', border: 'none',
              background: activeTab === 'single' ? 'hsl(var(--accent))' : 'transparent',
              color: activeTab === 'single' ? '#fff' : 'hsl(var(--ink-soft))',
              fontWeight: 700, fontSize: '.88rem', fontFamily: 'Inter, sans-serif',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
              transition: 'all .2s',
            }}
          >
            <Mic size={16} /> Single Voice Recorder
          </button>
          <button
            type="button"
            onClick={() => { setActiveTab('folder'); setError('') }}
            style={{
              flex: 1, padding: '.7rem 1rem', borderRadius: '9px', border: 'none',
              background: activeTab === 'folder' ? 'hsl(var(--accent))' : 'transparent',
              color: activeTab === 'folder' ? '#fff' : 'hsl(var(--ink-soft))',
              fontWeight: 700, fontSize: '.88rem', fontFamily: 'Inter, sans-serif',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
              transition: 'all .2s',
            }}
          >
            <FolderUp size={16} /> Bulk Folder Import
          </button>
        </div>

        {activeTab === 'single' ? (
          <>
            {/* Name */}
            <div className="animate-slide-up" style={{
              padding: '1.5rem', marginBottom: '1.5rem',
              background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--ink) / .1)',
              borderRadius: '12px', animationDelay: '0.08s', animationFillMode: 'both'
            }}>
              <label className="label" style={{ marginBottom: '.75rem', fontSize: '.88rem' }}>
                Name / Label *
              </label>
              <input
                id="voice-label"
                className="input"
                placeholder="e.g. Alice, John, Boss"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
              />
            </div>

            {/* Sample scripts */}
            <div className="animate-slide-up" style={{
              padding: '1.25rem', marginBottom: '1.5rem',
              background: 'hsl(205,90%,55% / .07)', border: '1.5px solid hsl(205,90%,55% / .25)',
              borderRadius: '12px', animationDelay: '0.12s', animationFillMode: 'both'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}>
                <Sparkles size={16} style={{ color: 'hsl(205,90%,55%)' }} />
                <h3 style={{ fontSize: '.95rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                  Sample Scripts <span style={{ fontWeight: 400, color: 'hsl(var(--pencil))' }}>(read aloud)</span>
                </h3>
              </div>
              {SAMPLE_SCRIPTS.map((script, idx) => (
                <div key={idx} style={{
                  padding: '.85rem 1rem',
                  background: step === idx ? 'hsl(var(--accent) / .07)' : 'hsl(var(--card))',
                  borderRadius: '8px',
                  border: step === idx ? '1.5px solid hsl(var(--accent) / .3)' : '1.5px solid hsl(var(--ink) / .08)',
                  marginBottom: idx < SAMPLE_SCRIPTS.length - 1 ? '.75rem' : 0,
                  fontSize: '.88rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif',
                  color: 'hsl(var(--ink-soft))', display: 'flex', gap: '.75rem', transition: 'all .2s',
                }}>
                  <span style={{ fontWeight: 700, color: step === idx ? 'hsl(var(--accent))' : 'hsl(var(--pencil))', flexShrink: 0, fontSize: '.85rem' }}>
                    {idx + 1}.
                  </span>
                  {script}
                </div>
              ))}
            </div>

            {/* Step tabs */}
            <div className="animate-slide-up" style={{
              display: 'flex', gap: '8px', marginBottom: '1.5rem',
              animationDelay: '0.16s', animationFillMode: 'both'
            }}>
              {STEPS.map((s, i) => {
                const isDone = !!savedPaths[i]
                const isActive = step === i
                return (
                  <button
                    key={i}
                    onClick={() => { setError(''); setStep(i) }}
                    style={{
                      flex: 1, padding: '.85rem .5rem', borderRadius: '10px',
                      border: isActive ? '2px solid hsl(var(--accent))' : isDone ? '2px solid hsl(130,60%,45% / .4)' : '1.5px dashed hsl(var(--ink) / .15)',
                      cursor: 'pointer',
                      background: isActive ? 'hsl(var(--accent) / .1)' : isDone ? 'hsl(130,60%,45% / .08)' : 'hsl(var(--card))',
                      color: isActive ? 'hsl(var(--accent))' : isDone ? 'hsl(130,60%,45%)' : 'hsl(var(--pencil))',
                      fontSize: '0.85rem', fontWeight: 600,
                      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px',
                      transition: 'all 0.2s', fontFamily: 'Inter, sans-serif',
                      boxShadow: isActive ? '0 2px 8px hsl(var(--accent) / .15)' : 'none'
                    }}
                  >
                    <div style={{
                      width: '24px', height: '24px', borderRadius: '50%',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: isDone ? 'hsl(130,60%,45%)' : isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                      color: isDone || isActive ? '#fff' : 'hsl(var(--pencil))',
                      fontSize: '.72rem', fontWeight: 700, transition: 'all .2s'
                    }}>
                      {isDone ? '✓' : s.num}
                    </div>
                    {s.title}
                    {s.optional && (
                      <span style={{ fontSize: '.62rem', fontWeight: 400, color: 'hsl(var(--pencil))', opacity: .75 }}>Optional</span>
                    )}
                  </button>
                )
              })}
            </div>

            {/* Current recorder */}
            <div className="animate-slide-up" style={{ marginBottom: '1.25rem', animationDelay: '0.2s', animationFillMode: 'both' }}>
              <div style={{
                padding: '1.5rem', background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--ink) / .1)', borderRadius: '12px', marginBottom: '1rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.1rem' }}>
                  <Mic size={16} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
                  <p style={{ fontSize: '.9rem', color: 'hsl(var(--ink-soft))', fontFamily: 'Inter, sans-serif', fontWeight: 500 }}>
                    {STEPS[step].instruction}
                  </p>
                </div>
                <VoiceRecorder
                  key={step}
                  sampleIndex={step}
                  label={label || 'speaker'}
                  onSampleSaved={handleSampleSaved}
                />
              </div>

              {/* Navigation */}
              <div style={{ display: 'flex', gap: '8px' }}>
                {step > 0 && (
                  <button
                    className="btn btn-ghost"
                    onClick={() => { setError(''); setStep(step - 1) }}
                    style={{ flex: 1, justifyContent: 'center', padding: '.65rem 1.25rem', fontSize: '.9rem' }}
                  >
                    Previous
                  </button>
                )}
                {step < STEPS.length - 1 && (
                  <button
                    className="btn btn-ghost"
                    onClick={goNext}
                    style={{ flex: 1, justifyContent: 'center', padding: '.65rem 1.25rem', fontSize: '.9rem' }}
                  >
                    Next sample <ArrowRight size={14} style={{ marginLeft: '4px' }} />
                  </button>
                )}
              </div>
            </div>

            {/* Progress dots */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.25rem',
              padding: '.75rem 1rem', borderRadius: '10px',
              background: 'hsl(var(--muted) / .5)', border: '1.5px solid hsl(var(--ink) / .08)',
              fontFamily: 'Inter, sans-serif', fontSize: '.82rem', color: 'hsl(var(--pencil))',
            }}>
              <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                {savedPaths.map((p, i) => (
                  <div key={i} style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: p ? 'hsl(130,60%,45%)' : 'hsl(var(--muted))',
                    border: `1.5px solid ${p ? 'hsl(130,60%,45% / .5)' : 'hsl(var(--ink) / .15)'}`,
                    transition: 'all .2s',
                  }} />
                ))}
              </div>
              <span>
                {doneCount} of 3 samples recorded
                {doneCount >= 2 ? ' — ready to save!' : ' — need at least 2'}
              </span>
            </div>

            {/* Error */}
            {error && (
              <div className="animate-shake" style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                color: 'hsl(var(--destructive))', fontSize: '0.88rem', marginBottom: '1.25rem',
                padding: '.75rem 1rem', background: 'hsl(var(--destructive) / .08)',
                border: '1.5px solid hsl(var(--destructive) / .25)',
                borderRadius: '10px', fontFamily: 'Inter, sans-serif', fontWeight: 500
              }}>
                <AlertTriangle size={15} /> {error}
              </div>
            )}

            {/* Save */}
            <button
              className="btn btn-primary"
              onClick={handleSave}
              disabled={submitting || doneCount < 2}
              id="save-voice-btn"
              style={{ width: '100%', justifyContent: 'center', padding: '.8rem 1.5rem', fontSize: '.95rem', marginTop: '.5rem' }}
            >
              {submitting ? <Loader size={16} className="spin" /> : <CheckCircle size={16} />}
              {submitting ? 'Saving...' : `Save Voice Profile (${doneCount} sample${doneCount !== 1 ? 's' : ''})`}
            </button>
          </>
        ) : (
          /* BULK FOLDER IMPORT TAB */
          <div className="animate-slide-up">
            {/* Instructions */}
            <div style={{
              padding: '1.25rem 1.5rem', marginBottom: '1.5rem',
              background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--ink) / .1)',
              borderRadius: '12px', lineHeight: 1.6, fontSize: '.88rem', color: 'hsl(var(--ink-soft))',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '.65rem' }}>
                <FolderUp size={18} style={{ color: 'hsl(var(--accent))' }} />
                <h3 style={{ fontSize: '.95rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                  Bulk Voice Profile Folder Import
                </h3>
              </div>
              <p style={{ margin: 0, color: 'hsl(var(--pencil))' }}>
                Select a root folder containing speaker subfolders. Subfolder names will automatically be used as speaker names:
              </p>
              <div style={{
                marginTop: '.8rem', padding: '.75rem 1rem', background: 'hsl(var(--ink) / .04)',
                borderRadius: '8px', border: '1px dashed hsl(var(--ink) / .15)',
                fontFamily: 'JetBrains Mono, monospace', fontSize: '.8rem', color: 'hsl(var(--ink))',
              }}>
                📁 RootFolder/<br />
                &nbsp;&nbsp;├── 📁 Alice/ &nbsp;&nbsp;&nbsp;&nbsp;(sample1.wav, sample2.mp3)<br />
                &nbsp;&nbsp;├── 📁 Bob/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(sample1.m4a, sample2.wav)<br />
                &nbsp;&nbsp;└── 📁 Charlie/ &nbsp;&nbsp;(sample1.flac)
              </div>
            </div>

            {/* Folder Input Picker */}
            <input
              type="file"
              ref={folderInputRef}
              style={{ display: 'none' }}
              // @ts-expect-error webkitdirectory is standard in HTML5 directory pickers
              webkitdirectory=""
              directory=""
              multiple
              onChange={(e) => handleFolderSelect(e.target.files)}
            />

            <div
              onClick={() => folderInputRef.current?.click()}
              style={{
                padding: '2.5rem 1.5rem', marginBottom: '1.5rem',
                border: '2px dashed hsl(var(--accent) / .4)', borderRadius: '14px',
                background: 'hsl(var(--accent) / .03)', textAlign: 'center',
                cursor: 'pointer', transition: 'all .2s',
              }}
            >
              <FolderUp size={42} style={{ color: 'hsl(var(--accent))', marginBottom: '.75rem' }} />
              <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: '0 0 .35rem' }}>
                {folderFiles.length > 0 ? `Selected Folder: ${totalSpeakersFound} Speakers Found` : 'Click to Select Root Speaker Folder'}
              </h4>
              <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', margin: 0 }}>
                {folderFiles.length > 0
                  ? `${folderFiles.length} total files scanned across ${totalSpeakersFound} subfolders`
                  : 'Supports WAV, MP3, M4A, FLAC, OGG, AAC, WMA, WEBM audio formats'}
              </p>
              <button
                type="button"
                className="btn btn-ghost"
                style={{ marginTop: '1.25rem', pointerEvents: 'none' }}
              >
                <Folder size={15} style={{ marginRight: '6px' }} /> Browse Speaker Directory
              </button>
            </div>

            {/* Folder Inspection Summary Card */}
            {totalSpeakersFound > 0 && (
              <div style={{
                padding: '1.25rem 1.5rem', marginBottom: '1.5rem',
                background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--ink) / .1)',
                borderRadius: '12px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h4 style={{ fontSize: '.92rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', margin: 0 }}>
                    Detected Speaker Subfolders ({totalSpeakersFound})
                  </h4>
                  <span style={{ fontSize: '.75rem', fontWeight: 600, padding: '3px 9px', borderRadius: '999px', background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))' }}>
                    Ready for Training
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '240px', overflowY: 'auto' }}>
                  {Object.entries(parsedSpeakers).map(([spk, items]) => (
                    <div
                      key={spk}
                      style={{
                        padding: '.65rem .9rem', borderRadius: '8px',
                        background: 'hsl(var(--paper) / .6)', border: '1px solid hsl(var(--ink) / .08)',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Folder size={16} style={{ color: 'hsl(var(--accent))' }} />
                        <span style={{ fontWeight: 700, fontSize: '.88rem', color: 'hsl(var(--ink))' }}>
                          {spk}
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
                          <FileAudio size={12} style={{ display: 'inline', marginRight: '4px' }} />
                          {items.length} audio sample{items.length !== 1 ? 's' : ''}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {skippedFolderFiles.length > 0 && (
                  <div style={{ marginTop: '1rem', paddingTop: '.85rem', borderTop: '1px solid hsl(var(--border) / .4)', fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
                    ⚠️ {skippedFolderFiles.length} file{skippedFolderFiles.length !== 1 ? 's' : ''} will be skipped (non-audio or directly in root folder).
                  </div>
                )}
              </div>
            )}

            {/* Import Progress Indicator */}
            {folderImporting && (
              <div style={{
                padding: '1.25rem 1.5rem', marginBottom: '1.5rem',
                background: 'hsl(var(--accent) / .05)', border: '1.5px solid hsl(var(--accent) / .3)',
                borderRadius: '12px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.65rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Loader size={16} className="spin" style={{ color: 'hsl(var(--accent))' }} />
                    <span style={{ fontWeight: 700, fontSize: '.88rem', color: 'hsl(var(--ink))' }}>
                      Extracting & Training Speaker Voice Profiles...
                    </span>
                  </div>
                  <span style={{ fontSize: '.85rem', fontWeight: 700, color: 'hsl(var(--accent))' }}>
                    {folderProgress}%
                  </span>
                </div>
                <div style={{ width: '100%', height: '8px', borderRadius: '999px', background: 'hsl(var(--accent) / .15)', overflow: 'hidden' }}>
                  <div style={{ width: `${folderProgress}%`, height: '100%', background: 'hsl(var(--accent))', transition: 'width .2s' }} />
                </div>
              </div>
            )}

            {/* Detailed Import Summary Report */}
            {folderSummary && (
              <div style={{
                padding: '1.25rem 1.5rem', marginBottom: '1.5rem',
                background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--ink) / .1)',
                borderRadius: '12px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
                  <CheckCircle size={20} style={{ color: 'hsl(var(--success))' }} />
                  <h4 style={{ fontSize: '1rem', fontWeight: 800, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', margin: 0 }}>
                    Bulk Voice Import Summary
                  </h4>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div style={{ padding: '.85rem', background: 'hsl(var(--success) / .1)', borderRadius: '10px', border: '1px solid hsl(var(--success) / .25)' }}>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'hsl(var(--success))' }}>
                      {folderSummary.successful_speakers}
                    </div>
                    <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>Speakers Trained</div>
                  </div>
                  {folderSummary.failed_speakers > 0 && (
                    <div style={{ padding: '.85rem', background: 'hsl(var(--destructive) / .1)', borderRadius: '10px', border: '1px solid hsl(var(--destructive) / .25)' }}>
                      <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'hsl(var(--destructive))' }}>
                        {folderSummary.failed_speakers}
                      </div>
                      <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>Failed Speakers</div>
                    </div>
                  )}
                  <div style={{ padding: '.85rem', background: 'hsl(var(--ink) / .04)', borderRadius: '10px', border: '1px solid hsl(var(--ink) / .1)' }}>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'hsl(var(--ink))' }}>
                      {folderSummary.skipped_files?.length ?? 0}
                    </div>
                    <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>Skipped Files</div>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
                  {folderSummary.speaker_results?.map((res: any) => (
                    <div
                      key={res.speaker}
                      style={{
                        padding: '.6rem .85rem', borderRadius: '8px',
                        background: res.status === 'success' ? 'hsl(var(--success) / .06)' : 'hsl(var(--destructive) / .06)',
                        border: `1px solid ${res.status === 'success' ? 'hsl(var(--success) / .2)' : 'hsl(var(--destructive) / .2)'}`,
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '.84rem',
                      }}
                    >
                      <span style={{ fontWeight: 700, color: 'hsl(var(--ink))' }}>{res.speaker}</span>
                      {res.status === 'success' ? (
                        <span style={{ color: 'hsl(var(--success))', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px', fontSize: '.78rem' }}>
                          <Check size={14} /> {res.action === 'created' ? 'Created' : 'Updated'} ({res.samples_trained} samples)
                        </span>
                      ) : (
                        <span style={{ color: 'hsl(var(--destructive))', fontWeight: 600, fontSize: '.78rem' }}>
                          Failed: {res.error}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="animate-shake" style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                color: 'hsl(var(--destructive))', fontSize: '0.88rem', marginBottom: '1.25rem',
                padding: '.75rem 1rem', background: 'hsl(var(--destructive) / .08)',
                border: '1.5px solid hsl(var(--destructive) / .25)',
                borderRadius: '10px', fontFamily: 'Inter, sans-serif', fontWeight: 500
              }}>
                <AlertTriangle size={15} /> {error}
              </div>
            )}

            {/* Action Submit Button */}
            <button
              className="btn btn-primary"
              onClick={handleFolderImportSubmit}
              disabled={folderImporting || totalSpeakersFound === 0}
              style={{ width: '100%', justifyContent: 'center', padding: '.8rem 1.5rem', fontSize: '.95rem', marginTop: '.5rem' }}
            >
              {folderImporting ? <Loader size={16} className="spin" /> : <FolderUp size={16} />}
              {folderImporting ? 'Importing & Training Voices...' : `Import & Train (${totalSpeakersFound} Speaker${totalSpeakersFound !== 1 ? 's' : ''})`}
            </button>
          </div>
        )}

      </div>
    </div>
  )
}
