import type { CSSProperties, ReactNode } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import type { OcrFieldSchema, OcrSettingValue } from '../types'
import { formatValue } from '../utils'

const shell: CSSProperties = {
  padding: '.8rem .9rem',
  borderRadius: 10,
  background: 'hsl(var(--paper) / .4)',
  border: '1px solid hsl(var(--border) / .45)',
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  minWidth: 0,
}

const inputBase: CSSProperties = {
  padding: '5px 8px',
  borderRadius: 6,
  border: '1px solid hsl(var(--border))',
  background: 'hsl(var(--card))',
  fontSize: '.78rem',
  color: 'hsl(var(--ink))',
  fontFamily: 'Inter, sans-serif',
}

interface FieldProps {
  field: OcrFieldSchema
  value: OcrSettingValue
  savedValue: OcrSettingValue | undefined
  error?: string
  onChange: (value: OcrSettingValue) => void
}

/** Renders the right control for a schema field (slider, toggle or select). */
export function SettingField(props: FieldProps) {
  const { field } = props
  if (field.type === 'bool') return <ToggleField {...props} />
  if (field.type === 'enum') return <SelectField {...props} />
  return <NumberField {...props} />
}

function FieldShell({
  field, value, savedValue, error, onChange, children, inline,
}: FieldProps & { children: ReactNode; inline?: boolean }) {
  const modified = savedValue !== undefined && value !== savedValue
  const isDefault = value === field.default
  const showWarning = Boolean(field.warning) && !isDefault
  return (
    <div
      style={{
        ...shell,
        border: error
          ? '1.5px solid hsl(var(--destructive) / .7)'
          : modified
            ? '1.5px solid hsl(var(--accent) / .45)'
            : shell.border,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <label
          htmlFor={`ocr-${field.name}`}
          style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}
        >
          {field.label}
          {modified && (
            <span
              title="Changed — not saved yet"
              style={{ width: 7, height: 7, borderRadius: '50%', background: 'hsl(var(--accent))', flexShrink: 0 }}
            />
          )}
        </label>
        {inline ? children : (
          <span style={{
            fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--accent))', background: 'hsl(var(--accent) / .1)',
            padding: '1px 6px', borderRadius: 4, fontFamily: 'JetBrains Mono, monospace', whiteSpace: 'nowrap',
          }}>
            {formatValue(field, value)}
          </span>
        )}
      </div>
      <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', lineHeight: 1.4 }}>{field.help}</div>
      {!inline && children}
      {showWarning && (
        <div style={{
          display: 'flex', gap: 6, alignItems: 'flex-start', fontSize: '.68rem', lineHeight: 1.35,
          color: 'hsl(35 90% 38%)', background: 'hsl(35 90% 50% / .1)', padding: '5px 7px', borderRadius: 6,
        }}>
          <AlertTriangle size={12} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>{field.warning}</span>
        </div>
      )}
      {error && <div role="alert" style={{ fontSize: '.7rem', color: 'hsl(var(--destructive))', fontWeight: 600 }}>{error}</div>}
      {!isDefault && (
        <button
          type="button"
          onClick={() => onChange(field.default)}
          style={{
            alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 4, padding: 0,
            fontSize: '.66rem', color: 'hsl(var(--pencil))', background: 'none', border: 'none', cursor: 'pointer',
          }}
          title="Restore the default value"
        >
          <RotateCcw size={10} /> Default: {formatValue(field, field.default)}
        </button>
      )}
    </div>
  )
}

function NumberField(props: FieldProps) {
  const { field, value, onChange } = props
  const num = typeof value === 'number' ? value : Number(value)
  const step = field.step ?? (field.type === 'int' ? 1 : 0.01)
  const parse = (raw: string) => (raw.trim() === '' ? NaN : field.type === 'int' ? parseInt(raw, 10) : parseFloat(raw))
  return (
    <FieldShell {...props}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="range"
          aria-label={`${field.label} slider`}
          min={field.min}
          max={field.max}
          step={step}
          value={Number.isNaN(num) ? (field.min ?? 0) : num}
          onChange={e => onChange(parse(e.target.value))}
          style={{ flex: 1, minWidth: 0, accentColor: 'hsl(var(--accent))', cursor: 'pointer' }}
        />
        <input
          id={`ocr-${field.name}`}
          type="number"
          min={field.min}
          max={field.max}
          step={step}
          value={Number.isNaN(num) ? '' : num}
          onChange={e => onChange(parse(e.target.value))}
          style={{ ...inputBase, width: 86, textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}
        />
      </div>
    </FieldShell>
  )
}

function ToggleField(props: FieldProps) {
  const { field, value, onChange } = props
  const on = Boolean(value)
  return (
    <FieldShell {...props} inline>
      <button
        id={`ocr-${field.name}`}
        type="button"
        role="switch"
        aria-checked={on}
        aria-label={field.label}
        onClick={() => onChange(!on)}
        style={{
          width: 36, height: 20, borderRadius: 999, flexShrink: 0, position: 'relative', cursor: 'pointer',
          border: '1px solid hsl(var(--border))',
          background: on ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
          transition: 'background .15s ease',
        }}
      >
        <span style={{
          position: 'absolute', top: 2, left: on ? 18 : 2, width: 14, height: 14, borderRadius: '50%',
          background: 'white', boxShadow: '0 1px 2px rgba(0,0,0,.25)', transition: 'left .15s ease',
        }} />
      </button>
    </FieldShell>
  )
}

function SelectField(props: FieldProps) {
  const { field, value, onChange } = props
  return (
    <FieldShell {...props}>
      <select
        id={`ocr-${field.name}`}
        value={String(value)}
        onChange={e => onChange(e.target.value)}
        style={{ ...inputBase, width: '100%', padding: '6px 8px', cursor: 'pointer' }}
      >
        {(field.options ?? []).map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </FieldShell>
  )
}

export function Stat({ label, value, icon }: { label: string; value: string; icon?: ReactNode }) {
  return (
    <div style={{ padding: '.55rem .7rem', borderRadius: 8, background: 'hsl(var(--paper) / .45)', border: '1px solid hsl(var(--border) / .35)', minWidth: 0 }}>
      <div style={{ fontSize: '.64rem', color: 'hsl(var(--pencil))', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.04em', display: 'flex', alignItems: 'center', gap: 4 }}>
        {icon}{label}
      </div>
      <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {value}
      </div>
    </div>
  )
}

export function Banner({ tone, icon, children }: { tone: 'error' | 'warning' | 'info'; icon: ReactNode; children: ReactNode }) {
  const color = tone === 'error' ? 'var(--destructive)' : tone === 'warning' ? '35 90% 40%' : '205 85% 42%'
  return (
    <div role={tone === 'error' ? 'alert' : 'status'} style={{
      display: 'flex', gap: 8, alignItems: 'flex-start', padding: '.6rem .8rem', borderRadius: 8, fontSize: '.76rem', lineHeight: 1.45,
      background: `hsl(${color} / .09)`, color: `hsl(${color})`, fontWeight: 600,
    }}>
      <span style={{ flexShrink: 0, marginTop: 1, display: 'flex' }}>{icon}</span>
      <span>{children}</span>
    </div>
  )
}

/** Card used for each settings group and for the test panels. `color` uses space syntax: hsl(205 85% 52%). */
export function SectionCard({
  title, description, icon, color, children, actions,
}: {
  title: string
  description?: string
  icon: ReactNode
  color: string
  children: ReactNode
  actions?: ReactNode
}) {
  return (
    <section style={{
      borderRadius: 12, border: `1.5px solid ${color.replace(')', ' / .28)')}`,
      background: 'hsl(var(--card))', padding: '1.1rem 1.15rem',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: description ? '.2rem' : '.8rem' }}>
        <h3 style={{ fontSize: '.98rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color, display: 'flex' }}>{icon}</span>
          {title}
        </h3>
        {actions}
      </div>
      {description && (
        <p style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', margin: '0 0 .9rem', lineHeight: 1.45 }}>{description}</p>
      )}
      {children}
    </section>
  )
}
