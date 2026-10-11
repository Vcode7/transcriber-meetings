import type { ReactNode } from 'react'
import { AlertCircle, AlignLeft, FileText, Gauge, Layers, RotateCw, ScanSearch, Type, WandSparkles } from 'lucide-react'
import type { OcrPreset, OcrSchema, OcrSettings, OcrSettingValue } from '../types'
import { isFieldVisible, settingsEqual, type FieldErrors } from '../utils'
import { SectionCard, SettingField } from './Fields'

const GROUP_STYLE: Record<string, { icon: ReactNode; color: string }> = {
  preprocessing: { icon: <WandSparkles size={16} />, color: 'hsl(280 65% 58%)' },
  detection: { icon: <ScanSearch size={16} />, color: 'hsl(205 85% 50%)' },
  orientation: { icon: <RotateCw size={16} />, color: 'hsl(35 90% 48%)' },
  recognition: { icon: <Type size={16} />, color: 'hsl(150 60% 40%)' },
  layout: { icon: <AlignLeft size={16} />, color: 'hsl(330 70% 55%)' },
  pdf: { icon: <FileText size={16} />, color: 'hsl(14 85% 55%)' },
  performance: { icon: <Gauge size={16} />, color: 'hsl(250 65% 60%)' },
}

interface OcrSettingsFormProps {
  schema: OcrSchema
  values: OcrSettings
  saved: OcrSettings
  presets: OcrPreset[]
  errors: FieldErrors
  onChange: (name: string, value: OcrSettingValue) => void
  onApplyPreset: (preset: OcrPreset) => void
}

export default function OcrSettingsForm({
  schema, values, saved, presets, errors, onChange, onApplyPreset,
}: OcrSettingsFormProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
      <SectionCard
        title="Presets"
        icon={<Layers size={16} />}
        color="hsl(14 90% 56%)"
        description="Start from a preset and fine-tune below. Presets only fill in the form; nothing changes until you save."
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '.6rem' }}>
          {presets.map(preset => {
            const active = settingsEqual({ ...values, schema_version: 0 }, { ...preset.settings, schema_version: 0 })
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => onApplyPreset(preset)}
                aria-pressed={active}
                style={{
                  textAlign: 'left', padding: '.7rem .8rem', borderRadius: 10, cursor: 'pointer',
                  background: active ? 'hsl(var(--accent) / .08)' : 'hsl(var(--paper) / .4)',
                  border: active ? '1.5px solid hsl(var(--accent) / .55)' : '1px solid hsl(var(--border) / .5)',
                  fontFamily: 'Inter, sans-serif', transition: 'all .15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{preset.label}</span>
                  {preset.id === 'official' && <Badge color="hsl(150 60% 38%)">Recommended</Badge>}
                  {active && <Badge color="hsl(14 90% 50%)">In form</Badge>}
                </div>
                <div style={{ fontSize: '.69rem', color: 'hsl(var(--pencil))', lineHeight: 1.4 }}>{preset.description}</div>
              </button>
            )
          })}
        </div>
      </SectionCard>

      {errors._form && (
        <div role="alert" style={{
          display: 'flex', gap: 8, alignItems: 'center', padding: '.65rem .85rem', borderRadius: 10,
          background: 'hsl(var(--destructive) / .08)', color: 'hsl(var(--destructive))', fontSize: '.78rem', fontWeight: 600,
        }}>
          <AlertCircle size={15} /> {errors._form}
        </div>
      )}

      {schema.groups.map(group => {
        const fields = schema.fields.filter(f => f.group === group.id && isFieldVisible(f, values))
        if (!fields.length) return null
        const style = GROUP_STYLE[group.id] ?? GROUP_STYLE.performance
        const changed = fields.filter(f => values[f.name] !== saved[f.name]).length
        return (
          <SectionCard
            key={group.id}
            title={group.title}
            description={group.description}
            icon={style.icon}
            color={style.color}
            actions={changed > 0 ? <Badge color="hsl(14 90% 50%)">{changed} unsaved</Badge> : undefined}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '.75rem' }}>
              {fields.map(field => (
                <SettingField
                  key={field.name}
                  field={field}
                  value={values[field.name]}
                  savedValue={saved[field.name]}
                  error={errors[field.name]}
                  onChange={value => onChange(field.name, value)}
                />
              ))}
            </div>
          </SectionCard>
        )
      })}
    </div>
  )
}

function Badge({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span style={{
      fontSize: '.62rem', fontWeight: 700, padding: '1px 6px', borderRadius: 6, whiteSpace: 'nowrap',
      color, background: color.replace(')', ' / .12)'), letterSpacing: '.02em',
    }}>
      {children}
    </span>
  )
}
