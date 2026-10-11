export const STATUS_STYLE: Record<string, { label: string; color: string }> = {
  completed: { label: 'Completed', color: 'hsl(150 60% 36%)' },
  ongoing: { label: 'Ongoing', color: 'hsl(205 85% 45%)' },
  delayed: { label: 'Delayed', color: 'hsl(0 72% 50%)' },
  at_risk: { label: 'At risk', color: 'hsl(25 90% 48%)' },
  planned: { label: 'Planned', color: 'hsl(265 60% 55%)' },
  mixed: { label: 'Mixed', color: 'hsl(220 10% 46%)' },
  unknown: { label: 'Status n/a', color: 'hsl(220 10% 52%)' },
}

export function statusColor(status: string): string {
  return (STATUS_STYLE[status] ?? STATUS_STYLE.unknown).color
}
