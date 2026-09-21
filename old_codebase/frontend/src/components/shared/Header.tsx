import { ReactNode } from 'react'

interface HeaderProps {
  icon?: ReactNode
  title: string
  subtitle?: string
  actions?: ReactNode
}

export default function Header({ icon, title, subtitle, actions }: HeaderProps) {
  return (
    <div className="panel-header">
      {icon}
      <div style={{ flex: 1 }}>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {actions}
    </div>
  )
}
