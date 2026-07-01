import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  Mic, Upload, History, UserPlus, Settings,
  LogOut, Zap, PanelLeftClose, PanelLeftOpen,
  Sun, Moon, MonitorSpeaker, Sparkles, Lock, Loader, BookOpen,
} from 'lucide-react'
import { useAuthStore } from '../store/auth'
import { useUIStore } from '../store/ui'
import { useProcessingStore } from '../store/processing'
import api from '../api/client'

const NAV = [
  { to: '/dashboard', icon: Mic, label: 'Record', end: true },
  { to: '/dashboard/tab-audio', icon: MonitorSpeaker, label: 'Tab Audio' },
  { to: '/dashboard/upload', icon: Upload, label: 'Upload' },
  { to: '/dashboard/history', icon: History, label: 'History' },
  { to: '/dashboard/dictionary', icon: BookOpen, label: 'Dictionary' },
]
const VOICE_NAV = [
  { to: '/dashboard/add-voice', icon: UserPlus, label: 'Add Voice' },
  { to: '/dashboard/settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar() {
  const logout = useAuthStore((s) => s.logout)
  const user = useAuthStore((s) => s.user)
  const { theme, sidebarCollapsed: collapsed, toggleTheme, toggleSidebar } = useUIStore()
  const { isProcessing, stage } = useProcessingStore()
  const navigate = useNavigate()
  const [loggingOut, setLoggingOut] = useState(false)

  const handleLogout = async () => {
    if (isProcessing || loggingOut) return
    setLoggingOut(true)
    try {
      // Revoke session server-side (clears HttpOnly cookie)
      await api.post('/auth/logout')
    } catch {
      // Even if the request fails, clear local state
    } finally {
      logout()
      navigate('/login')
    }
  }

  const tip = (label: string) => collapsed ? label : undefined
  const avatarLetter = user?.name?.charAt(0).toUpperCase() ?? '?'
  const avatarColors = ['hsl(14,90%,56%)', 'hsl(205,90%,55%)', 'hsl(130,60%,45%)', 'hsl(280,70%,60%)', 'hsl(45,90%,50%)']
  const avatarColor = avatarColors[(user?.name?.charCodeAt(0) ?? 0) % avatarColors.length]

  // Stage labels for the banner
  const STAGE_LABELS: Record<string, string> = {
    uploading: 'Uploading…',
    queued: 'Queued…',
    transcribing: 'Transcribing…',
    diarizing: 'Diarizing…',
    identifying_speakers: 'Matching voices…',
    generating_insights: 'AI insights…',
  }

  const handleLockedClick = () => {
    // Visual feedback — the overlay on the page is the primary UX,
    // but if the sidebar is collapsed we still need some hint.
    // Nothing needed; the nav-locked class already blocks pointer events.
  }

  return (
    <aside className="sidebar" style={{ position: 'relative' }}>

      {/* Logo */}
      <div className="sidebar-logo">
        <div style={{ position: 'relative', flexShrink: 0 }}>
          <Zap
            size={22}
            fill="currentColor"
            style={{ color: 'hsl(var(--accent))' }}
            className="animate-float"
          />
          {/* Glow dot */}
          <span style={{
            position: 'absolute',
            top: -2, right: -2,
            width: 7, height: 7,
            borderRadius: '50%',
            background: isProcessing ? 'hsl(var(--accent))' : 'hsl(var(--accent))',
            boxShadow: '0 0 6px hsl(var(--accent))',
            border: '1.5px solid hsl(var(--card))',
          }} className="animate-pulse-rec" />
        </div>
        {!collapsed && (
          <span style={{ overflow: 'hidden', whiteSpace: 'nowrap', fontSize: '1.7rem', letterSpacing: '-0.02em' }}>
            Voice<span style={{ color: 'hsl(var(--accent))' }}>Sum</span>
          </span>
        )}
      </div>

      {/* ── Processing banner (shown when a job is running) ── */}
      {isProcessing && (
        <div className="processing-sidebar-banner">
          <Loader size={11} className="spin" style={{ flexShrink: 0 }} />
          {!collapsed && (
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {STAGE_LABELS[stage ?? ''] || 'Processing…'}
            </span>
          )}
          {!collapsed && <Lock size={10} style={{ flexShrink: 0, marginLeft: 'auto', opacity: 0.7 }} />}
        </div>
      )}

      {/* Main nav */}
      {!collapsed && (
        <div className="nav-section">Workspace</div>
      )}
      {NAV.map(({ to, icon: Icon, label, end }) => (
        <NavLink
          key={to} to={to} end={end}
          className={({ isActive }) =>
            `nav-item ${isActive ? 'active' : ''} ${isProcessing ? 'nav-locked' : ''}`
          }
          title={isProcessing ? 'Processing in progress — please wait' : tip(label)}
          onClick={isProcessing ? (e) => e.preventDefault() : undefined}
          aria-disabled={isProcessing}
          tabIndex={isProcessing ? -1 : undefined}
        >
          <Icon size={16} className="nav-icon" />
          {!collapsed && <span className="nav-label">{label}</span>}
          {!collapsed && isProcessing && (
            <Lock size={10} style={{ marginLeft: 'auto', opacity: 0.5, flexShrink: 0 }} />
          )}
        </NavLink>
      ))}

      {/* Voice nav */}
      {!collapsed && <div className="nav-section" style={{ marginTop: '1rem' }}>Voice &amp; Config</div>}
      {collapsed && <div className="nav-divider" />}
      {VOICE_NAV.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to} to={to}
          className={({ isActive }) =>
            `nav-item ${isActive ? 'active' : ''} ${isProcessing ? 'nav-locked' : ''}`
          }
          title={isProcessing ? 'Processing in progress — please wait' : tip(label)}
          onClick={isProcessing ? (e) => e.preventDefault() : undefined}
          aria-disabled={isProcessing}
          tabIndex={isProcessing ? -1 : undefined}
        >
          <Icon size={16} className="nav-icon" />
          {!collapsed && <span className="nav-label">{label}</span>}
          {!collapsed && isProcessing && (
            <Lock size={10} style={{ marginLeft: 'auto', opacity: 0.5, flexShrink: 0 }} />
          )}
        </NavLink>
      ))}

      <div className="sidebar-spacer" style={{ flex: 1 }} />

      {/* Bottom section */}
      <div className="sidebar-bottom" style={{ padding: '0 .5rem', display: 'flex', flexDirection: 'column', gap: '2px' }}>

        {/* Theme toggle */}
        <button
          className="nav-item"
          onClick={toggleTheme}
          title={collapsed ? (theme === 'dark' ? 'Light mode' : 'Dark mode') : undefined}
          style={{ gap: collapsed ? 0 : 11 }}
        >
          {theme === 'dark'
            ? <Sun size={16} className="nav-icon" style={{ color: 'hsl(45,90%,55%)' }} />
            : <Moon size={16} className="nav-icon" style={{ color: 'hsl(235,70%,65%)' }} />}
          {!collapsed && (
            <span className="nav-label" style={{ color: theme === 'dark' ? 'hsl(45,90%,55%)' : 'hsl(235,70%,65%)' }}>
              {theme === 'dark' ? 'Light mode' : 'Dark mode'}
            </span>
          )}
        </button>

        {/* Collapse toggle */}
        <button
          className="sidebar-collapse-btn"
          onClick={toggleSidebar}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{ width: '100%', margin: '2px 0' }}
        >
          {collapsed
            ? <PanelLeftOpen size={16} />
            : <PanelLeftClose size={16} />}
        </button>

        <div className="nav-divider" />

        {/* User */}
        {!collapsed ? (
          <div style={{
            padding: '.6rem .75rem',
            display: 'flex', alignItems: 'center', gap: '10px',
            borderRadius: '12px',
            background: 'hsl(var(--sidebar-accent))',
            border: '1.5px solid hsl(var(--ink) / .08)',
            margin: '2px 0',
          }}>
            <div style={{
              width: '32px', height: '32px',
              borderRadius: '50%',
              background: `${avatarColor}22`,
              border: `2px solid ${avatarColor}60`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '.82rem', fontWeight: 800,
              color: avatarColor,
              flexShrink: 0,
              fontFamily: 'Inter, sans-serif',
              boxShadow: `0 0 8px ${avatarColor}30`,
            }}>
              {avatarLetter}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                overflow: 'hidden', textOverflow: 'ellipsis',
                fontSize: '.82rem', fontWeight: 600,
                color: 'hsl(var(--sidebar-foreground))',
                fontFamily: 'Inter, sans-serif',
                whiteSpace: 'nowrap',
                lineHeight: 1.3,
              }}>
                {user?.name}
              </div>
              {user?.email && (
                <div style={{
                  overflow: 'hidden', textOverflow: 'ellipsis',
                  fontSize: '.68rem', fontWeight: 400,
                  color: 'hsl(var(--pencil))',
                  fontFamily: 'Inter, sans-serif',
                  whiteSpace: 'nowrap',
                  opacity: 0.75,
                }}>
                  {user.email}
                </div>
              )}
            </div>
            <Sparkles size={12} style={{ color: avatarColor, flexShrink: 0, opacity: 0.7 }} />
          </div>
        ) : (
          <div
            title={user?.name ?? ''}
            className="tooltip"
            data-tooltip={user?.name ?? ''}
            style={{
              margin: '4px auto',
              width: '32px', height: '32px',
              borderRadius: '50%',
              background: `${avatarColor}22`,
              border: `2px solid ${avatarColor}60`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '.82rem', fontWeight: 800,
              color: avatarColor,
              cursor: 'default',
              fontFamily: 'Inter, sans-serif',
              boxShadow: `0 0 8px ${avatarColor}30`,
            }}>
            {avatarLetter}
          </div>
        )}

        <button
          className={`nav-item ${isProcessing ? 'nav-locked' : ''}`}
          style={{ color: 'hsl(var(--destructive))', margin: '2px 0' }}
          onClick={handleLogout}
          title={isProcessing ? 'Cannot log out while processing' : tip('Logout')}
          disabled={isProcessing || loggingOut}
        >
          {loggingOut
            ? <Loader size={16} className="spin nav-icon" />
            : <LogOut size={16} className="nav-icon" />}
          {!collapsed && <span className="nav-label">{loggingOut ? 'Signing out…' : 'Logout'}</span>}
        </button>
      </div>
    </aside>
  )
}
