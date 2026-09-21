import { create } from 'zustand'

type Theme = 'dark' | 'light'

interface UIState {
  theme: Theme
  sidebarCollapsed: boolean
  toggleTheme: () => void
  toggleSidebar: () => void
}

function applyTheme(t: Theme) {
  document.documentElement.setAttribute('data-theme', t)
}

export const useUIStore = create<UIState>((set) => {
  // Default to the cream "paper" theme from the Sketchy design prompt.
  // Migrate any previously-saved 'dark' default to 'light' once.
  const stored = localStorage.getItem('vs_theme') as Theme | null
  if (!stored || !localStorage.getItem('vs_theme_v2')) {
    localStorage.setItem('vs_theme', 'light')
    localStorage.setItem('vs_theme_v2', '1')
  }
  const savedTheme = (localStorage.getItem('vs_theme') as Theme) || 'light'
  const savedCollapsed = localStorage.getItem('vs_sidebar') === 'true'
  applyTheme(savedTheme)

  return {
    theme: savedTheme,
    sidebarCollapsed: savedCollapsed,

    toggleTheme: () =>
      set((s) => {
        const next: Theme = s.theme === 'dark' ? 'light' : 'dark'
        localStorage.setItem('vs_theme', next)
        applyTheme(next)
        return { theme: next }
      }),

    toggleSidebar: () =>
      set((s) => {
        const next = !s.sidebarCollapsed
        localStorage.setItem('vs_sidebar', String(next))
        return { sidebarCollapsed: next }
      }),
  }
})
