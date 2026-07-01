import { useEffect } from 'react'
import { Outlet, Navigate, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Sidebar from '../components/Sidebar'
import { useAuthStore } from '../store/auth'
import { useUIStore } from '../store/ui'
import { useProcessingStore } from '../store/processing'

export default function Dashboard() {
  const user = useAuthStore((s) => s.user)
  const collapsed = useUIStore((s) => s.sidebarCollapsed)
  const isProcessing = useProcessingStore((s) => s.isProcessing)
  const location = useLocation()

  // ── Block browser back / forward while processing ──────────────────────────
  // Push a duplicate history entry so pressing "Back" stays on the same page.
  useEffect(() => {
    if (!isProcessing) return

    // Push a guard entry
    window.history.pushState(null, '', window.location.href)

    const onPopState = () => {
      if (useProcessingStore.getState().isProcessing) {
        // Re-push so user stays on the same page
        window.history.pushState(null, '', window.location.href)
      }
    }

    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [isProcessing])

  // ── Warn before tab close / page refresh ──────────────────────────────────
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!useProcessingStore.getState().isProcessing) return
      e.preventDefault()
      e.returnValue = 'Audio is still being processed. If you leave now, results will be lost.'
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [])

  if (!user) return <Navigate to="/login" replace />
  if (user.needs_setup) return <Navigate to="/setup" replace />

  const sidebarW = collapsed ? '64px' : '220px'

  return (
    <div
      className="dashboard-layout"
      style={{
        gridTemplateColumns: `${sidebarW} 1fr`,
      }}
    >
      <Sidebar />
      <main className="dashboard-content" style={{
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
      }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, scale: 0.975, y: 6 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.99 }}
            transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
            className="dashboard-route"
            style={{ display: 'flex', flexDirection: 'column' }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
