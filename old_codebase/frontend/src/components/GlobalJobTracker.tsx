/**
 * GlobalJobTracker — null-rendering daemon component.
 *
 * Mounted once in App.tsx (above Routes, inside HashRouter).
 * It activates the global job poller which tracks ALL in-flight jobs
 * regardless of which page the user is currently viewing.
 * Returns null — renders nothing in the DOM.
 */
import { useGlobalJobPoller } from '../hooks/useGlobalJobPoller'

export default function GlobalJobTracker() {
  useGlobalJobPoller()
  return null
}
