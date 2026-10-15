import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import api from '../api/client'

/**
 * The user's minimum action-point importance (0 = show all), saved in /settings as
 * action_point_min_importance. Action points scored below it are hidden and left out of DOCX exports.
 */
export function useImportanceThreshold(): [number, (value: number) => void] {
  const [threshold, setThreshold] = useState(0)

  useEffect(() => {
    let cancelled = false
    api.get('/settings').then(res => {
      const value = Number(res.data?.action_point_min_importance)
      if (!cancelled && Number.isFinite(value)) setThreshold(value)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  const update = useCallback((value: number) => {
    setThreshold(value)
    api.put('/settings', { action_point_min_importance: value })
      .catch(() => toast.error('Could not save the minimum importance setting'))
  }, [])

  return [threshold, update]
}
