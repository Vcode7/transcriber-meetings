import type { JobStatus } from '../store/jobs'

export type ActiveJobStatus = Extract<JobStatus, 'pending' | 'processing' | 'transcript_ready'>

export function isActiveJobStatus(status: JobStatus | string | null | undefined): status is ActiveJobStatus {
  return status === 'pending' || status === 'processing' || status === 'transcript_ready'
}
