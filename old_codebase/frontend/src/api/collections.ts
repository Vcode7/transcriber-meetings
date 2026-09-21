import api from './client'
import type { Collection, CollectionDetail } from '../types/recording'

const BASE = '/collections'

export async function createCollection(name: string, description: string = ''): Promise<Collection> {
  const res = await api.post(BASE, { name, description })
  return res.data
}

export async function listCollections(): Promise<Collection[]> {
  const res = await api.get(BASE)
  return res.data
}

export async function getCollection(id: string, sort: string = 'manual'): Promise<CollectionDetail> {
  const res = await api.get(`${BASE}/${id}`, { params: { sort } })
  return res.data
}

export async function updateCollection(id: string, data: { name?: string; description?: string }): Promise<Collection> {
  const res = await api.patch(`${BASE}/${id}`, data)
  return res.data
}

export async function deleteCollection(id: string): Promise<void> {
  await api.delete(`${BASE}/${id}`)
}

export async function addMeetingsToCollection(collectionId: string, meetingIds: string[]): Promise<{ added_count: number }> {
  const res = await api.post(`${BASE}/${collectionId}/meetings`, { meeting_ids: meetingIds })
  return res.data
}

export async function removeMeetingsFromCollection(collectionId: string, meetingIds: string[]): Promise<{ removed_count: number }> {
  const res = await api.delete(`${BASE}/${collectionId}/meetings`, { data: { meeting_ids: meetingIds } })
  return res.data
}

export async function reorderMeetings(collectionId: string, meetingIds: string[]): Promise<void> {
  await api.patch(`${BASE}/${collectionId}/reorder`, { meeting_ids: meetingIds })
}
