import api from '../../../api/client';

// Meetings
export const listMeetings = () => api.get('/api/rom-training/meetings').then(r => r.data);
export const getLongRomPoints = (recordingId: string) =>
  api.get(`/api/rom-training/meetings/${recordingId}/long-rom-points`).then(r => r.data);
export const getDataForMeeting = (recordingId: string) =>
  api.get(`/api/rom-training/data/meeting/${recordingId}`).then(r => r.data);

// Data Creation
export const extractMom = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/rom-training/extract-mom', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};
export const matchAgendas = (momAgendas: any[], meetingAgendas: any[]) =>
  api.post('/api/rom-training/match-agendas', { mom_agendas: momAgendas, meeting_agendas: meetingAgendas }).then(r => r.data);
export const saveTrainingData = (body: any) =>
  api.post('/api/rom-training/data', body).then(r => r.data);

// Database
export const listTrainingData = () => api.get('/api/rom-training/data').then(r => r.data);
export const deleteTrainingData = (id: string) => api.delete(`/api/rom-training/data/${id}`).then(r => r.data);

// Training
export const startTraining = (body: any) => api.post('/api/rom-training/variants', body).then(r => r.data);
export const listVariants = () => api.get('/api/rom-training/variants').then(r => r.data);
export const getVariant = (id: string) => api.get(`/api/rom-training/variants/${id}`).then(r => r.data);
export const deleteVariant = (id: string) => api.delete(`/api/rom-training/variants/${id}`).then(r => r.data);
