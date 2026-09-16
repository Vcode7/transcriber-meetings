/**
 * embeddingTrainingApi.ts — API client for ADA domain embedding training and variant management.
 */
import api from '../api/client';

export interface BaseModelInfo {
  id: string;
  name: string;
  path: string;
  dimension: number;
  architecture: string;
  size_mb: number;
  is_default: boolean;
  status: string;
}

export interface DocumentStats {
  filename: string;
  size_bytes: number;
  word_count: number;
  char_count: number;
  status: 'extracted' | 'error';
  error?: string | null;
}

export interface DomainTerm {
  term: string;
  count: number;
}

export interface ExtractionResult {
  session_id: string;
  total_documents: number;
  successful_documents: number;
  failed_documents: number;
  total_cleaned_words: number;
  total_cleaned_chars: number;
  total_chunks: number;
  deduplicated_chunks: number;
  domain_terms: DomainTerm[];
  document_stats: DocumentStats[];
  sample_chunks: string[];
}

export interface LossPoint {
  step: number;
  epoch: number;
  loss: number;
}

export interface TrainingJobStatus {
  job_id: string;
  variant_name: string;
  base_model_name: string;
  base_model_path: string;
  status: 'initializing' | 'training' | 'completed' | 'failed' | 'cancelling' | 'cancelled';
  total_chunks: number;
  epochs: number;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  current_loss: number | null;
  final_loss: number | null;
  progress_percent: number;
  elapsed_seconds: number;
  eta_seconds: number;
  loss_history: LossPoint[];
  logs: string[];
  error?: string | null;
  staged_model_path?: string | null;
}

export interface EmbeddingVariant {
  id: string;
  name: string;
  base_model: string;
  created_at: string;
  document_count: number;
  document_names: string[];
  chunk_count: number;
  status: string;
  model_path: string;
  size_bytes: number;
  size_mb: number;
  metrics: {
    final_loss?: number;
    epochs?: number;
    learning_rate?: number;
    duration_seconds?: number;
    [key: string]: any;
  };
  is_active: boolean;
}

export const embeddingTrainingApi = {
  /** Fetch available local base embedding models */
  getBaseModels: async (): Promise<BaseModelInfo[]> => {
    const res = await api.get('/api/embedding-training/base-models');
    return res.data.models;
  },

  /** Upload and extract text from documents */
  extractDocuments: async (files: File[]): Promise<ExtractionResult> => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    const res = await api.post('/api/embedding-training/extract-documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  /** Trigger domain adaptation training */
  startTraining: async (params: {
    base_model_name: string;
    variant_name: string;
    session_id: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    temperature?: number;
  }): Promise<{ job_id: string; status: string }> => {
    const res = await api.post('/api/embedding-training/train', params);
    return res.data;
  },

  /** Check ongoing training status */
  getTrainingStatus: async (job_id: string): Promise<TrainingJobStatus> => {
    const res = await api.get(`/api/embedding-training/train/status/${job_id}`);
    return res.data;
  },

  /** Cancel an active training job */
  cancelTraining: async (job_id: string): Promise<{ status: string }> => {
    const res = await api.post(`/api/embedding-training/train/cancel/${job_id}`);
    return res.data;
  },

  /** Save trained variant to persistent disk and database */
  saveVariant: async (params: {
    variant_name: string;
    base_model_name: string;
    staged_model_path: string;
    document_count: number;
    document_names: string[];
    chunk_count: number;
    metrics: Record<string, any>;
  }): Promise<EmbeddingVariant> => {
    const res = await api.post('/api/embedding-training/variants/save', params);
    return res.data;
  },

  /** List all saved embedding variants */
  getVariants: async (): Promise<EmbeddingVariant[]> => {
    const res = await api.get('/api/embedding-training/variants');
    return res.data.variants;
  },

  /** Select an embedding variant for active RAG pipeline use */
  selectVariant: async (variant_id: string): Promise<{ status: string; active_variant_name: string }> => {
    const res = await api.post(`/api/embedding-training/variants/${variant_id}/select`);
    return res.data;
  },

  /** Delete a saved variant */
  deleteVariant: async (variant_id: string): Promise<{ status: string; name: string }> => {
    const res = await api.delete(`/api/embedding-training/variants/${variant_id}`);
    return res.data;
  },
};
