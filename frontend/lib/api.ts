import axios from 'axios';
import { useAuthStore } from './store';
import {
  Token,
  User,
  Document,
  ChatResponse,
  EvaluationRun,
  Experiment,
  HealthStatus,
  ComparisonResult,
  Source
} from '@/types';

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const apiClient = {
  auth: {
    login: async (username: string, password: string): Promise<Token> => {
      const data = new URLSearchParams();
      data.append('username', username);
      data.append('password', password);
      return api
        .post<Token>('/api/auth/login', data, {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        })
        .then((r) => r.data);
    },
    register: async (email: string, password: string, full_name: string): Promise<Token> =>
      api.post<Token>('/api/auth/register', { email, password, full_name }).then((r) => r.data),
    me: async (): Promise<User> => api.get<User>('/api/auth/me').then((r) => r.data)
  },
  documents: {
    list: async (): Promise<Document[]> => api.get<Document[]>('/api/documents').then((r) => r.data),
    get: async (id: string): Promise<Document> => api.get<Document>(`/api/documents/${id}`).then((r) => r.data),
    upload: async (file: File): Promise<Document> => {
      const form = new FormData();
      form.append('file', file);
      return api.post<Document>('/api/documents/upload', form).then((r) => r.data);
    },
    delete: async (id: string): Promise<{ message: string }> =>
      api.delete<{ message: string }>(`/api/documents/${id}`).then((r) => r.data),
    summary: async (id: string): Promise<{ summary: string; sources: Source[] }> =>
      api.post<{ summary: string; sources: Source[] }>(`/api/documents/${id}/summary`).then((r) => r.data),
    compare: async (ids: string[], query?: string): Promise<ComparisonResult> =>
      api.post<ComparisonResult>('/api/documents/compare', { document_ids: ids, query }).then((r) => r.data)
  },
  chat: {
    send: async (query: string, conversation_id?: string): Promise<ChatResponse> =>
      api.post<ChatResponse>('/api/chat', { query, conversation_id }).then((r) => r.data),
    stream: (query: string, conversation_id?: string): EventSource => {
      const token = useAuthStore.getState().token;
      const base = (api.defaults.baseURL || '').replace(/\/+$/, '');
      const url = new URL(`${base}/api/chat/stream`);
      if (token) url.searchParams.append('token', token);
      url.searchParams.append('query', query);
      if (conversation_id) url.searchParams.append('conversation_id', conversation_id);
      return new EventSource(url.toString());
    }
  },
  feedback: {
    submit: async (message_id: string, rating: number, text?: string): Promise<any> =>
      api.post('/api/feedback', { message_id, rating, feedback_text: text }).then((r) => r.data)
  },
  evaluations: {
    list: async (): Promise<EvaluationRun[]> => api.get<EvaluationRun[]>('/api/evaluations').then((r) => r.data),
    run: async (name?: string): Promise<EvaluationRun> =>
      api.post<EvaluationRun>('/api/evaluations/run', { name }).then((r) => r.data)
  },
  experiments: {
    list: async (): Promise<Experiment[]> => api.get<Experiment[]>('/api/experiments').then((r) => r.data)
  },
  health: {
    check: async (): Promise<HealthStatus> => api.get<HealthStatus>('/api/health').then((r) => r.data)
  }
};