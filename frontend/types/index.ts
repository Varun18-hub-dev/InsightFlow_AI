export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  created_at: string;
}

export interface Document {
  id: string;
  user_id: string;
  filename: string;
  original_filename: string;
  document_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  page_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface Source {
  document: string;
  page: number | string;
  chunk_id: string;
  score?: number;
  snippet?: string;
}

export interface MessageMetadata {
  intent?: string;
  retrieved_count?: number;
  reranked_count?: number;
  context_characters?: number;
  model?: string;
  sources_count?: number;
  response_time_seconds?: number;
  generation_time_seconds?: number;
  [key: string]: any;
}

export interface Message {
  id: string;
  conversation_id: string;
  user_id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
  metadata?: MessageMetadata;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  conversation_id: string;
  message_id: string;
}

export interface StreamEvent {
  type: 'token' | 'sources' | 'done' | 'error';
  content?: string;
  sources?: Source[];
  conversation_id?: string;
  message_id?: string;
  metadata?: MessageMetadata;
  details?: string;
}

export interface EvaluationRun {
  id: string;
  name: string;
  status: string;
  config: any;
  results: any;
  mlflow_run_id: string | null;
  started_at: string;
  completed_at: string | null;
}

export interface ExperimentRun {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | string;
  config: {
    chunk_size?: number;
    chunk_overlap?: number;
    top_k_retrieval?: number;
    top_k_rerank?: number;
    reranker_type?: string;
    model?: string;
    [key: string]: any;
  };
  results?: {
    metrics?: {
      recall_at_5?: number;
      hit_rate_at_5?: number;
      mrr?: number;
      faithfulness?: number;
      answer_relevance?: number;
      total_latency_seconds?: number;
      questions_evaluated?: number;
      [key: string]: any;
    };
    per_question?: Array<{
      question: string;
      answer: string;
      sources: string[];
      confidence?: number;
    }>;
    error?: string;
    [key: string]: any;
  };
  error_message?: string | null;
  mlflow_run_id?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface ExperimentRunRequest {
  name?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  top_k_retrieval?: number;
  top_k_rerank?: number;
  reranker_type?: string;
  model?: string;
}

export interface HealthStatus {
  status: string;
  database: string;
  redis: string;
  pinecone: string;
  version: string;
  llm_provider?: string;
}

export interface Experiment {
  experiment_id: string;
  name: string;
  artifact_location?: string;
  lifecycle_stage?: string;
  runs?: any[];
}

export interface Token {
  access_token: string;
  token_type: string;
}

export interface ComparisonResult {
  summary: string;
  similarities: string[];
  differences: string[];
  sources: Source[];
}