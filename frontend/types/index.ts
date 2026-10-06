export interface User { id: string; email: string; full_name: string; is_active: boolean; created_at: string; }
export interface Document { id: string; user_id: string; filename: string; original_filename: string; document_type: string; status: 'pending'|'processing'|'completed'|'failed'; page_count: number; chunk_count: number; created_at: string; updated_at: string; }
export interface Message { id: string; conversation_id: string; user_id: string; role: 'user'|'assistant'; content: string; sources: Source[]; created_at: string; }
export interface Source { document: string; page: number; chunk_id: string; score?: number; }
export interface ChatResponse { answer: string; sources: Source[]; confidence: number; conversation_id: string; message_id: string; }
export interface StreamEvent { type: 'token'|'sources'|'done'|'error'; content?: string; sources?: Source[]; }
export interface EvaluationRun { id: string; name: string; status: string; config: any; results: any; mlflow_run_id: string; started_at: string; completed_at: string; }
export interface HealthStatus { status: string; database: string; redis: string; pinecone: string; version: string; }
export interface Experiment { experiment_id: string; name: string; artifact_location: string; lifecycle_stage: string; }
export interface Token { access_token: string; token_type: string; }
export interface ComparisonResult { summary: string; similarities: string[]; differences: string[]; sources: Source[]; }