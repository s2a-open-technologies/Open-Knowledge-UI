export interface ChatSession {
  id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  created_at?: string | null;
}

export interface SessionWithMessages extends ChatSession {
  messages: ChatMessage[];
}

export interface UserDocument {
  id: number;
  document_id: string;
  filename: string;
  uploaded_at: string | null;
  content_preview: string;
  source: string;
  has_data_file: boolean;
  has_markdown: boolean;
}

export interface DataFile {
  filename: string;
  size: number;
  size_human: string;
  modified: number;
}

export interface VectorDocument {
  id: string;
  filename: string;
  source: string;
  content_preview: string;
  has_data_file: boolean;
}

export interface VectorListResponse {
  documents: VectorDocument[];
  next_offset: string | number | null;
  filenames: string[];
  limit: number;
}

export interface VectorDocumentDetail {
  id: string;
  page_content: string;
  original_content: string;
  metadata: Record<string, unknown>;
}

export interface SummaryFile {
  filename: string;
  size: number;
  size_human: string;
  modified: number;
  created_at: string;
}

export interface SummaryContent {
  filename: string;
  content: string;
}

export interface Knowledge {
  id: number;
  name: string;
  description: string;
  collection_name: string;
  is_default: boolean;
  embedding_model: string | null;
  embedding_dimension: number | null;
  is_public: boolean;
  retrieval_profile_id: number | null;
  created_at: string | null;
  can_write: boolean;
  document_count: number;
  chunk_count: number | null;
  // Index passt nicht zum aktuellen Modell bzw. den Zerlege-Einstellungen
  stale?: boolean;
  last_reindex_error?: string | null;
  // 'external' = fremder Vektorspeicher, nur lesen
  kind?: "local" | "external" | "sql";
  source_config?: ExternalSource | SqlSource;
}

export interface SqlSource {
  dialect: "postgresql" | "mysql" | "mssql" | "sqlite";
  host: string;
  instance?: string;
  port: number;
  database: string;
  user: string;
  has_password: boolean;
  schema: string;
  tables: string[];
  blocked_columns: Record<string, string[]>;
}

export interface ExternalAccess {
  enabled: boolean;
  user_field: string;
  user_attr: "username" | "display_name";
  group_field: string;
  groups_from: "local" | "external" | "both";
  static: { field: string; values: string[] }[];
  admins_unfiltered: boolean;
}

export interface ExternalSource {
  kind: string;
  access?: ExternalAccess;
  url: string;
  has_key: boolean;
  collection: string;
  dense_vector: string;
  sparse_vector: string;
  fields: {
    text: string;
    filename: string;
    title: string;
    headers: string;
    page_start: string;
    page_end: string;
    chunk_index: string;
    group: string;
  };
}

export type AttachmentStatus = "uploading" | "processing" | "ready" | "error";

export interface ChatAttachment {
  // Lokale ID bis der Server die echte vergibt.
  id: string;
  filename: string;
  status: AttachmentStatus;
  error?: string | null;
}

export interface Connection {
  id: number;
  name: string;
  base_url: string;
  enabled: boolean;
  // llm = OpenAI-kompatibel, tei = Text Embeddings Inference, infinity = Infinity (Embedding/Rerank)
  kind: "llm" | "tei" | "infinity";
  // Der Key selbst wird nie ausgeliefert.
  has_key: boolean;
}

export interface ModelParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  num_ctx?: number;
  thinking?: string;
}

export interface ModelConfig {
  id: number;
  connection_id: number;
  base_model: string;
  name: string;
  is_clone: boolean;
  visible: boolean;
  is_public: boolean;
  system_prompt: string;
  web_search: boolean;
  vision?: boolean;
  params: ModelParams;
  knowledge_ids: number[] | null;
  // Nutzer dürfen zusätzlich eigenes Knowledge wählen (Standard an); aus = nur Modell-Knowledge
  allow_extra_knowledge?: boolean;
  ref: string;
}

export interface ModelEntry {
  ref: string;
  name: string;
  model_id: string;
  connection_id: number;
  connection_name: string;
  is_clone: boolean;
  customized: boolean;
  visible: boolean;
  available: boolean;
  // Nur für Admins; Nutzer bekommen stattdessen knowledge_locked.
  config?: ModelConfig | null;
  knowledge_locked?: boolean;
  // Knowledge, das das Modell mitbringt (immer Teil der Suche)
  knowledge_base?: { id: number; name: string }[];
  // Versteht Bilder: Bild-Anhänge gehen direkt an das Modell
  vision?: boolean;
}

export interface ModelsResponse {
  models: ModelEntry[];
  errors: { connection_id: number; connection: string; error: string }[];
  default: string;
  // Websuche pro Modell nur zuschaltbar, wenn SearXNG und PII-Filter eingerichtet sind
  web_search_available?: boolean;
  // true: der PII-Filter maskiert Suchanfragen; false: sie gehen unmaskiert raus (Warnhinweis)
  web_search_masked?: boolean;
  // Compaction-Einstellungen (Prozentwerte) für die Budget-Anzeige im Modell-Dialog
  compaction?: { enabled: boolean; headroom_percent: number; keep_percent: number; output_reserve_percent: number };
  // Standardwerte, die gelten, wenn ein Modell einen Parameter leer lässt
  param_defaults?: Partial<Record<"temperature" | "top_p" | "top_k" | "max_tokens" | "num_ctx" | "thinking", number | string | null>>;
}

export type Role = "user" | "editor" | "admin";

export interface AuthUser {
  id: number;
  username: string;
  display_name: string;
  role: Role;
  auth_source?: AuthSource;
}

export type AuthSource = "local" | "ldap" | "oidc";

export interface ManagedUser {
  id: number;
  username: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  auth_source?: AuthSource;
  group_ids: number[];
  created_at: string | null;
}

export interface Group {
  id: number;
  name: string;
  description: string;
  member_ids: number[];
}

export interface Principals {
  users: { id: number; name: string; username: string }[];
  groups: { id: number; name: string }[];
}

export type AccessLevel = "read" | "write";

export interface Grant {
  principal_type: "user" | "group";
  principal_id: number;
  name?: string;
  access: AccessLevel;
}

export interface AccessInfo {
  is_public: boolean;
  grants: Grant[];
}

export interface SearchConfig {
  mode: "hybrid" | "dense" | "sparse";
  fusion: "rrf" | "dbsf";
  dense_weight: number | "";
  sparse_weight: number | "";
  candidates: number | "";
  dense_min_score: number | "" | null;
}

export interface RerankConfig {
  enabled: boolean;
  model: string;
  candidates: number | "";
  min_score: number | "" | null;
}

export interface QueryEnhancementConfig {
  mode: "off" | "multi_query";
  n_queries: number | "";
  include_original: boolean;
  model_ref: string | null;
  prompt: string;
  temperature: number | "";
  timeout_s: number | "";
}

export interface ProfileConfig {
  search: SearchConfig;
  rerank: RerankConfig;
  mmr: { enabled: boolean; lambda_mult: number | "" };
  query_enhancement: QueryEnhancementConfig;
  final_k: number | "";
  neighbors: number | "";
}

export interface RetrievalProfile {
  id: number;
  name: string;
  description: string;
  is_default: boolean;
  config: ProfileConfig;
}

export interface TraceStage {
  stage: string;
  ms?: number;
  [key: string]: unknown;
}

export interface RetrievalTestDocument {
  filename: string;
  score: number;
  chunk_index: number | null;
  chunks: [number, number] | null;
  text: string;
}

export interface RetrievalTestResult {
  knowledge_id: number;
  knowledge: string;
  profile_id: number | null;
  documents: RetrievalTestDocument[];
  trace: { stages: TraceStage[]; total_ms: number };
}

export type ModelSource = "hf" | "endpoint" | "fastembed";

export interface ModelSpec {
  source: ModelSource;
  model: string;
  connection_id?: number;
}

export interface ModelStatus {
  state: "ready" | "loading" | "error" | "not_loaded";
  seconds?: number;
  dimension?: number | null;
  error?: string;
}

export interface TaskSettings {
  enabled: boolean;
  model_ref: string;
  temperature: number | null | "";
  max_tokens: number | null | "";
  prompt: string;
  default_prompt: string;
  excerpt_chars?: number;
  max_row_deviation?: number;
}

export interface IngestSettings {
  chunk_size: number;
  chunk_overlap: number;
  keep_original: boolean;
  index_enabled: boolean;
  // Sprache der Stichwortsuche (BM25)
  bm25_language: string;
}

export interface EmbeddingOverview {
  embedding: ModelSpec & { status: ModelStatus };
  reranker: ModelSpec & { status: ModelStatus };
  ingest: IngestSettings;
  bm25_languages?: string[];
  device?: string;
  tasks: Record<"vision" | "context" | "tables" | "images", TaskSettings>;
  stale: { id: number; name: string; error: string | null }[];
  suggestions: { embedding: string[]; reranker: string[] };
}

export interface ReindexStatus {
  running: boolean;
  queue: number[];
  current: { id: number; name: string; done: number; total: number } | null;
  results: { id: number; name: string; ok: boolean; error: string | null }[];
}

export interface MyUpload {
  kind: "chat" | "knowledge";
  id: string;
  filename: string;
  source: string;
  created_at: string | null;
  size: number | null;
  has_markdown: boolean;
}
