-- Supabase table definitions for documents and chats
-- Run these in Supabase SQL editor or via psql connected to your Supabase Postgres

-- Documents: stores metadata and OCR/extracted text as JSON
CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  file_name text NOT NULL,
  file_path text, -- optional: path or storage key in Supabase Storage
  file_size bigint,
  extracted_text text, -- raw extracted text
  cleaned_text text,   -- cleaned/normalized text used for LLM
  ocr_metadata jsonb,  -- per-page OCR metadata (optional)
  processed_at timestamptz,
  created_at timestamptz DEFAULT now()
);

-- Chats: store per-document chat messages to support chat-like interactions
CREATE TABLE IF NOT EXISTS chats (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
  role text NOT NULL, -- 'user' | 'assistant' | 'system'
  message text NOT NULL,
  metadata jsonb, -- optional structured metadata (e.g. model, tokens)
  created_at timestamptz DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents(created_at);
CREATE INDEX IF NOT EXISTS chats_document_id_idx ON chats(document_id);
