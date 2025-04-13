CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    case_id TEXT,
    case_title TEXT,
    case_text TEXT,
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', case_text)) STORED, -- FTS vector
    embedding vector(512) -- Ensure dimension matches your model (e.g., 512)
);

-- 2. Enable Row Level Security (Recommended)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- 3. Add Comments (Optional)
COMMENT ON COLUMN documents.case_text IS 'Stores the main case text, used for FTS.';
COMMENT ON COLUMN documents.fts IS 'Generated tsvector for full-text search on the case_text column.';
COMMENT ON COLUMN documents.embedding IS 'Stores the embedding vector for semantic search.';
COMMENT ON COLUMN documents.case_id IS 'Original identifier from the case data source.';

-- 4. Create Indexes for Performance
-- Index for Full-Text Search
CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents USING GIN(fts);
-- Index for Vector Search (using inner product)
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING HNSW (embedding vector_ip_ops);

-- 5. Grant Base Permissions
GRANT SELECT ON TABLE documents TO anon, authenticated; -- Potential read access
GRANT INSERT ON TABLE documents TO service_role;       -- Allow backend to insert
GRANT USAGE, SELECT ON SEQUENCE documents_id_seq TO service_role; -- For ID generation

-- 6. Define RLS Policies (Crucial if RLS is enabled and using anon/auth keys)
CREATE POLICY "Allow authenticated read access"
ON documents
FOR SELECT
TO authenticated
USING (true);

-- 7. Create the Hybrid Search Function
CREATE OR REPLACE FUNCTION hybrid_search(
  query_text text,              -- The user's text query
  query_embedding vector(512),  -- The embedding vector for the query (match dimension)
  match_count int,              -- How many results to return
  full_text_weight float = 1.0, -- Weight for FTS results in RRF
  semantic_weight float = 1.0,  -- Weight for semantic results in RRF
  rrf_k int = 50                -- RRF smoothing constant
)
-- The function returns rows that match the structure of the 'documents' table
returns setof documents
language sql
as $$
with full_text as (
  -- Perform Full-Text Search
  select
    id,
    -- Calculate rank based on FTS relevance score
    row_number() over(order by ts_rank_cd(fts, websearch_to_tsquery('english', query_text)) desc) as rank_ix
  from
    documents
  where
    -- Match documents where the 'fts' vector matches the query text
    fts @@ websearch_to_tsquery('english', query_text) -- Use 'english' or your text config
  order by rank_ix
  -- Fetch more results initially to allow for better ranking/fusion
  limit least(match_count, 30) * 2
),
semantic as (
  -- Perform Semantic Search
  select
    id,
    -- Calculate rank based on vector distance (inner product '<#>')
    row_number() over (order by embedding <#> query_embedding) as rank_ix
  from
    documents
  order by rank_ix
  -- Fetch more results initially
  limit least(match_count, 30) * 2
)
-- Combine results using Reciprocal Rank Fusion (RRF)
select
  documents.* -- Select all columns from the original documents table
from
  full_text
  -- Use FULL OUTER JOIN to include results found only in one search type
  full outer join semantic
    on full_text.id = semantic.id
  -- Join back to the documents table to get the full row data
  join documents
    on coalesce(full_text.id, semantic.id) = documents.id
order by
  -- Calculate the RRF score, giving 0 score if a document wasn't found in a list
  (coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight) +
  (coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight)
  desc -- Order by highest RRF score first
limit
  -- Return the requested number of results (up to a max of 30 for safety)
  least(match_count, 30)
$$;

-- Grant execution permission for the function to roles that will call it via RPC
GRANT EXECUTE ON FUNCTION hybrid_search TO anon, authenticated, service_role;
