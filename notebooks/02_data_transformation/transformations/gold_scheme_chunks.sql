CREATE OR REFRESH MATERIALIZED VIEW gold_scheme_chunks
COMMENT 'Government schemes chunked for RAG vector search. Creates overview and application chunks per scheme for FAISS embedding.'
AS
WITH overview_chunks AS (
  SELECT
    scheme_id,
    name_en,
    category,
    'overview' as chunk_type,
    CONCAT(
      'Scheme: ', COALESCE(name_en, ''), '\n',
      'Category: ', COALESCE(category, ''), '\n',
      'Description: ', COALESCE(description, ''), '\n',
      'Benefits: ', COALESCE(benefits, ''), '\n',
      'Coverage: ', COALESCE(coverage, '')
    ) as chunk_text
  FROM govt_schemes
  WHERE scheme_id IS NOT NULL
),
application_chunks AS (
  SELECT
    scheme_id,
    name_en,
    category,
    'application' as chunk_type,
    CONCAT(
      'Scheme: ', COALESCE(name_en, ''), '\n',
      'Eligibility: ', COALESCE(CAST(eligibility AS STRING), ''), '\n',
      'How to Apply: ', COALESCE(how_to_apply, ''), '\n',
      'Documents Required: ', COALESCE(CAST(documents_required AS STRING), ''), '\n',
      'Helpline: ', COALESCE(helpline, ''), '\n',
      'Official URL: ', COALESCE(official_url, '')
    ) as chunk_text
  FROM govt_schemes
  WHERE scheme_id IS NOT NULL
)
SELECT * FROM overview_chunks
UNION ALL
SELECT * FROM application_chunks
