CREATE OR REFRESH STREAMING TABLE silver_mandi_prices (
  CONSTRAINT valid_price EXPECT (modal_price > 0) ON VIOLATION DROP ROW,
  CONSTRAINT valid_date EXPECT (arrival_date IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_commodity EXPECT (commodity IS NOT NULL AND LENGTH(TRIM(commodity)) > 0) ON VIOLATION DROP ROW
)
COMMENT 'Cleaned mandi prices with derived time features. Filters invalid prices and null dates. Deduplicates by date+market+commodity+variety. Ready for window function enrichment in notebook 02.'
AS
SELECT
  price_id,
  state,
  district,
  market,
  commodity,
  variety,
  grade,
  arrival_date,
  min_price,
  max_price,
  modal_price,
  -- Time features
  YEAR(arrival_date) as year,
  MONTH(arrival_date) as month,
  DAYOFWEEK(arrival_date) as day_of_week,
  source_file
FROM STREAM(mandi_prices_raw)
WHERE
  state IS NOT NULL
  AND market IS NOT NULL
