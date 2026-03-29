CREATE OR REFRESH MATERIALIZED VIEW gold_price_analytics
COMMENT 'Aggregated price analytics per commodity-market-month. Provides summary statistics for dashboards and Genie queries.'
AS
SELECT
  commodity,
  market,
  state,
  year,
  month,
  COUNT(*) as total_records,
  COUNT(DISTINCT arrival_date) as trading_days,
  ROUND(AVG(modal_price), 2) as avg_modal_price,
  ROUND(MIN(modal_price), 2) as min_modal_price,
  ROUND(MAX(modal_price), 2) as max_modal_price,
  ROUND(STDDEV(modal_price), 2) as price_stddev,
  ROUND(AVG(min_price), 2) as avg_min_price,
  ROUND(AVG(max_price), 2) as avg_max_price
FROM silver_mandi_prices
GROUP BY commodity, market, state, year, month
