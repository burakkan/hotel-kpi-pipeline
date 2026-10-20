-- Monthly RevPAR and occupancy by room type
SELECT
    substr(date, 1, 7)                 AS month,
    room_type,
    ROUND(AVG(occupancy) * 100, 1)     AS avg_occupancy_pct,
    ROUND(SUM(revenue_try) * 1.0 / SUM(rooms_available), 0) AS revpar_try,
    ROUND(SUM(revenue_try) * 1.0 / NULLIF(SUM(rooms_sold), 0), 0) AS adr_try
FROM daily_kpis
GROUP BY month, room_type
ORDER BY month, room_type;

-- Days flagged as anomalies, largest deviations first
SELECT
    date,
    room_type,
    ROUND(revpar_try, 0) AS revpar_try,
    ROUND(z_score, 2)    AS z_score
FROM daily_kpis
WHERE is_anomaly = 1
ORDER BY ABS(z_score) DESC
LIMIT 20;
