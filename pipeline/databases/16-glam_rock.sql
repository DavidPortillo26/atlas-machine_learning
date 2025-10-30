-- Glam rock bands ordered by lifespan through 2020
SELECT band_name,
       COALESCE(NULLIF(split, 0), 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC, band_name;
