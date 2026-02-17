-- Add cron expression column and migrate from interval_secs
ALTER TABLE cron_jobs ADD COLUMN schedule TEXT;

UPDATE cron_jobs SET schedule = CASE
    WHEN interval_secs <= 60 THEN '* * * * *'
    WHEN interval_secs <= 3600 THEN '0 * * * *'
    WHEN interval_secs <= 86400 THEN '0 0 * * *'
    ELSE '0 0 * * *'
END
WHERE schedule IS NULL;
