create table it_final.media as 
select 
	datetime,
	title,
	url,
	source,
	sentiment_label,
	'cryptonews' as table
from it_final.cryptonews
where datetime is not null
and sentiment_label is not null

union all

select 
	datetime,
	title,
	url,
	source,
	sentiment_label,
	'cryptopanic' as table
from it_final.cryptopanic_news
where datetime is not null
and sentiment_label is not null

union all

select
	date::timestamp as datetime,
	content,
	url,
	'Financial news' as source,
	sentiment,
	'financial news' as table
from it_final.financial_news;

--count rows
select count(*)
from it_final.media;

--check duplicate
SELECT
    count(*)
FROM
    (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY datetime,title,url,source, sentiment_label -- Liệt kê TẤT CẢ các cột để xác định trùng lặp
            ORDER BY ctid -- Sử dụng ctid hoặc cột khóa chính/thời gian nếu bạn muốn giữ lại một bản ghi cụ thể
        ) AS row_num
    FROM
        it_final.media
    ) AS subquery
WHERE
    row_num > 1;

--delete duplicate
WITH duplicate_rows AS (
    SELECT
        ctid, -- Cột ID vật lý của PostgreSQL, dùng để xác định dòng cần xóa
        ROW_NUMBER() OVER (
            PARTITION BY datetime, title, url, source, sentiment_label
            ORDER BY ctid
        ) AS row_num
    FROM
        it_final.media
)
DELETE FROM
    it_final.media
WHERE
    ctid IN (
        SELECT ctid
        FROM duplicate_rows
        WHERE row_num > 1 -- Xóa tất cả các bản sao (từ bản thứ 2 trở đi)
    );

-- chuấn hóa label
UPDATE it_final.media
SET sentiment_label = 'positive'
WHERE sentiment_label = 'Positve';

UPDATE it_final.media
SET sentiment_label = 'positive'
WHERE sentiment_label = 'Positive';


UPDATE it_final.media
SET sentiment_label = 'negative'
WHERE sentiment_label = 'Negative';

UPDATE it_final.media
SET sentiment_label = 'neutral'
WHERE sentiment_label = 'Neutral';

--bỏ 140k rows neutral
WITH del AS (
    SELECT ctid
    FROM it_final.media
    WHERE sentiment_label = 'neutral'
    LIMIT 140000
)
DELETE FROM it_final.media
WHERE ctid IN (SELECT ctid FROM del);

--bỏ 14k rows postive
WITH del AS (
    SELECT ctid
    FROM it_final.media
    WHERE sentiment_label = 'positive'
    LIMIT 24000
)
DELETE FROM it_final.media
WHERE ctid IN (SELECT ctid FROM del);

-- bỏ các dòng content quá dài
SELECT COUNT(*) AS overlong_count
FROM it_final.media
WHERE char_length(title) > 2000;

DELETE FROM it_final.media
WHERE char_length(title) > 2000;
