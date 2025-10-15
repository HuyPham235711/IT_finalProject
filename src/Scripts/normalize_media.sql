-- table dataset
alter table it_final_raw.dataset add column datetime_parsed timestamp;
alter table it_final_raw.dataset add column sentiment_label text;

update it_final_raw.dataset
set datetime_parsed = to_timestamp(datetime, 'yyyy-mm-dd');

update it_final_raw.dataset 
set sentiment_label = 
	case
		when label = 1 then 'Negative'
		when label = 0 then 'Positive'
	end;

-- table cryptonews
alter table it_final_raw.cryptonews add column datetime timestamp;
alter table it_final_raw.cryptonews add column sentiment_label text;

update it_final_raw.cryptonews 
set datetime = to_timestamp(date, 'YYYY-MM-DD HH24:MI:SS');

update it_final_raw.cryptonews
set sentiment_label = (REPLACE(sentiment, '''', '"')::jsonb) ->> 'class';



--table cryptopacnic_news;
alter table it_final_raw.cryptopanic_news add column sentiment_label text;
alter table it_final_raw.cryptopanic_news add column source text;

update it_final_raw.cryptopanic_news 
set sentiment_label = 
case 
	when negative < positive then 'Positve'
	when negative > positive then 'Negative'
	when negative = positive then 'Neutral'
end;

update it_final_raw.cryptopanic_news 
set source = ds.name
from it_final_raw."source" ds
where ds.id::text = "sourceId";



