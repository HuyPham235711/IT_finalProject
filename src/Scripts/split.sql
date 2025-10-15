

/*
 -------------------------------------------
 			MEDIA
 
 -------------------------------------------
 */
--104k
select count(*)
from it_final.media;

--train 52k - 7,8k
select count(*)
from it_final.media
where datetime < '2024-11-25';

create table it_final.media_train as
select *
from it_final.media
where datetime < '2024-11-25';

--valid 26k - 4k
select count(*)
from it_final.media
where datetime between '2025-01-01' and '2025-03-31';

create table it_final.media_valid as
select *
from it_final.media
where datetime between '2025-01-01' and '2025-03-31';

--test 26k - 4k
select count(*)
from it_final.media
where datetime between '2025-04-11' and '2025-09-10';

create table it_final.media_test as
select *
from it_final.media
where datetime between '2025-04-11' and '2025-09-10';

--backtest 17k
create table it_final.media_backtest as
	select *
	from it_final.media
	where datetime between '2024-11-26' and '2024-12-31'
	
	union all
	
	select *
	from it_final.media
	where datetime between '2025-04-01' and '2025-04-10'
	
	union all
	
	select *
	from it_final.media
	where datetime between '2025-09-11' and now();
	
--count
select count(*) from it_final.media_train;
select count(*) from it_final.media_test;
select count(*) from it_final.media_valid;
select count(*) from it_final.media_backtest;
/*
 -------------------------------------------
 			OHLCV
 
 -------------------------------------------
 */
	
--train
create table it_final.ohlcv_train as
select *
from it_final.raw_btcusd_1hr
where time_stamp between '2017-09-23' and '2024-11-25';

--valid
create table it_final.ohlcv_valid as
select *
from it_final.raw_btcusd_1hr
where time_stamp between '2025-01-01' and '2025-03-31';

--test
create table it_final.ohlcv_test as
select *
from it_final.raw_btcusd_1hr
where time_stamp between '2025-04-11' and '2025-09-10';

--backtest
create table it_final.ohlcv_backtest as
	select *
	from it_final.raw_btcusd_1hr
	where time_stamp between '2024-11-26' and '2024-12-31'
	
	union all
	
	select *
	from it_final.raw_btcusd_1hr
	where time_stamp between '2025-04-01' and '2025-04-10'
	
	union all
	
	select *
	from it_final.raw_btcusd_1hr
	where time_stamp between '2025-09-11' and '2025-10-02';
	
-- create processed tables
	
create table it_final.processed_ohlcv_train 
(
	datetime timestamp,
	open numeric,
	high numeric,
	low numeric,
	close numeric,
	volume numeric,
	sma14 numeric,
	rsi14 numeric
);

create table it_final.processed_ohlcv_test
(
	datetime timestamp,
	open numeric,
	high numeric,
	low numeric,
	close numeric,
	volume numeric,
	sma14 numeric,
	rsi14 numeric
);
create table it_final.processed_ohlcv_valid 
(
	datetime timestamp,
	open numeric,
	high numeric,
	low numeric,
	close numeric,
	volume numeric,
	sma14 numeric,
	rsi14 numeric
);

create table it_final.processed_ohlcv_backtest
(
	datetime timestamp,
	open numeric,
	high numeric,
	low numeric,
	close numeric,
	volume numeric,
	sma14 numeric,
	rsi14 numeric
);