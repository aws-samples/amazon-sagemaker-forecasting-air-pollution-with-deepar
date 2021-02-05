CREATE EXTERNAL TABLE IF NOT EXISTS openaq(
  `date` struct<utc:string,local:string>, 
  `parameter` string, 
  `location` string, 
  value float, 
  unit string, 
  city string, 
  attribution array<struct<name:string,url:string>>, 
  averagingperiod struct<unit:string,value:float>, 
  coordinates struct<latitude:float,longitude:float>, 
  country string, 
  sourcename string, 
  sourcetype string, 
  mobile string
)
ROW FORMAT SERDE  'org.openx.data.jsonserde.JsonSerDe' 
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://openaq-fetches/realtime-gzipped'