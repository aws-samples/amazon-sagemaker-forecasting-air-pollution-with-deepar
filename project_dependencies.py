import boto3, time, s3fs, json, warnings, os
import urllib.request
from datetime import date, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.model import Model
from multiprocessing import Pool

from IPython.display import display, HTML, Javascript

from bokeh.plotting import curdoc, figure, output_notebook, show
from bokeh.layouts import column, row, layout
from bokeh.models import GeoJSONDataSource, ColumnDataSource
from bokeh.models import Select, Slider
from bokeh.models.callbacks import CustomJS
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.palettes import inferno

# the train test split date is used to split each time series into train and test sets
train_test_split_date = date.today() - timedelta(days=30)#).strftime('%y-%m-%d 00:00:00')

# the sampling frequency determines the number of hours per sample
# and is used for aggregating and filling missing values
frequency = '1' # hours

# prediction length is how many hours into future to predict values for
prediction_length = 48

# context length is how many prior time steps the predictor needs to make a prediction
context_length = 3

# the file to save predictions to
predictions_file = 'data/predictions.pkl'

# quantiles that will be predicted
quantiles = list(range(1,10))
quantile_names = [f'0.{q}' for q in quantiles]

warnings.filterwarnings('ignore')

session = boto3.Session()
region = session.region_name
account_id = session.client('sts').get_caller_identity().get('Account')
bucket_name = f"{account_id}-openaq-lab"
console_s3_uri= 'https://s3.console.aws.amazon.com/s3/object/'

s3 = boto3.client('s3', region_name = region)
os.makedirs('model', exist_ok=True)
urllib.request.urlretrieve('https://d8pl0xx4oqh22.cloudfront.net/model.tar.gz', 'model/model.tar.gz')
try:
    if 'us-east-1' == region:
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
except:
    pass
s3.upload_file('model/model.tar.gz', bucket_name, 'sagemaker/model/model.tar.gz')

def athena_create_table(query_file, wait=None):
    create_table_uri = athena_execute(query_file, 'txt', wait)
    return create_table_uri
     
def athena_query_table(query_file, wait=None):
    results_uri = athena_execute(query_file, 'csv', wait)
    return results_uri

def athena_execute(query_file, ext, wait):
    with open(query_file) as f:
        query_str = f.read()  
        
    display(HTML(f'Executing query:<br><br><code>{query_str}</code><br><br>'))
        
    athena = boto3.client('athena')
    s3_dest = f's3://{bucket_name}/athena/results/'
    query_id = athena.start_query_execution(
        QueryString= query_str, 
         ResultConfiguration={'OutputLocation': s3_dest}
    )['QueryExecutionId']
        
    results_uri = f'{s3_dest}{query_id}.{ext}'
        
    start = time.time()
    while wait == None or wait == 0 or time.time() - start < wait:
        result = athena.get_query_execution(QueryExecutionId=query_id)
        status = result['QueryExecution']['Status']['State']
        if wait == 0 or status == 'SUCCEEDED':
            break
        elif status in ['QUEUED','RUNNING']:
            continue
        else:
            raise Exception(f'query {query_id} failed with status {status}')

            time.sleep(3) 

    console_url = f'{console_s3_uri}{bucket_name}/athena/results/{query_id}.{ext}?region={region}&tab=overview'
    display(HTML(f'results are located at <a target="_blank" href="{console_url}">{results_uri}</a>'))
    
    return results_uri

def display_hpo_tuner_advice(hpo_tuner):
    display(HTML(f'''<br>The hyperparameter tuning job "{hpo_tuner.latest_tuning_job.name}" is now running. 
            To view it in the console click 
            <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs">here</a>.
        '''))

def display_training_job_advice(training_job):
    display(HTML(f'''<br>The training job "{training_job.name}" is now running. 
        To view it in the console click 
        <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs">here</a>.
    '''))  

def display_endpoint_advice(session, endpoint_name, wait=False):
    display(HTML(f'''<br>The end point "{endpoint_name}" is now being deployed. 
        To view it in the console click 
        <a target="_blank" href="https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints">here</a>.
    ''')) 
    if wait:
        session.wait_for_endpoint(endpoint_name)
        display(HTML(f'The end point "{endpoint_name}" is now deployed.'))
        
def filter_dates(df, min_time, max_time, frequency):
    min_time = None if min_time is None else pd.to_datetime(min_time)
    max_time = None if max_time is None else pd.to_datetime(max_time)
    interval = pd.Timedelta(frequency)
    
    def _filter_dates(r): 
        if min_time is not None and r['start'] < min_time:
            start_idx = int((min_time - r['start']) / interval)
            r['target'] = r['target'][start_idx:]
            r['start'] = min_time
        
        end_time = r['start'] + len(r['target']) * interval
        if max_time is not None and end_time > max_time:
            end_idx = int((end_time - max_time) / interval)
            r['target'] = r['target'][:-end_idx]
            
        return r
    
    filtered = df.apply(_filter_dates, axis=1) 
    filtered = filtered[filtered['target'].str.len() > 0]
    return filtered

def get_tests(features, split_dates, frequency, context_length, prediction_length):
    tests = []
    end_date_delta = pd.Timedelta(f'{frequency} hour') * context_length
    prediction_id = 0
    for split_date in split_dates:
        context_end = split_date + end_date_delta
        test = filter_dates(features, split_date, context_end, f'{frequency}H')
        test['prediction_start'] = context_end
        test['prediction_id'] = prediction_id
        test['start'] = test['start'].dt.strftime('%Y-%m-%d %H:%M:%S')
        tests.append(test)
        prediction_id += 1
        
    tests = pd.concat(tests).reset_index().set_index(['id', 'prediction_id', 'prediction_start']).sort_index()
    return tests

def init_predict_process(func, endpoint_name, quantiles):
    func.predictor = Predictor(
        endpoint_name, 
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )
    func.quantiles = quantiles
    
def call_endpoint(feature):
    request = dict(
        instances= [feature],
        configuration= dict(
            num_samples= 20,
            output_types= ['quantiles'],
            quantiles= call_endpoint.quantiles
        )   
    )
    
    response = call_endpoint.predictor.predict(request)  
    raw_quantiles = response['predictions'][0]['quantiles']
    return {q: [[np.around(v, 2) for v in l]] for q,l in raw_quantiles.items()}      

def predict(endpoint_name, samples, quantiles, processes=10):
    features = samples[['start', 'target', 'cat']].to_dict(orient='records')
    quantile_strs = [f'0.{q}' for q in quantiles]
    with Pool(processes, init_predict_process, [call_endpoint, endpoint_name, quantile_strs]) as pool:
        inferences = pool.map(call_endpoint, features)
      
    df = pd.concat([pd.DataFrame(inference) for inference in inferences], ignore_index=True)                          
    df = df[sorted(df.columns.values)]
    df.set_index(samples.index, inplace=True)
    df.index.names = ['id', 'prediction_id', 'start']
    df.reset_index(level=2, inplace=True)
    return df

def plot_error(actuals, predictions, horizon=None): 
    def error_for_prediction(predictions):
        location_id = predictions.index[0][0]
        actual = actuals.loc[location_id]
        prediction = predictions.iloc[0]
        offset = int((prediction.start - actual.start)/pd.Timedelta('1 hour'))
        
        def error(F):
            if horizon and horizon < len(F):
                offset_end = offset + horizon
            else:
                offset_end = offset + len(F)
            A = np.array(actual.target[offset: offset_end])
            A_mean = A.mean()
            F = np.array(F[:len(A)])
            error = np.abs((A - F)/A_mean)
            return error
        
        return prediction[quantile_names].apply(error)
            
    MAPE = predictions.groupby(level=['id', 'prediction_id']).apply(error_for_prediction)
    max_length = MAPE['0.1'].apply(len).max()
    MAPE = MAPE[MAPE['0.1'].map(len) == max_length]
    return MAPE.mean().plot()

def moving_average(values, averaging_period=24, roundby=2):
    return pd.Series(values)\
            .rolling(averaging_period, min_periods=1)\
            .mean()\
            .round(roundby)\
            .to_numpy()

def add_data_to_indexdb(actuals, predictions):
    with open('javascript/create_table.js', 'r') as f:
        create_table_script = f.read()
        display(Javascript(create_table_script))
        
    with open('javascript/index_data.js', 'r') as f:
        index_script = f.read()
        
    def exec_js(actual_strs, prediction_strs):
        actual_str = '['+','.join(actual_strs)+']'
        prediction_str = '['+','.join(prediction_strs)+']'
        display(Javascript(f"""
            {index_script}
            index_data('openaq','actuals', {actual_str});
            index_data('openaq','predictions', {prediction_str});
        """))
             
    actual_strs = []
    prediction_strs = []
    last_actual_lid = None
    for index, row in predictions.reset_index().iterrows():
        lid = row['id']
        
        if last_actual_lid != lid:
            actual_row = actuals.loc[lid]
            start = actual_row['start'].strftime('%Y-%m-%dT%H:%M:%SZ')
            target_str = json.dumps(actual_row['target'])
            ma_str = json.dumps(actual_row['ma'].tolist())
            actual_strs.append('{' + f'id:"{lid}",start:"{start}",target:{target_str}, ma:{ma_str}' + '}')
            last_actual_lid = lid
        
        pid = row['prediction_id']
        start = row['start'].strftime('%Y-%m-%dT%H:%M:%SZ')
        quantile_results = []
        for q in range(1,10):
            quantile = f'0.{q}'  
            if quantile in row:
                values = json.dumps(row[quantile])
                quantile_results.append(f'"{quantile}":{values}')
        
        prediction_str = ','.join(quantile_results)
        prediction_strs.append('{' + f'id:"{lid}:{pid}",start:"{start}",{prediction_str}' + '}')
        
        if len(prediction_strs) == 100:
            exec_js(actual_strs, prediction_strs)
            actual_strs = []
            prediction_strs = []
            
    if len(actual_strs):
        exec_js(actual_strs, prediction_strs)

def create_analysis_chart(metadata, actuals, predictions):
    add_data_to_indexdb(actuals, predictions)
    
    ######################
    # create actuals plot
    actuals_source = ColumnDataSource(dict(id=[], start=[], target=[], ma=[])) # empty

    filtered_predictions = predictions.reset_index(level=1)
    predictions_source = ColumnDataSource(dict(id=[], start=[], **{q:[] for q in quantile_names}))

    # create the plot
    predictions_plot = figure(
        title='', 
        plot_width=800, plot_height=400, 
        x_axis_label='date/time', 
        y_axis_label='pm10 ',
        x_axis_type='datetime',
        y_range= [0, max(predictions['0.5'].max())],
        tools=''
    )

    # plot vertical areas for the quantiles
    predictions_plot.varea_stack(
        stackers=quantile_names, 
        x='start', 
        color= inferno(len(quantiles)), 
        legend_label=quantile_names, 
        source=predictions_source,
        alpha=1,
    )

    # plot actual values
    predictions_plot.line(
        x= "start", y= "target", 
        color= 'red', 
        source= actuals_source
    )
    
    # plot actual values
    predictions_plot.line(
        x= "start", y= "ma", 
        color= 'black', 
        source= actuals_source
    )

    # add a legend
    predictions_plot.legend.items.reverse()
    
    
    #############################
    # Create location selector
    options = metadata.reset_index()[['id', 'country', 'city', 'location']].astype('str').agg('-'.join, axis=1).tolist()
    location_select = Select(title='Select Location:', value=options[0], options=options)
    
    
    ################################
    # Create prediction start slider
    start_min = filtered_predictions.reset_index()['start'].min()
    start_slider = Slider(
        start=0, 
        end= predictions.index.get_level_values(1).unique().max(), 
        value=0, 
        step=1, 
        title=f'prediction time delta'
    )
    
    
    #############################
    # Create javascript callback
    # The javascript callback function connects all the plots and 
    # gui components together so changes will update the plots.
    callback_args=dict(
        actuals= actuals_source, 
        predictions= predictions_source, 
        location_select= location_select, 
        start_slider= start_slider
    )

    with open('javascript/plot_update_callback.js', 'r') as f:
        callback_code = f.read()    
        plot_update_callback = CustomJS(code=callback_code, args=callback_args)
        location_select.js_on_change('value', plot_update_callback)
        start_slider.js_on_change('value', plot_update_callback)
                                                                   
    return column(location_select, predictions_plot, start_slider) 