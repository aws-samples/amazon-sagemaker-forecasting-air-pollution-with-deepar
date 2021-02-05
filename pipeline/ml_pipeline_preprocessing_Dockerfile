FROM python:3.7-slim-buster
    
COPY ./sql /opt/ml/processing/sql

# boto3 & botocore package version specified for aiobotocore 1.1.1 compatible.
RUN pip install pandas numpy geopandas scikit-learn fsspec s3fs "boto3==1.14.44" "botocore==1.17.44"

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
