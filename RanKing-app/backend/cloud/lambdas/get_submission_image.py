import base64
import boto3
import json
import logging
import pymysql
import os

user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_proxy_host = os.environ['RDS_PROXY_HOST']
db_name = os.environ['DB_NAME']
# jwt_secret = os.environ['JWT_SECRET']


s3 = boto3.client('s3')

def lambda_handler(event, context):
    
    image = response['Body'].read()
    try:
        submission_id = event['pathParameters']['submission_id']
        
    except Exception as e:
        logger.log(e)
        return {"statusCode": 400, "body": json.dumps({"error": "Missing submission_id"})}

    response = s3.get_object(
        Bucket='s3-use2',
        Key=f'{submission_id}.jpg',
    )
    
    return {
        'headers': { "Content-Type": "image/jpg" },
        'statusCode': 200,
        'body': base64.b64encode(image).decode('utf-8'),
        'isBase64Encoded': True
    }
    return {
        'headers': { "Content-type": "text/html" },
        'statusCode': 200,
        'body': "<h1>This is text</h1>",
    }