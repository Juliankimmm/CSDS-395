# get_contest_details_lambda.py
import json
import boto3
import os
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table_name = os.environ['CONTESTS_TABLE']
table = dynamodb.Table(table_name)

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def lambda_handler(event, context):
    try:
        contest_id = event['pathParameters']['contest_id']
        
        response = table.get_item(Key={'contest_id': contest_id})
        
        if 'Item' not in response:
            return {"statusCode": 404, "body": json.dumps({"error": "Contest not found"})}
            
        return {"statusCode": 200, "body": json.dumps(response['Item'], cls=DecimalEncoder)}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
