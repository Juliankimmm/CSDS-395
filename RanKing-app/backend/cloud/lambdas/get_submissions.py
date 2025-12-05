# get_submissions_lambda.py
import json
import boto3
import os
from decimal import Decimal
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
sub_table = dynamodb.Table(os.environ['SUBMISSIONS_TABLE'])

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj)
        return super(DecimalEncoder, self).default(obj)

def lambda_handler(event, context):
    try:
        contest_id = event['pathParameters']['contest_id']
        
        # Query using the GSI (ContestIndex)
        response = sub_table.query(
            IndexName='ContestIndex',
            KeyConditionExpression=Key('contest_id').eq(contest_id)
        )
        
        items = response.get('Items', [])
        
        return {"statusCode": 200, "body": json.dumps(items, cls=DecimalEncoder)}
        
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
