import json
import boto3
import os
import logging
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table_name = os.environ['CONTESTS_TABLE']
table = dynamodb.Table(table_name)

# Helper to fix JSON serialization of Decimals
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def lambda_handler(event, context):
    query = event.get('queryStringParameters') or {}
    status = query.get('status')
    
    # Scan all contests (acceptable for small datasets)
    response = table.scan()
    contests = response.get('Items', [])
    
    now = datetime.utcnow().isoformat()
    
    filtered_contests = []
    
    # Filter logic in Python
    for c in contests:
        # Ensure dates are strings in ISO format in DB: "2023-12-01T12:00:00"
        sub_start = c.get('submission_start_date')
        sub_end = c.get('submission_end_date')
        vote_end = c.get('voting_end_date')
        
        if status == "submission":
            if sub_start <= now and sub_end >= now:
                filtered_contests.append(c)
        elif status == "voting":
            if sub_end <= now and vote_end >= now:
                filtered_contests.append(c)
        elif status == "finished":
            if vote_end < now:
                filtered_contests.append(c)
        else:
            filtered_contests.append(c)

    return {
        "statusCode": 200, 
        "body": json.dumps(filtered_contests, cls=DecimalEncoder)
    }