import json
import boto3
import os
import uuid
import base64
import time
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

sub_table = dynamodb.Table(os.environ['SUBMISSIONS_TABLE'])
bucket_name = os.environ['BUCKET_NAME']

def lambda_handler(event, context):
    try:
        contest_id = event['pathParameters']['contest_id']
        
        # Parse Body
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        
        user_id = body.get('user_id') # In real app, get from Token
        image_b64 = body.get('image') # Expecting base64 string
        
        if not image_b64 or not user_id:
            return {"statusCode": 400, "body": "Missing image or user_id"}
            
        # Clean base64 string
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
            
        image_bytes = base64.b64decode(image_b64)
        
        # Generate IDs
        submission_id = str(uuid.uuid4())
        filename = f"{submission_id}.jpg" # Simple filename
        
        # 1. Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=image_bytes,
            ContentType='image/jpeg'
        )
        
        # 2. Save metadata to DynamoDB
        item = {
            'submission_id': submission_id,
            'contest_id': contest_id,
            'user_id': user_id,
            's3_key': filename,
            'vote_count': 0,
            'submitted_at': datetime.utcnow().isoformat()
        }
        
        sub_table.put_item(Item=item)
        
        return {
            "statusCode": 201, 
            "body": json.dumps({"message": "Submission received", "submission_id": submission_id})
        }
        
    except Exception as e:
        print(e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}