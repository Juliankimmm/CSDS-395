import json
import base64
import boto3
import time
import uuid
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Submissions')

TARGET_BUCKET = "ranking-images-debug-01" # <--- YOUR BUCKET NAME

def lambda_handler(event, context):
    try:
        # 1. Parse Input
        body = event.get('body', {})
        if event.get('isBase64Encoded', False):
            body = base64.b64decode(body).decode('utf-8')
        if isinstance(body, str):
            body = json.loads(body)

        # 2. Extract Data
        # We generate a unique ID for the submission
        sub_id = str(uuid.uuid4()) 
        contest_id = body.get('contest_id', 1) # Default to 1 if missing
        user_id = body.get('user_id', 0)
        
        filename_raw = body.get('filename', 'image.jpg')
        # We rename the file in S3 to ensure it's unique and organized
        s3_key = f"contest_{contest_id}/{sub_id}.jpg"
        
        image_data = body.get('image', '')
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # 3. Upload Image to S3
        image_bytes = base64.b64decode(image_data)
        s3_client.put_object(
            Bucket=TARGET_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType='image/jpeg'
        )

        # 4. Save Metadata to DynamoDB
        # We store the "Key", not the full URL, to keep it clean.
        timestamp = str(int(time.time()))
        
        item = {
            'contest_id': int(contest_id),
            'sub_id': sub_id,
            'user_id': int(user_id),
            'image_s3_key': s3_key, # Storing the key, we will convert to URL on GET
            'submitted_at': timestamp
        }
        
        table.put_item(Item=item)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Success', 'sub_id': sub_id})
        }

    except Exception as e:
        logger.error(str(e))
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}