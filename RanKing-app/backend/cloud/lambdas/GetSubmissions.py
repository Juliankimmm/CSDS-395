import json
import boto3
from boto3.dynamodb.conditions import Key

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Submissions')

TARGET_BUCKET = "ranking-images-debug-01" # <--- YOUR BUCKET NAME

def lambda_handler(event, context):
    try:
        # Get contest_id from the URL path parameters
        # API Gateway will pass: /contests/{contest_id}/submissions
        params = event.get('pathParameters', {})
        contest_id = params.get('contest_id')

        if not contest_id:
            return {'statusCode': 400, 'body': 'Missing contest_id'}

        # 1. Query DynamoDB
        response = table.query(
            KeyConditionExpression=Key('contest_id').eq(int(contest_id))
        )
        items = response.get('Items', [])

        # 2. Transform Data & Generate URLs
        client_response = []
        
        for item in items:
            s3_key = item.get('image_s3_key')
            
            # GENERATE PRESIGNED URL
            # This creates a temporary public link to the private file
            image_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': TARGET_BUCKET, 'Key': s3_key},
                ExpiresIn=3600 # Link valid for 1 hour
            )
            
            # Map to your Swift Struct naming convention
            submission_obj = {
                'sub_id': item['sub_id'], # Keep as string or int depending on your DB
                'user_id': int(item['user_id']),
                'contest_id': int(item['contest_id']),
                'image_path': image_url, # <--- The actual workable URL
                'submitted_at': item['submitted_at']
            }
            client_response.append(submission_obj)

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(client_response)
        }

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}