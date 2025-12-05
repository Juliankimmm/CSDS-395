import json
import boto3
import os
import uuid
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table_name = os.environ['USERS_TABLE']
table = dynamodb.Table(table_name)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        
        username = body['username']
        email = body['email']
        password_hash = body['password_hash']
        
        # Check if user exists (by email)
        response = table.get_item(Key={'email': email})
        if 'Item' in response:
            return {'statusCode': 400, 'body': json.dumps({"error": "Email already exists"})}
            
        # Create User
        user_id = str(uuid.uuid4())
        
        item = {
            'email': email,
            'user_id': user_id,
            'username': username,
            'password_hash': password_hash
        }
        
        table.put_item(Item=item)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                "status": "Successfully Added User",
                "user_id": user_id
            })
        }
    except Exception as e:
        logger.error(e)
        return {'statusCode': 500, 'body': json.dumps({"error": str(e)})}