import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table_name = os.environ['USERS_TABLE']
table = dynamodb.Table(table_name)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        
        email = body['email']
        password_hash = body['password_hash']
        
        # Fetch user by Email
        response = table.get_item(Key={'email': email})
        
        if 'Item' not in response:
             return {"statusCode": 401, "body": json.dumps({"error": "Invalid credentials"})}
             
        user = response['Item']
        
        # Check password
        if user['password_hash'] != password_hash:
            return {"statusCode": 401, "body": json.dumps({"error": "Invalid credentials"})}
            
        # Return success (Simulated token/user_id)
        return {"statusCode": 200, "body": json.dumps({"user_id": user['user_id']})}
        
    except Exception as e:
        logger.error(e)
        return {'statusCode': 500, 'body': json.dumps({"error": str(e)})}

