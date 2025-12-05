import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
votes_table = dynamodb.Table(os.environ['VOTES_TABLE'])
sub_table = dynamodb.Table(os.environ['SUBMISSIONS_TABLE'])

def lambda_handler(event, context):
    try:
        submission_id = event['pathParameters']['submission_id']
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        user_id = str(body['user_id']) 
        
        # 1. Record the Vote in Votes Table
        # PK: submission_id, SK: user_id
        # This prevents the SAME user voting on the SAME submission twice.
        try:
            votes_table.put_item(
                Item={
                    'submission_id': submission_id,
                    'user_id': user_id
                },
                ConditionExpression='attribute_not_exists(submission_id)'
            )
        except boto3.client('dynamodb').exceptions.ConditionalCheckFailedException:
             return {"statusCode": 400, "body": json.dumps({"error": "User has already voted"})}

        # 2. Increment vote count on Submissions Table
        # FIX: Key is ONLY submission_id. 
        # The user_id is the VOTER, not part of the submission's identity.
        sub_table.update_item(
            Key={'submission_id': submission_id}, 
            UpdateExpression="SET vote_count = vote_count + :inc",
            ExpressionAttributeValues={':inc': 1}
        )

        return {"statusCode": 201, "body": json.dumps({"message": "Vote recorded"})}
        
    except Exception as e:
        logger.error(e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}