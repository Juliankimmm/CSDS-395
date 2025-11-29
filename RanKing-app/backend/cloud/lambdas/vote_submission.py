# vote_submission_lambda.py
import sys
import logging
import pymysql
import json
import os
# import jwt

user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_proxy_host = os.environ['RDS_PROXY_HOST']
db_name = os.environ['DB_NAME']
# jwt_secret = os.environ['JWT_SECRET']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    conn = pymysql.connect(host=rds_proxy_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit(1)

def lambda_handler(event, context):
    # token = event['headers'].get('Authorization')
    # if not token:
    #     return {"statusCode": 401, "body": json.dumps({"error": "Missing token"})}

    # try:
    #     payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
    #     user_id = payload['user_id']
    # except Exception:
    #     return {"statusCode": 401, "body": json.dumps({"error": "Invalid token"})}
    try:
        submission_id = event['pathParameters']['submission_id']
        
    except Exception as e:
        logger.log(e)
        return {"statusCode": 400, "body": json.dumps({"error": "Missing submission_id"})}
    user_id = event['body']['user_id']
    with conn.cursor() as cur:
        # prevent duplicate votes
        cur.execute("SELECT * FROM Votes WHERE user_id=%s AND submission_id=%s", (user_id, submission_id))
        if cur.fetchone():
            return {"statusCode": 400, "body": json.dumps({"error": "User has already voted"})}

        sql = "INSERT INTO Votes (user_id, submission_id) VALUES (%s, %s)"
        cur.execute(sql, (user_id, submission_id))
        conn.commit()

    return {"statusCode": 201, "body": json.dumps({"message": "Vote recorded"})}
