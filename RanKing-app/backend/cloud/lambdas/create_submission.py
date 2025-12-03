# create_submission_lambda.py
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

    raw_body = event['body']
    if isinstance(raw_body, str):
        body = json.loads(raw_body)
    else:
        body = raw_body

    image_data = body.get('image')  # expecting base64 string
    contest_id = event['pathParameters']['contest_id']

    with conn.cursor() as cur:
        sql = "INSERT INTO Submissions (user_id, contest_id, image_data) VALUES (%s, %s, %s)"
        cur.execute(sql, (user_id, contest_id, image_data))
        conn.commit()

    return {"statusCode": 201, "body": json.dumps({"message": "Submission created"})}
