# login_lambda.py
import sys
import logging
import pymysql
import json
import os
# import jwt
import datetime

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
    raw_body = event['body']
    if isinstance(raw_body, str):
        body = json.loads(raw_body)
    else:
        body = raw_body

    email = body['email']
    password_hash = body['password_hash']

    with conn.cursor() as cur:
        sql = "SELECT user_id, username FROM Users WHERE email=%s AND password_hash=%s"
        cur.execute(sql, (email, password_hash))
        user = cur.fetchone()
        if not user:
            return {"statusCode": 401, "body": json.dumps({"error": "Invalid credentials"})}

        payload = {"user_id": user[0], "username": user[1], "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)}
        # token = jwt.encode(payload, jwt_secret, algorithm="HS256")

    return {"statusCode": 200, "body": json.dumps({"token": "test_token"})}
