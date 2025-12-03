# get_submissions_lambda.py
import sys
import logging
import pymysql
import json
import os

user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_proxy_host = os.environ['RDS_PROXY_HOST']
db_name = os.environ['DB_NAME']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    conn = pymysql.connect(host=rds_proxy_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit(1)

def lambda_handler(event, context):
    try:
        contest_id = event['pathParameters']['contest_id']
    except:
        return {"statusCode": 400, "body": "Missing contest_id"}

    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        sql = "SELECT * FROM Submissions WHERE contest_id=%s"
        cur.execute(sql, (contest_id,))
        submissions = cur.fetchall()

    return {"statusCode": 200, "body": json.dumps(submissions, default=str)}
