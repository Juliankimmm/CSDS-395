# get_contests_lambda.py
import sys
import logging
import pymysql
import json
import os
import datetime

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
    query = event.get('queryStringParameters') or {}
    status = query.get('status')
    now = datetime.datetime.utcnow()

    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        if status == "submission":
            sql = "SELECT * FROM Contests WHERE submission_start_date <= %s AND submission_end_date >= %s"
            cur.execute(sql, (now, now))
        elif status == "voting":
            sql = "SELECT * FROM Contests WHERE submission_end_date <= %s AND voting_end_date >= %s"
            cur.execute(sql, (now, now))
        elif status == "finished":
            sql = "SELECT * FROM Contests WHERE voting_end_date < %s"
            cur.execute(sql, (now,))
        else:
            sql = "SELECT * FROM Contests"
            cur.execute(sql)

        contests = cur.fetchall()

    return {"statusCode": 200, "body": json.dumps(contests, default=str)}
