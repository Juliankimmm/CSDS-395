import sys
import logging
import pymysql
import json
import os

# rds settings
user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_proxy_host = os.environ['RDS_PROXY_HOST']
db_name = os.environ['DB_NAME']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create the database connection outside of the handler to allow connections to be
# re-used by subsequent function invocations.
try:
        conn = pymysql.connect(host=rds_proxy_host, user=user_name, passwd=password, db=db_name, connect_timeout=5)
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)
    sys.exit(1)

logger.info("SUCCESS: Connection to RDS for MySQL instance succeeded")

def handler(event, context):
    """
    This function creates a new user
    """
    
    raw_body = event['body']
    if isinstance(raw_body, str):
        body = json.loads(raw_body)
    else:
        body = raw_body
    
    username = body['username']
    email = body['email']
    password_hash = body['password_hash']

    item_count = 0
    sql_string = "insert into Users (username, email, password_hash) values(%s, %s, %s)"

    with conn.cursor() as cur:
        cur.execute(sql_string, (username, email, password_hash))
        conn.commit()
        cur.execute("select * from Users")
        logger.info("The following items have been added to the database:")
        for row in cur:
            item_count += 1
            logger.info(row)
    conn.commit()

    return "Added %d items to RDS for MySQL table" %(item_count)
    
