from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import snowflake.connector

load_dotenv()  # Loads variables from .env into os.environ

app = Flask(__name__)

@app.route('/query-consumer-zone', methods=['GET'])
def run_query():
    ctx = snowflake.connector.connect(
        user='YOUR_USERNAME',
        password='YOUR_PASSWORD',
        account='YOUR_ACCOUNT',
        warehouse='YOUR_WAREHOUSE',
        database='YOUR_DATABASE',
        schema='YOUR_SCHEMA'
    )

    cs = ctx.cursor()
    try:
        cs.execute("SELECT * FROM CONSUMER_ZONE_TABLE LIMIT 100")
        results = cs.fetchall()
        return jsonify(results)
    finally:
        cs.close()
        ctx.close()

if __name__ == '__main__':
    app.run(debug=True)
