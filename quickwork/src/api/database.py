import mysql.connector
from mysql.connector import Error

def get_database_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="vikram12345",
        database="market_dashboard"
    )

# Get the database connection
db = get_database_connection()
