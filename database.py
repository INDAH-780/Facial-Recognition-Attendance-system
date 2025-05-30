import psycopg2

conn = psycopg2.connect(
    dbname="attendance_db",
    user="postgres",
    password="postgress123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()
