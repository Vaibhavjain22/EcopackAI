import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ecopackdb",
        user="postgres",
        password="radha123",
        port="5432"
    )