import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ecopackdb",
        user="postgres",
        password="radha123",
        port="5432"
    )

def init_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                product_name VARCHAR(255) NOT NULL,
                weight_kg DOUBLE PRECISION NOT NULL
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database connection warning/notice: {e}")