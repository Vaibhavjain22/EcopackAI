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
                weight_kg DOUBLE PRECISION NOT NULL,
                category VARCHAR(50) DEFAULT 'general'
            );
        """)
        # Migration: add category column if table already exists without it
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'products' AND column_name = 'category'
                ) THEN
                    ALTER TABLE products ADD COLUMN category VARCHAR(50) DEFAULT 'general';
                END IF;
            END $$;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database connection warning/notice: {e}")
