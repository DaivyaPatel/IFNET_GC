import psycopg2
import os

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/satellite"

print("Connecting to database...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='experiments' and column_name='user_id';
    """)
    if not cursor.fetchone():
        print("Adding user_id column...")
        cursor.execute("ALTER TABLE experiments ADD COLUMN user_id UUID REFERENCES users(id);")
        print("Column added successfully!")
    else:
        print("Column already exists.")
        
    conn.close()
except Exception as e:
    print("Database error:", e)
