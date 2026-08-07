import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    try:
        conn.execute(text('ALTER TABLE output_comparisons ADD COLUMN lpips FLOAT'))
        print('Added to output_comparisons')
    except Exception as e:
        print('output_comparisons:', e)
        
    try:
        conn.execute(text('ALTER TABLE input_comparisons ADD COLUMN lpips FLOAT'))
        print('Added to input_comparisons')
    except Exception as e:
        print('input_comparisons:', e)
        
    conn.commit()

print('Done!')
