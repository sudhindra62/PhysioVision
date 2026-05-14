import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Root%40123@localhost:5432/physiovision");

def get_db_connection():
    """Returns a raw psycopg2 connection that outputs dicts/JSON."""
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def init_db():
    """Run this once on startup to ensure the schema exists."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Using Raw SQL to bypass heavy ORMs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_stats (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            client_id VARCHAR(255) NOT NULL,
            therapist_id VARCHAR(255) NOT NULL,
            duration_seconds INT,
            avg_similarity FLOAT,
            most_corrected_joint VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_client_history ON session_stats(client_id);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database tables verified.")
