import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Securely locate the data folder to store the SQLite file
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "physiometrics.db"

def init_db():
    """Initializes the database schema exactly as planned."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS workout_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            exercise_name TEXT,
            target_joint TEXT,
            total_reps INTEGER,
            average_score REAL,
            duration_seconds INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_workout(exercise_name: str, target_joint: str, reps: int, avg_score: float, duration: int):
    """Inserts a completed session into the database. Prevents 0-rep saves."""
    if reps <= 0: 
        return False
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO workout_sessions 
        (timestamp, exercise_name, target_joint, total_reps, average_score, duration_seconds) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, exercise_name, target_joint, reps, avg_score, duration))
    
    conn.commit()
    conn.close()
    return True

def get_analytics_df() -> pd.DataFrame:
    """Extracts the entire workout ledger as a Pandas DataFrame for charting."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM workout_sessions ORDER BY timestamp ASC", conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df