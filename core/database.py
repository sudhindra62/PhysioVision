"""
Database & Persistence Layer Module

Manages relational data storage and historical tracking ledger profiles using SQLite.
Handles transactional data transformations, relational schema execution, and bridges 
the persistence layer with frontend Pandas charting components.
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Securely locate the data folder to store the SQLite file
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "physiometrics.db"

def init_db():
    """
    Initializes the local relational database schema.

    Creates two tables if they do not already exist:
    1. 'workout_sessions': Stores macroscopic session-level metrics (averages, reps, metadata).
    2. 'rep_details': A child table linked via a foreign key relationship to record 
       the precise performance score of each isolated repetition.
    """
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
    
    # Create relational table for granular rep data
    c.execute('''
        CREATE TABLE IF NOT EXISTS rep_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            rep_number INTEGER,
            rep_score REAL,
            FOREIGN KEY(session_id) REFERENCES workout_sessions(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_workout(exercise_name: str, target_joint: str, reps: int, avg_score: float, duration: int, rep_scores: list = None) -> bool:
    """
    Inserts a completed exercise session and its granular rep logs transactionally.

    Prevents database pollution by enforcing a structural safeguard that rejects 
    empty or unexecuted sessions (reps <= 0). Uses atomic operations via parameters 
    to prevent SQL injection vectors.

    Args:
        exercise_name (str): The display identity of the exercise executed.
        target_joint (str): Text string detailing the primary muscle mover joint tracked.
        reps (int): Total count of valid, locked repetitions completed.
        avg_score (float): Mean geometric form accuracy across the entire session.
        duration (int): Runtime duration of the active workout block in seconds.
        rep_scores (list, optional): Sequential list of float scores tracking individual reps.

    Returns:
        bool: True if transaction committed successfully, False if session was rejected.
    """
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
    
    # Get the ID of the parent session row we just inserted
    session_id = c.lastrowid 
    
    # Iterate and save individual rep scores linked via the session ID
    if rep_scores:
        for idx, score in enumerate(rep_scores):
            c.execute('''
                INSERT INTO rep_details (session_id, rep_number, rep_score) 
                VALUES (?, ?, ?)
            ''', (session_id, idx + 1, score))
    
    conn.commit()
    conn.close()
    return True

def get_analytics_df() -> pd.DataFrame:
    """
    Extracts the global workout session ledger sorted chronologically.

    Executes a direct raw query on the parent dataset, loads the relational matrix 
    directly into memory as a Pandas DataFrame, and parses the text-based column 
    values into specialized DateTime objects for downstream time-series plotting.

    Returns:
        pd.DataFrame: A structured data frame mapping historical exercise records.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM workout_sessions ORDER BY timestamp ASC", conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df