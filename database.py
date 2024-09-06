import sqlite3

def initialize_database():
    # Connect to SQLite3 database
    con = sqlite3.connect('attendance.db')

    # Create user table with 'status' column defaulting to 'accept'
    con.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        userid INTEGER NOT NULL UNIQUE,
        status TEXT DEFAULT 'accept'
    )''')

    # Create attendance table
    con.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        userid INTEGER NOT NULL,
        timestamp TEXT NOT NULL
    )''')

    # Commit changes and close connection
    con.commit()
    con.close()

if __name__ == '__main__':
    initialize_database()