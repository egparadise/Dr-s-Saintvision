"""Check existing database contents"""
import sys
sys.path.insert(0, r"C:\Project\Dr's Saintvision\Lib\site-packages")

import sqlite3
from pathlib import Path

DATABASE_PATH = Path("./database/history.db")

def check_database():
    print("=" * 60)
    print("  DATABASE CHECK")
    print("=" * 60)

    if not DATABASE_PATH.exists():
        print("Database file not found!")
        return

    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nTables: {[t[0] for t in tables]}")

    # Check conversations
    print("\n--- Conversations ---")
    try:
        cursor.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC LIMIT 5")
        convs = cursor.fetchall()
        if convs:
            for c in convs:
                print(f"  ID {c[0]}: {c[1][:40]}... ({c[2]})")
        else:
            print("  No conversations found")
    except Exception as e:
        print(f"  Error: {e}")

    # Check messages
    print("\n--- Messages ---")
    try:
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        print(f"  Total messages: {count}")

        cursor.execute("SELECT conversation_id, role, content FROM messages ORDER BY id DESC LIMIT 3")
        msgs = cursor.fetchall()
        for m in msgs:
            print(f"  Conv {m[0]} [{m[1]}]: {m[2][:50]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # Check learning data
    print("\n--- Learning Data ---")
    try:
        cursor.execute("SELECT COUNT(*) FROM learning_data")
        count = cursor.fetchone()[0]
        print(f"  Total learning entries: {count}")

        cursor.execute("SELECT id, query, similarity_score, learned FROM learning_data ORDER BY id DESC LIMIT 5")
        data = cursor.fetchall()
        for d in data:
            status = "Learned" if d[3] else "Pending"
            sim = f"{d[2]*100:.1f}%" if d[2] else "N/A"
            print(f"  ID {d[0]}: {d[1][:40]}... (Sim: {sim}, {status})")
    except Exception as e:
        print(f"  Error: {e}")

    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_database()
