#!/usr/bin/env python3
"""
Script to clear all data from the face attendance system database
"""

import sqlite3
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def clear_database():
    """Clear all data from the database tables"""
    
    # Database path (adjust if your database is in a different location)
    db_paths = [
        "smart_kiosk/data/attendance.db",
        "data/attendance.db",
        "attendance.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("Database file not found. Please check the path.")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Connected to database: {db_path}")
        
        # Disable foreign key checks temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # Clear all tables in the correct order (to avoid foreign key constraints)
        tables_to_clear = [
            "attendance_records",
            "face_encodings", 
            "employees",
            "departments"
        ]
        
        for table in tables_to_clear:
            try:
                cursor.execute(f"DELETE FROM {table}")
                rows_deleted = cursor.rowcount
                print(f"Cleared {rows_deleted} records from {table}")
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not clear table {table}: {e}")
        
        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence")
        print("Reset auto-increment counters")
        
        # Re-enable foreign key checks
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Commit changes
        conn.commit()
        print("\nDatabase cleared successfully!")
        
        # Vacuum database to reclaim space
        cursor.execute("VACUUM")
        print("Database vacuumed (optimized)")
        
        return True
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def clear_specific_table(table_name):
    """Clear data from a specific table"""
    
    # Find database
    db_paths = [
        "smart_kiosk/data/attendance.db",
        "data/attendance.db", 
        "attendance.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("Database file not found.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"DELETE FROM {table_name}")
        rows_deleted = cursor.rowcount
        
        conn.commit()
        print(f"Cleared {rows_deleted} records from {table_name}")
        return True
        
    except Exception as e:
        print(f"Error clearing table {table_name}: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear database tables")
    parser.add_argument("--table", help="Clear specific table only")
    parser.add_argument("--all", action="store_true", help="Clear all tables")
    
    args = parser.parse_args()
    
    if args.table:
        clear_specific_table(args.table)
    elif args.all:
        confirm = input("Are you sure you want to clear ALL data? (yes/no): ")
        if confirm.lower() == 'yes':
            clear_database()
        else:
            print("Operation cancelled")
    else:
        print("Use --all to clear all data or --table <table_name> to clear specific table")
        print("\nAvailable tables:")
        print("  - employees")
        print("  - departments") 
        print("  - face_encodings")
        print("  - attendance_records")