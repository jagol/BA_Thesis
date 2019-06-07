import sqlite3

"""
Table format: 
term_id, doc_id, embedding as string
"""

def create_elmo_db(path_db: str) -> None:
    conn = sqlite3.connect(path_db)

