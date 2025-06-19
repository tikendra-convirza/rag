from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uuid
import sqlite3
import os

app = FastAPI()
DB_DIR = "user_dbs"
os.makedirs(DB_DIR, exist_ok=True)

class IngestRequest(BaseModel):
    data: str

class QueryRequest(BaseModel):
    user_id: str
    query: str

def get_db_path(user_id: str) -> str:
    return os.path.join(DB_DIR, f"{user_id}.db")

def init_user_db(user_id: str):
    db_path = get_db_path(user_id)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, data TEXT)")
    conn.commit()
    conn.close()

@app.post("/ingest")
def ingest(req: IngestRequest):
    user_id = str(uuid.uuid4())
    init_user_db(user_id)
    db_path = get_db_path(user_id)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO records (data) VALUES (?)", (req.data,))
    conn.commit()
    conn.close()
    return {"user_id": user_id}

@app.post("/query")
def query(req: QueryRequest):
    db_path = get_db_path(req.user_id)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="User DB not found")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # For demo, just search for data containing the query string
    c.execute("SELECT data FROM records WHERE data LIKE ?", (f"%{req.query}%",))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return {"results": results}

def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
