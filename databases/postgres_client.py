from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
from .base import VectorDB


class PostgresDB(VectorDB):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "movies",
        user: str = "postgres",
        password: str = "",
        table: str = "movies"
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.conn = None
        self.dim = None

    def setup(self, dim: int) -> None:
        self.dim = dim
        
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            
            cur.execute(f"""
                CREATE TABLE {self.table} (
                    id SERIAL PRIMARY KEY,
                    movie_id INTEGER,
                    title TEXT,
                    genres TEXT,
                    year INTEGER,
                    embedding vector({dim})
                )
            """)
            
            cur.execute(f"""
                CREATE INDEX ON {self.table} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
        self.conn.commit()

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        if not vectors or not payloads:
            return
        
        with self.conn.cursor() as cur:
            data = []
            for vec, payload in zip(vectors, payloads):
                data.append((
                    payload.get('movie_id'),
                    payload.get('title'),
                    payload.get('genres'),
                    payload.get('year'),
                    vec
                ))
            
            execute_values(
                cur,
                f"""
                INSERT INTO {self.table} (movie_id, title, genres, year, embedding)
                VALUES %s
                """,
                data,
                template="(%s, %s, %s, %s, %s)",
                page_size=1000
            )
        
        self.conn.commit()

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, movie_id, title, genres, year, 
                       1 - (embedding <=> %s::vector) as score
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query, query, top_k)
            )
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'movie_id': row[1],
                    'title': row[2],
                    'genres': row[3],
                    'year': row[4],
                    'score': float(row[5])
                })
            
            return results

    def teardown(self) -> None:
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
