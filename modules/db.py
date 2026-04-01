"""
Database connection management for SQLite
Handles initialization, connection pooling, and lifecycle
"""

import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_PATH = "data/intelligent_pdf_search.db"
SCHEMA_VERSION = 1


class Database:
    """SQLite database manager with connection pooling"""
    
    _instance: Optional['Database'] = None
    
    def __new__(cls):
        """Singleton pattern - only one database instance"""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = DATABASE_PATH
        self.connection: Optional[sqlite3.Connection] = None
        logger.info(f"Initializing database: {self.db_path}")
        self.initialize()
        self._initialized = True
    
    def initialize(self) -> None:
        """Initialize database connection and create schema if needed"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.connection = self._create_connection()
            if self.connection:
                self._create_schema()
                logger.info("Database initialized successfully")
            else:
                raise Exception("Failed to create database connection")
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
            raise
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Journal mode for better concurrent access
        conn.execute("PRAGMA journal_mode = WAL")
        
        logger.debug(f"Database connection established: {self.db_path}")
        return conn
    
    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist"""
        cursor = self.connection.cursor()
        
        # Check if schema already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        if cursor.fetchone():
            logger.debug("Schema already exists")
            return
        
        logger.info("Creating database schema...")
        
        # Documents table
        cursor.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                type TEXT DEFAULT 'notes',
                subject TEXT DEFAULT 'General',
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_chunks INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                file_hash TEXT UNIQUE
            )
        """)
        logger.debug("Created documents table")
        
        # Chunks/Metadata table
        cursor.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                indexed_chunk TEXT NOT NULL,
                page INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created chunks table")
        cursor.execute("CREATE INDEX idx_chunks_document ON chunks(document_id)")
        
        # TF-IDF vectors table
        cursor.execute("""
            CREATE TABLE vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER UNIQUE NOT NULL,
                vector_data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created vectors table")
        cursor.execute("CREATE INDEX idx_vectors_chunk ON vectors(chunk_id)")
        
        # Semantic vectors table
        cursor.execute("""
            CREATE TABLE semantic_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created semantic_vectors table")
        cursor.execute("CREATE INDEX idx_semantic_vectors_chunk ON semantic_vectors(chunk_id)")
        
        # Vectorizer state table
        cursor.execute("""
            CREATE TABLE vectorizer_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vectorizer_pickle BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT 1
            )
        """)
        logger.debug("Created vectorizer_state table")
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                document_id INTEGER,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
            )
        """)
        logger.debug("Created audit_log table")
        cursor.execute("CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp)")
        
        # Metadata table for schema version
        cursor.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        logger.debug("Created schema_version table")
        
        self.connection.commit()
        logger.info("Database schema created successfully")
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database operation error: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute SELECT query and return results"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution error: {query} | {e}")
            return []
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update execution error: {query} | {e}")
            return 0
    
    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def backup(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            logger.info(f"Database backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Document count
            result = self.execute_query("SELECT COUNT(*) as count FROM documents")
            stats['total_documents'] = result[0]['count'] if result else 0
            
            # Chunk count
            result = self.execute_query("SELECT COUNT(*) as count FROM chunks")
            stats['total_chunks'] = result[0]['count'] if result else 0
            
            # Database size
            if os.path.exists(self.db_path):
                stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            logger.debug(f"Database stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def get_db() -> Database:
    """Get database instance (singleton)"""
    return Database()