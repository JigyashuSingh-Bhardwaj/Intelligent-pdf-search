"""
Migration script: Convert from pickle files to SQLite database
Safely migrates all data while keeping pickle files as backup
"""

import os
import pickle
import logging
import shutil
from datetime import datetime
from pathlib import Path

from modules.db import get_db
from modules.db_layer import (
    DocumentManager, ChunkManager, VectorManager,
    VectorizerManager, AuditManager
)
from modules.config import DATA_PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PickleToSQLiteMigration:
    """Handles migration from pickle to SQLite"""
    
    def __init__(self):
        self.backup_dir = "data/backups"
        self.migration_log = []
        self.errors = []
    
    def create_backup(self) -> bool:
        """Create backup of current pickle files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.backup_dir, f"pickle_backup_{timestamp}")
            
            os.makedirs(backup_dir, exist_ok=True)
            logger.info(f"Creating backup in: {backup_dir}")
            
            for key, path in DATA_PATHS.items():
                if os.path.exists(path):
                    backup_path = os.path.join(backup_dir, os.path.basename(path))
                    shutil.copy2(path, backup_path)
                    logger.info(f"Backed up: {path} → {backup_path}")
                    self.migration_log.append(f"✓ Backed up {path}")
            
            return True
        except Exception as e:
            error_msg = f"Backup creation failed: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False
    
    def load_pickle_data(self) -> dict:
        """Load all data from pickle files"""
        try:
            data = {
                'metadata': [],
                'vectorizer': None,
                'vectors': None,
                'semantic_vectors': None
            }
            
            # Load metadata
            if os.path.exists(DATA_PATHS['metadata']):
                with open(DATA_PATHS['metadata'], 'rb') as f:
                    data['metadata'] = pickle.load(f)
                logger.info(f"Loaded {len(data['metadata'])} metadata items")
                self.migration_log.append(f"✓ Loaded {len(data['metadata'])} chunks from pickle")
            
            # Load vectorizer
            if os.path.exists(DATA_PATHS['vectorizer']):
                with open(DATA_PATHS['vectorizer'], 'rb') as f:
                    data['vectorizer'] = pickle.load(f)
                logger.info("Loaded vectorizer")
                self.migration_log.append("✓ Loaded vectorizer")
            
            # Load TF-IDF vectors
            if os.path.exists(DATA_PATHS['vectors']):
                with open(DATA_PATHS['vectors'], 'rb') as f:
                    data['vectors'] = pickle.load(f)
                logger.info(f"Loaded TF-IDF vectors: {data['vectors'].shape if hasattr(data['vectors'], 'shape') else len(data['vectors'])}")
                self.migration_log.append("✓ Loaded TF-IDF vectors")
            
            # Load semantic vectors
            if os.path.exists(DATA_PATHS['semantic_vectors']):
                with open(DATA_PATHS['semantic_vectors'], 'rb') as f:
                    data['semantic_vectors'] = pickle.load(f)
                logger.info(f"Loaded semantic vectors: {len(data['semantic_vectors']) if hasattr(data['semantic_vectors'], '__len__') else 'unknown'}")
                self.migration_log.append("✓ Loaded semantic vectors")
            
            return data
        except Exception as e:
            error_msg = f"Error loading pickle data: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
    
    def migrate_data(self, data: dict) -> bool:
        """Migrate data to SQLite database"""
        try:
            if not data or not data['metadata']:
                logger.warning("No metadata to migrate")
                return True
            
            # Group chunks by document
            documents = {}
            
            for item in data['metadata']:
                doc_name = item.get('document', 'Unknown')
                
                if doc_name not in documents:
                    documents[doc_name] = {
                        'type': item.get('type', 'notes'),
                        'subject': item.get('subject', 'General'),
                        'chunks': []
                    }
                
                documents[doc_name]['chunks'].append(item)
            
            logger.info(f"Processing {len(documents)} documents...")
            
            # Migrate each document
            doc_id_map = {}  # Map original document names to new IDs
            chunk_ids_all = []
            
            for doc_name, doc_data in documents.items():
                try:
                    # Add document
                    doc_id = DocumentManager.add_document(
                        filename=doc_name,
                        doc_type=doc_data['type'],
                        subject=doc_data['subject']
                    )
                    doc_id_map[doc_name] = doc_id
                    
                    # Add chunks
                    chunks = doc_data['chunks']
                    chunk_ids = ChunkManager.add_chunks(doc_id, chunks)
                    chunk_ids_all.extend(chunk_ids)
                    
                    # Update chunk count
                    DocumentManager.update_chunk_count(doc_id, len(chunks))
                    
                    logger.info(f"Migrated document: {doc_name} ({len(chunks)} chunks)")
                    self.migration_log.append(f"✓ Migrated {doc_name} ({len(chunks)} chunks)")
                
                except Exception as e:
                    error_msg = f"Error migrating document {doc_name}: {e}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    continue
            
            # Migrate vectors if they exist
            if data['vectors'] is not None:
                try:
                    VectorManager.add_tfidf_vectors(chunk_ids_all, data['vectors'])
                    self.migration_log.append("✓ Migrated TF-IDF vectors")
                except Exception as e:
                    error_msg = f"Error migrating TF-IDF vectors: {e}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
            
            if data['semantic_vectors'] is not None:
                try:
                    VectorManager.add_semantic_vectors(chunk_ids_all, data['semantic_vectors'])
                    self.migration_log.append("✓ Migrated semantic vectors")
                except Exception as e:
                    error_msg = f"Error migrating semantic vectors: {e}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
            
            # Migrate vectorizer
            if data['vectorizer'] is not None:
                try:
                    VectorizerManager.save_vectorizer(data['vectorizer'])
                    self.migration_log.append("✓ Migrated vectorizer")
                except Exception as e:
                    error_msg = f"Error migrating vectorizer: {e}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
            
            logger.info("Data migration completed")
            return True
        
        except Exception as e:
            error_msg = f"Data migration failed: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False
    
    def verify_migration(self) -> bool:
        """Verify that data was migrated correctly"""
        try:
            db = get_db()
            stats = db.get_stats()
            
            logger.info(f"Database stats after migration: {stats}")
            self.migration_log.append(f"✓ Database stats: {stats}")
            
            if stats['total_chunks'] == 0:
                logger.warning("No chunks found in database after migration")
                return False
            
            return True
        except Exception as e:
            error_msg = f"Verification failed: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False
    
    def run_migration(self) -> bool:
        """Run complete migration process"""
        logger.info("=" * 60)
        logger.info("PICKLE TO SQLITE MIGRATION")
        logger.info("=" * 60)
        
        # Step 1: Backup
        logger.info("\n[1/5] Creating backup...")
        if not self.create_backup():
            logger.error("Backup failed, aborting migration")
            return False
        
        # Step 2: Load pickle data
        logger.info("\n[2/5] Loading pickle data...")
        data = self.load_pickle_data()
        if data is None:
            logger.error("Failed to load pickle data")
            return False
        
        # Step 3: Migrate data
        logger.info("\n[3/5] Migrating data to SQLite...")
        if not self.migrate_data(data):
            logger.error("Data migration failed")
            return False
        
        # Step 4: Verify migration
        logger.info("\n[4/5] Verifying migration...")
        if not self.verify_migration():
            logger.error("Verification failed")
            return False
        
        # Step 5: Summary
        logger.info("\n[5/5] Migration summary")
        logger.info("=" * 60)
        
        for log_entry in self.migration_log:
            logger.info(log_entry)
        
        if self.errors:
            logger.warning(f"\n⚠️  {len(self.errors)} errors occurred during migration:")
            for error in self.errors:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
        logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Verify application works with new database")
        logger.info("2. Old pickle files are backed up in: data/backups/")
        logger.info("3. When confident, delete old pickle files to save space")
        
        return True


def main():
    """Run migration"""
    try:
        migration = PickleToSQLiteMigration()
        success = migration.run_migration()
        
        if not success:
            logger.error("\nMigration failed! Pickle files remain intact as backup.")
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
