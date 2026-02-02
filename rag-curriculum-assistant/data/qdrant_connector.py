# data/qdrant_connector.py
import sys
import os

# Force add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug: Print what we're importing
print(f"📂 Current dir: {current_dir}")
print(f"📂 Parent dir: {parent_dir}")
print(f"📂 sys.path[0]: {sys.path[0]}")

# Import with reload to avoid cache
import importlib
import config
importlib.reload(config)
from config import Config

# Debug: Print config values
print(f"\n🔍 Debug Config:")
print(f"   QDRANT_URL exists: {hasattr(Config, 'QDRANT_URL')}")
if hasattr(Config, 'QDRANT_URL'):
    url = Config.QDRANT_URL
    print(f"   QDRANT_URL: {url[:50] if url else 'None'}...")
print(f"   QDRANT_API_KEY exists: {hasattr(Config, 'QDRANT_API_KEY')}")
if hasattr(Config, 'QDRANT_API_KEY'):
    print(f"   QDRANT_API_KEY: {'Set' if Config.QDRANT_API_KEY else 'Not Set'}")
print()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantConnector:
    def __init__(self):
        try:
            print(f"🔗 Connecting to: {Config.QDRANT_URL[:50]}...")
            self.client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
            )
            print("✅ Connected to Qdrant Cloud")
        except Exception as e:
            print(f"❌ Qdrant Connection Error: {e}")
            self.client = None
    
    def create_collection(self, collection_name="curriculum"):
        """Create collection for storing embeddings"""
        if not self.client:
            return False
        
        try:
            # Delete if exists
            try:
                self.client.delete_collection(collection_name)
                print(f"🗑️  Deleted existing collection: {collection_name}")
            except:
                pass
            
            # Create new
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=Config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
    
    def upsert_points(self, collection_name, points):
        """Upload points to Qdrant"""
        if not self.client:
            return False
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"❌ Upload Error: {e}")
            return False
    
    def search(self, collection_name, query_vector, limit=5):
        """Search for similar vectors"""
        if not self.client:
            return []
    
        try:
            # Updated API for newer qdrant-client versions
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit
            ).points
            return results
        except Exception as e:
            # Try old API as fallback
            try:
                from qdrant_client.models import SearchRequest
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                return results
            except Exception as e2:
                print(f"❌ Search Error: {e}")
                return []
    
    def get_collection_info(self, collection_name="curriculum"):
        """Get info about collection"""
        if not self.client:
            return None
        
        try:
            info = self.client.get_collection(collection_name)
            return info
        except Exception as e:
            print(f"ℹ️  Collection doesn't exist yet")
            return None

# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qdrant Connection...")
    print("=" * 60)
    print()
    
    qdrant = QdrantConnector()
    
    if qdrant.client:
        # Test create collection
        qdrant.create_collection("test_collection")
        
        # Get info
    info = qdrant.get_collection_info("test_collection")
    if info:
        print(f"\n📊 Collection Info:")
        print(f"   Points: {info.points_count}")
        print(f"   Status: {info.status}")
    else:
        print("❌ Failed to connect to Qdrant")