import sys
import os

# Force correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Debug
print(f"📂 Current dir: {current_dir}")
print(f"📂 Parent dir: {parent_dir}")
print(f"📂 sys.path[0]: {sys.path[0]}\n")

from sentence_transformers import SentenceTransformer
from data.mysql_connector import MySQLConnector
from data.qdrant_connector import QdrantConnector
from qdrant_client.models import PointStruct
from tqdm import tqdm

class EmbeddingCreator:
    def __init__(self):
        print(" Initializing...")
        
        # Connect to MySQL
        self.mysql = MySQLConnector()
        if not self.mysql.connection:
            raise Exception("Failed to connect to MySQL")
        
        # Connect to Qdrant
        self.qdrant = QdrantConnector()
        if not self.qdrant.client:
            raise Exception("Failed to connect to Qdrant")
        
        # Load embedding model
        print(" Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" Model loaded\n")
    
    def create_course_text(self, course):
        """Create searchable text from course data"""
        text_parts = []
        
        # Course ID and name 
        if course.get('id') and course.get('name'):
            text_parts.append(f"Course {course['id']}: {course['name']}")
        
        # Vietnamese name
        if course.get('name_vn'):
            text_parts.append(f"Vietnamese: {course['name_vn']}")
        
        # Description
        if course.get('description'):
            desc = course['description'][:500]  # Limit to 500 chars
            text_parts.append(f"Description: {desc}")
        
        # Credits
        theory = course.get('credit_theory', 0) or 0
        lab = course.get('credit_lab', 0) or 0
        total = theory + lab
        if total > 0:
            text_parts.append(f"Credits: {total} ({theory} theory + {lab} lab)")
        
        # Program
        if course.get('program_name'):
            text_parts.append(f"Program: {course['program_name']}")
        
        return ". ".join(text_parts)
    
    def process_courses(self, batch_size=100, limit=None):
        """Extract courses, create embeddings, upload to Qdrant"""
        
        # 1. Get courses from MySQL
        print(" Extracting courses from MySQL...")
        courses = self.mysql.get_all_courses()
        
        if limit:
            courses = courses[:limit]
        
        print(f" Found {len(courses)} courses\n")
        
        # 2. Create collection
        print(" Creating Qdrant collection...")
        self.qdrant.create_collection("curriculum")
        print()
        
        # 3. Process in batches
        print(f" Creating embeddings and uploading (batch size: {batch_size})...\n")
        
        points = []
        total_uploaded = 0
        
        for idx, course in enumerate(tqdm(courses, desc="Processing")):
            try:
                # Create searchable text
                text = self.create_course_text(course)
                
                # Skip if text too short
                if len(text) < 10:
                    continue
                
                # Generate embedding
                embedding = self.model.encode(text).tolist()
                
                # Create point
                point = PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "course_id": course.get('id', ''),
                        "course_name": course.get('name', ''),
                        "name_vn": course.get('name_vn', ''),
                        "description": course.get('description', '')[:200] if course.get('description') else '',
                        "credits_theory": course.get('credit_theory', 0) or 0,
                        "credits_lab": course.get('credit_lab', 0) or 0,
                        "program": course.get('program_name', ''),
                        "text": text[:500]  # Store for display
                    }
                )
                points.append(point)
                
                # Upload in batches
                if len(points) >= batch_size:
                    success = self.qdrant.upsert_points("curriculum", points)
                    if success:
                        total_uploaded += len(points)
                        print(f"   Uploaded batch: {total_uploaded}/{len(courses)}")
                    points = []
                
            except Exception as e:
                print(f"    Error processing course {course.get('id', '?')}: {e}")
                continue
        
        # Upload remaining points
        if points:
            success = self.qdrant.upsert_points("curriculum", points)
            if success:
                total_uploaded += len(points)
                print(f"   Uploaded final batch: {total_uploaded}/{len(courses)}")
        
        print(f"\n Complete! Uploaded {total_uploaded} courses to Qdrant")
        
        # 4. Verify
        info = self.qdrant.get_collection_info("curriculum")
        if info:
            print(f"\n Qdrant Collection Stats:")
            print(f"   Total points: {info.points_count}")
            print(f"   Status: {info.status}")
        
        # Cleanup
        self.mysql.close()

# Run
if __name__ == "__main__":
    print("=" * 60)
    print(" CREATING EMBEDDINGS & UPLOADING TO QDRANT")
    print("=" * 60)
    print()
    
    try:
        creator = EmbeddingCreator()
        
        # Process all courses (or limit for testing)
        # creator.process_courses(batch_size=100, limit=50)  # Test with 50 first
        creator.process_courses(batch_size=100)  # All courses
        
        print("\n" + "=" * 60)
        print(" SUCCESS!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n Error: {e}")