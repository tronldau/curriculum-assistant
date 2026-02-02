# test_search_only.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from config import Config

class SearchOnlyTest:
    def __init__(self):
        print(" Initializing Search Test System...")
        print("   (No SLM - Search Only)\n")
        
        # Load embedding model
        print(" Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Qdrant
        print(" Connecting to Qdrant...")
        self.qdrant = QdrantConnector()
        
        print(" Ready!\n")
    
    def search(self, query, top_k=10):
        """Search courses in Qdrant"""
        print("=" * 60)
        print(f" Query: {query}")
        print("=" * 60)
        print()
        
        # Generate embedding
        print(" Generating query embedding...")
        query_vector = self.embedding_model.encode(query).tolist()
        print(f"   Vector dimension: {len(query_vector)}")
        print()
        
        # Search Qdrant
        print(f" Searching Qdrant (top {top_k})...")
        results = self.qdrant.search(
            collection_name="curriculum",
            query_vector=query_vector,
            limit=top_k * 3  # Get more for deduplication
        )
        print(f"   Raw results: {len(results)}")
        print()
        
        # Deduplicate by course_id
        seen_courses = set()
        unique_results = []
        
        for result in results:
            course_id = result.payload['course_id']
            if course_id not in seen_courses:
                seen_courses.add(course_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        print(f" Found {len(unique_results)} unique courses:\n")
        
        # Display results
        print("=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print()
        
        for i, result in enumerate(unique_results, 1):
            payload = result.payload
            score = result.score
            
            # Main info
            print(f"{i}. {payload['course_id']}: {payload['course_name']}")
            print(f"   Score: {score:.4f} ({self.score_label(score)})")
            
            # Vietnamese name
            if payload.get('name_vn'):
                print(f"   Vietnamese: {payload['name_vn']}")
            
            # Credits
            theory = payload.get('credits_theory', 0)
            lab = payload.get('credits_lab', 0)
            total = theory + lab
            if lab > 0:
                print(f"   Credits: {total} ({theory} theory + {lab} lab)")
            else:
                print(f"   Credits: {total}")
            
            # Program
            if payload.get('program'):
                print(f"   Program: {payload['program']}")
            
            # Description preview
            if payload.get('description'):
                desc = payload['description'][:150].strip()
                print(f"   Description: {desc}...")
            
            # Text used for embedding
            if payload.get('text'):
                text_preview = payload['text'][:100].strip()
                print(f"   Indexed text: {text_preview}...")
            
            print()
        
        # Statistics
        self.print_statistics(unique_results)
    
    def score_label(self, score):
        """Convert score to label"""
        if score >= 0.8:
            return "Excellent match"
        elif score >= 0.7:
            return "Very good match"
        elif score >= 0.6:
            return "Good match"
        elif score >= 0.5:
            return "Moderate match"
        elif score >= 0.4:
            return "Weak match"
        else:
            return "Poor match"
    
    def print_statistics(self, results):
        """Print search statistics"""
        if not results:
            return
        
        print("=" * 60)
        print(" STATISTICS:")
        print("=" * 60)
        
        scores = [r.score for r in results]
        
        print(f"Total results: {len(results)}")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Average score: {sum(scores)/len(scores):.4f}")
        print()
        
        # Score distribution
        excellent = sum(1 for s in scores if s >= 0.8)
        very_good = sum(1 for s in scores if 0.7 <= s < 0.8)
        good = sum(1 for s in scores if 0.6 <= s < 0.7)
        moderate = sum(1 for s in scores if 0.5 <= s < 0.6)
        weak = sum(1 for s in scores if 0.4 <= s < 0.5)
        poor = sum(1 for s in scores if s < 0.4)
        
        print("Score distribution:")
        if excellent > 0:
            print(f"  Excellent (≥0.8): {excellent}")
        if very_good > 0:
            print(f"  Very good (0.7-0.8): {very_good}")
        if good > 0:
            print(f"  Good (0.6-0.7): {good}")
        if moderate > 0:
            print(f"  Moderate (0.5-0.6): {moderate}")
        if weak > 0:
            print(f"  Weak (0.4-0.5): {weak}")
        if poor > 0:
            print(f"  Poor (<0.4): {poor}")
        print()
    
    def batch_test(self, queries):
        """Test multiple queries"""
        print("=" * 60)
        print(" BATCH TEST MODE")
        print("=" * 60)
        print()
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}/{len(queries)}")
            print(f"{'='*60}\n")
            
            self.search(query, top_k=5)
            
            if i < len(queries):
                input("\nPress Enter to continue...")

# Main
def main():
    print("=" * 60)
    print("🔍 QDRANT SEARCH TEST")
    print("   Search Only - No SLM")
    print("=" * 60)
    print()
    
    searcher = SearchOnlyTest()
    
    # Menu
    while True:
        print("=" * 60)
        print("MENU:")
        print("=" * 60)
        print("1. Interactive search")
        print("2. Batch test (predefined queries)")
        print("3. Quick test (single query)")
        print("4. Quit")
        print()
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            # Interactive mode
            print("\n" + "=" * 60)
            print(" INTERACTIVE MODE")
            print("=" * 60)
            print("Type queries (or 'back' to return)\n")
            
            while True:
                query = input(" Search: ").strip()
                
                if query.lower() in ['back', 'b', 'menu']:
                    break
                
                if not query:
                    continue
                
                print()
                
                # Ask for top_k
                try:
                    top_k_input = input("Top K results (default 10): ").strip()
                    top_k = int(top_k_input) if top_k_input else 10
                except:
                    top_k = 10
                
                print()
                searcher.search(query, top_k)
                print()
        
        elif choice == '2':
            # Batch test
            test_queries = [
                "What AI and machine learning courses are available?",
                "Show me database courses",
                "Which courses teach programming?",
                "What are data science courses?",
                "Courses about web development",
                "Mathematics courses",
                "Security and cryptography",
                "Introduction to computer science",
                "Advanced algorithms",
                "Data structures",
            ]
            
            searcher.batch_test(test_queries)
        
        elif choice == '3':
            # Quick test
            print()
            query = input("🔍 Enter query: ").strip()
            if query:
                print()
                searcher.search(query, top_k=5)
                print()
        
        elif choice == '4':
            print("\n Goodbye!")
            break
        
        else:
            print("\n Invalid option. Try again.\n")

if __name__ == "__main__":
    main()