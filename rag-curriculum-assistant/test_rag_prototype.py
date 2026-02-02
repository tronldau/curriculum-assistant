# test_rag_prototype.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from rag.slm_openrouter import OpenRouterSLM
from config import Config

class SimpleRAG:
    def __init__(self):
        print("🔧 Initializing RAG System...")
        
        # Load embedding model
        print(" Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Qdrant
        print(" Connecting to Qdrant...")
        self.qdrant = QdrantConnector()
        
        # Load SLM via OpenRouter
        print(" Loading SLM (LLaMA 3.2 3B via OpenRouter)...")
        self.slm = OpenRouterSLM("meta-llama/llama-3.2-3b-instruct")
        
        print(" System Ready!\n")
    
    def search_courses(self, query, top_k=5):
        """Search for relevant courses"""
        print(f" Searching: '{query}'")
        
        # Generate embedding
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Search Qdrant (get more to account for duplicates)
        results = self.qdrant.search(
            collection_name="curriculum",
            query_vector=query_vector,
            limit=top_k * 3  # Get 3x to deduplicate
        )
        
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
        
        # Display & collect
        courses_info = []
        for i, result in enumerate(unique_results, 1):
            payload = result.payload
            score = result.score
            
            print(f"{i}. {payload['course_id']}: {payload['course_name']}")
            print(f"   Score: {score:.3f}")
            if payload.get('name_vn'):
                print(f"   Vietnamese: {payload['name_vn']}")
            print()
            
            courses_info.append({
                'id': payload['course_id'],
                'name': payload['course_name'],
                'name_vn': payload.get('name_vn', ''),
                'description': payload.get('description', ''),
                'credits': payload['credits_theory'] + payload['credits_lab'],
                'score': score
            })
        
        return courses_info
    
    def generate_answer(self, query, courses_info):
        """Generate answer with richer context"""
        print(" Generating answer with SLM (LLaMA 3.2 3B)...\n")
        
        # Build context with MORE details
        context = "Relevant courses from IU curriculum:\n\n"
        
        for i, course in enumerate(courses_info, 1):
            context += f"{i}. Course {course['id']}: {course['name']}\n"
            if course['name_vn']:
                context += f"   Vietnamese: {course['name_vn']}\n"
            context += f"   Credits: {course['credits']}\n"
            
            # ADD: More descriptive context
            if course['description']:
                context += f"   Description: {course['description'][:200]}...\n"
            
            # ADD: Relevance note
            context += f"   Relevance score: {course['score']:.2f}\n"
            
            # ADD: Logical inference hint for prerequisites
            if 'prerequisite' in query.lower():
                # Check if course name suggests prerequisite relationship
                course_name = course['name'].lower()
                if any(keyword in course_name for keyword in ['introduction', 'fundamental', 'basic', '1', 'i']):
                    context += f"   Note: This appears to be an introductory/prerequisite course\n"
            
            context += "\n"
        
        # Generate answer
        answer = self.slm.generate_answer(query, context, max_tokens=300)
        return answer
    
    def ask(self, query, top_k=5):
        """Complete RAG pipeline"""
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        print()
        
        # Search
        courses_info = self.search_courses(query, top_k)
        
        if not courses_info:
            print(" No courses found")
            return
        
        # Generate
        answer = self.generate_answer(query, courses_info)
        
        # Display
        print("=" * 60)
        print(" ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n")

# Main
def main():
    print("=" * 60)
    print(" IU CURRICULUM RAG ASSISTANT")
    print("   Small Language Model: LLaMA 3.2 3B")
    print("=" * 60)
    print()
    
    # Initialize
    rag = SimpleRAG()
    
    # Interactive mode
    print("=" * 60)
    print(" INTERACTIVE MODE")
    print("=" * 60)
    print("Type questions (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input(" Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            rag.ask(query, top_k=5)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    main()