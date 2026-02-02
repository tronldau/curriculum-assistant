# test_rag_comparison.py
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from rag.slm_openrouter import OpenRouterSLM
from groq import Groq
from config import Config

class RAGComparison:
    def __init__(self):
        print(" Initializing Comparison System...")
        print("   Loading both SLM and LLM...\n")
        
        # Shared components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant = QdrantConnector()
        
        # SLM: LLaMA 3.2 3B (OpenRouter)
        print(" Loading SLM (LLaMA 3.2 3B)...")
        self.slm = OpenRouterSLM("meta-llama/llama-3.2-3b-instruct")
        
        # LLM: LLaMA 3.1 8B (Groq)
        print(" Loading LLM (LLaMA 3.1 8B)...")
        self.groq = Groq(api_key=Config.GROQ_API_KEY)
        
        print(" Ready!\n")
    
    def search_courses(self, query, top_k=5):
        """Search (shared by both models)"""
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.qdrant.search(
            collection_name="curriculum",
            query_vector=query_vector,
            limit=top_k * 3
        )
        
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            cid = r.payload['course_id']
            if cid not in seen:
                seen.add(cid)
                unique.append(r)
                if len(unique) >= top_k:
                    break
        
        # Build info
        courses_info = []
        for result in unique:
            payload = result.payload
            courses_info.append({
                'id': payload['course_id'],
                'name': payload['course_name'],
                'name_vn': payload.get('name_vn', ''),
                'description': payload.get('description', ''),
                'credits': payload['credits_theory'] + payload['credits_lab'],
            })
        
        return courses_info
    
    def build_context(self, courses_info):
        """Build context (shared)"""
        context = "Relevant courses:\n\n"
        for i, course in enumerate(courses_info, 1):
            context += f"{i}. {course['id']}: {course['name']}\n"
            if course['name_vn']:
                context += f"   Vietnamese: {course['name_vn']}\n"
            context += f"   Credits: {course['credits']}\n"
            if course['description']:
                context += f"   {course['description'][:100]}...\n"
            context += "\n"
        return context
    
    def generate_with_slm(self, query, context):
        """Generate with SLM (3B)"""
        start = time.time()
        answer = self.slm.generate_answer(query, context, max_tokens=300)
        elapsed = time.time() - start
        return answer, elapsed
    
    def generate_with_llm(self, query, context):
        """Generate with LLM (8B)"""
        system_prompt = """You are a curriculum assistant for International University.
Provide concise answers. Cite course IDs. Use bullet points for lists."""

        user_prompt = f"""Curriculum:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        
        start = time.time()
        response = self.groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=Config.GROQ_MODEL,
            max_tokens=300,
            temperature=0.3
        )
        elapsed = time.time() - start
        
        return response.choices[0].message.content, elapsed
    
    def compare(self, query):
        """Compare both models on same query"""
        print("=" * 60)
        print(f" Query: {query}")
        print("=" * 60)
        print()
        
        # Search (once for both)
        print(" Searching courses...")
        courses_info = self.search_courses(query, top_k=5)
        print(f"   Found {len(courses_info)} courses\n")
        
        # Build context
        context = self.build_context(courses_info)
        
        # Generate with SLM
        print(" Generating with SLM (LLaMA 3.2 3B)...")
        slm_answer, slm_time = self.generate_with_slm(query, context)
        print(f"     Time: {slm_time:.2f}s\n")
        
        # Generate with LLM
        print(" Generating with LLM (LLaMA 3.1 8B)...")
        llm_answer, llm_time = self.generate_with_llm(query, context)
        print(f"     Time: {llm_time:.2f}s\n")
        
        # Display comparison
        print("=" * 60)
        print(" COMPARISON RESULTS")
        print("=" * 60)
        print()
        
        print("┌─ SLM (3B) Answer ─────────────────────────────────┐")
        print(slm_answer)
        print("└───────────────────────────────────────────────────┘")
        print()
        
        print("┌─ LLM (8B) Answer ─────────────────────────────────┐")
        print(llm_answer)
        print("└───────────────────────────────────────────────────┘")
        print()
        
        print("=" * 60)
        print("  PERFORMANCE:")
        print("=" * 60)
        print(f"SLM (3B): {slm_time:.2f}s")
        print(f"LLM (8B): {llm_time:.2f}s")
        print(f"Speed ratio: {llm_time/slm_time:.2f}x {'faster' if llm_time < slm_time else 'slower'}")
        print()

# Main
def main():
    print("=" * 60)
    print(" RAG SYSTEM COMPARISON")
    print("   SLM (3B) vs LLM (8B)")
    print("=" * 60)
    print()
    
    comparator = RAGComparison()
    
    # Test queries
    test_queries = [
        "What AI and machine learning courses are available?",
        "Show me database courses",
        "Which courses teach programming?",
        "What are data science courses?",
    ]
    
    print("=" * 60)
    print(" COMPARISON TEST")
    print("=" * 60)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*60}\n")
        
        comparator.compare(query)
        
        if i < len(test_queries):
            input("\nPress Enter for next test...")
    
    print("\n" + "=" * 60)
    print(" COMPARISON COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()