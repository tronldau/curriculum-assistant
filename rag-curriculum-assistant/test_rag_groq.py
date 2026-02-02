# test_rag_groq.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from groq import Groq
from config import Config

class GroqRAG:
    def __init__(self):
        print(" Initializing RAG System with Groq...")
        
        # Load embedding model
        print(" Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Qdrant
        print(" Connecting to Qdrant...")
        self.qdrant = QdrantConnector()
        
        # Connect to Groq (LLaMA 8B)
        print(" Connecting to Groq API (LLaMA 3.1 8B)...")
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        
        print(" System Ready!\n")
    
    def search_courses(self, query, top_k=5):
        """Search for relevant courses"""
        print(f" Searching: '{query}'")
        
        # Generate query embedding
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        results = self.qdrant.search(
            collection_name="curriculum",
            query_vector=query_vector,
            limit=top_k * 3  # Get more for deduplication
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
        
        # Display results
        courses_info = []
        for i, result in enumerate(unique_results, 1):
            payload = result.payload
            score = result.score
            
            print(f"{i}. {payload['course_id']}: {payload['course_name']}")
            print(f"   Score: {score:.3f}")
            print(f"   Credits: {payload['credits_theory']} + {payload['credits_lab']}")
            if payload.get('name_vn'):
                print(f"   Vietnamese: {payload['name_vn']}")
            print()
            
            # Collect for context
            courses_info.append({
                'id': payload['course_id'],
                'name': payload['course_name'],
                'name_vn': payload.get('name_vn', ''),
                'description': payload.get('description', ''),
                'credits': payload['credits_theory'] + payload['credits_lab'],
                'program': payload.get('program', ''),
                'score': score
            })
        
        return courses_info
    
    def generate_answer(self, query, courses_info):
        """Generate answer using Groq LLM (LLaMA 8B)"""
        print(" Generating answer with Groq (LLaMA 3.1 8B)...\n")
        
        # Build context from search results
        context = "Relevant courses from IU curriculum:\n\n"
        for i, course in enumerate(courses_info, 1):
            context += f"{i}. Course {course['id']}: {course['name']}\n"
            if course['name_vn']:
                context += f"   Vietnamese Name: {course['name_vn']}\n"
            context += f"   Credits: {course['credits']}\n"
            if course['description']:
                desc = course['description'][:150]
                context += f"   Description: {desc}...\n"
            if course['program']:
                context += f"   Program: {course['program']}\n"
            context += "\n"
        
        # Create prompt for LLM
        system_prompt = """You are a helpful curriculum assistant for International University (IU).
Your role is to help faculty and staff quickly find information about courses.

Rules:
- Provide direct, concise answers
- Use bullet points for lists
- Always cite course IDs (e.g., IT079, CSAI301)
- If information is not in the provided context, say so clearly
- Be professional and accurate"""

        user_prompt = f"""Based on the following curriculum information, please answer this question:

Question: {query}

Curriculum Information:
{context}

Provide a clear, concise answer:"""

        # Call Groq API
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=Config.GROQ_MODEL,
                max_tokens=500,
                temperature=0.3  # Low temperature for factual responses
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def ask(self, query, top_k=5):
        """Complete RAG pipeline: Search + Generate"""
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        print()
        
        # Step 1: Search
        courses_info = self.search_courses(query, top_k)
        
        if not courses_info:
            print(" No relevant courses found")
            return
        
        # Step 2: Generate answer
        answer = self.generate_answer(query, courses_info)
        
        # Display answer
        print("=" * 60)
        print(" ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n")

# Interactive test
def main():
    print("=" * 60)
    print(" IU CURRICULUM RAG ASSISTANT")
    print("   Large Language Model: LLaMA 3.1 8B (Groq)")
    print("=" * 60)
    print()
    
    # Initialize RAG
    rag = GroqRAG()
    
    # Interactive mode
    print("=" * 60)
    print(" INTERACTIVE MODE")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input(" Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            rag.ask(query, top_k=5)
            print()
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    main()