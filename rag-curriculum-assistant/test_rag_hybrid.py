# test_rag_hybrid.py
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from data.mysql_connector import MySQLConnector
from rag.slm_openrouter import OpenRouterSLM
from config import Config

class HybridRAG:
    def __init__(self):
        print(" Initializing Hybrid RAG System...")
        
        print(" Loading components...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant = QdrantConnector()
        self.mysql = MySQLConnector()
        self.slm = OpenRouterSLM("meta-llama/llama-3.2-3b-instruct")
        
        print(" System Ready!\n")
    
    def extract_course_identifier(self, query):
        """Extract course name/ID from query - IMPROVED"""
        # First try: Direct course ID pattern (IT079, MA001, etc.)
        course_id_pattern = r'\b[A-Z]{2,4}\d{3,4}\b'
        matches = re.findall(course_id_pattern, query.upper())
        if matches:
            return matches[0]
        
        # Second try: Extract key course name
        query_lower = query.lower()
        
        # Remove common question words
        stop_words = [
            'what', 'are', 'is', 'the', 'prerequisites', 'for', 'of', 'course',
            'which', 'courses', 'require', 'need', 'before', 'taking', 'to',
            'do', 'i', 'as', 'prerequisite', 'a', 'an', '?', 'show', 'me', 'all'
        ]
        
        words = query_lower.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Join meaningful keywords
        course_name = ' '.join(keywords)
        
        return course_name.strip()
    
    def classify_query(self, query):
        """
        Classify query type - IMPROVED LOGIC
        
        Returns:
            'prerequisite': What does X need? (prerequisites for X)
            'dependent': What needs X? (courses that require X)
            'semantic': General search
        """
        query_lower = query.lower()
        
        # Pattern 1: "prerequisites for X" or "what do I need before X"
        prerequisite_patterns = [
            'prerequisite',
            'prereq',
            'what do i need before',
            'what should i take before',
            'what is required for',
            'requirements for',
        ]
        
        if any(pattern in query_lower for pattern in prerequisite_patterns):
            return 'prerequisite'
        
        # Pattern 2: "which courses require X" or "what requires X"
        dependent_patterns = [
            'which courses require',
            'which course require',
            'what courses require',
            'what require',
            'courses that need',
            'courses that require',
        ]
        
        if any(pattern in query_lower for pattern in dependent_patterns):
            return 'dependent'
        
        # Pattern 3: Check word order for "require X" vs "require for X"
        if 'require' in query_lower:
            # "require MA001" or "require calculus" → dependent
            # "require for IT079" → prerequisite
            if 'for' in query_lower and query_lower.index('require') < query_lower.index('for'):
                return 'prerequisite'
            else:
                return 'dependent'
        
        # Default: semantic search
        return 'semantic'
    
    def search_courses(self, query, top_k=5):
        """Semantic search for courses"""
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
        
        return unique
    
    def handle_prerequisite_query(self, query):
        """Handle prerequisite questions using SQL"""
        print(" Query Type: PREREQUISITE")
        print("   (What does X need to take it?)")
        print("   Using SQL lookup...\n")
        
        # Extract course
        course_identifier = self.extract_course_identifier(query)
        print(f"   Looking for course: '{course_identifier}'\n")
        
        # Find course
        courses = self.mysql.find_course_by_name(course_identifier)
        
        if not courses:
            print(" Course not found in database\n")
            # Fallback to semantic search
            print(" Falling back to semantic search...\n")
            return None, self.handle_semantic_query(query)
        
        course = courses[0]
        course_id = course['id']
        course_name = course['name']
        
        print(f" Found course: {course_id}: {course_name}\n")
        
        # Get prerequisites
        prereqs = self.mysql.get_prerequisites(course_id)
        
        if not prereqs:
            answer = f" {course_id} ({course_name}) has NO prerequisites.\n"
            answer += "This course can be taken without prior coursework."
            return course, answer
        
        # Format answer
        answer = f"Prerequisites for {course_id} ({course_name}):\n\n"
        
        for i, prereq in enumerate(prereqs, 1):
            answer += f"{i}. {prereq['id']}: {prereq['name']}\n"
            
            if prereq.get('name_vn'):
                answer += f"   Vietnamese: {prereq['name_vn']}\n"
            
            theory = prereq.get('credit_theory', 0) or 0
            lab = prereq.get('credit_lab', 0) or 0
            total = prereq.get('total_credits', 0) or 0
            
            if lab > 0:
                answer += f"   Credits: {total} ({theory} theory + {lab} lab)\n"
            else:
                answer += f"   Credits: {total}\n"
            
            if prereq.get('description'):
                desc = prereq['description'][:200].strip()
                answer += f"   Description: {desc}...\n"
            
            if prereq.get('course_level_id'):
                answer += f"   Level: {prereq['course_level_id']}\n"
            
            answer += "\n"
        
        return course, answer
    
    def handle_dependent_query(self, query):
        """Handle 'what requires X' questions"""
        print(" Query Type: DEPENDENT")
        print("   (What courses need X as prerequisite?)")
        print("   Using SQL lookup...\n")
        
        # Extract course
        course_identifier = self.extract_course_identifier(query)
        print(f"   Looking for course: '{course_identifier}'\n")
        
        # Find course
        courses = self.mysql.find_course_by_name(course_identifier)
        
        if not courses:
            print(" Course not found in database\n")
            return None, "Course not found. Please check the course ID or name."
        
        course = courses[0]
        course_id = course['id']
        course_name = course['name']
        
        print(f" Found course: {course_id}: {course_name}\n")
        
        # Get dependent courses
        dependents = self.mysql.get_dependent_courses(course_id)
        
        if not dependents:
            answer = f" No courses require {course_id} ({course_name}) as prerequisite.\n"
            answer += "This course is not a prerequisite for any other courses in the current curriculum."
            return course, answer
        
        # Format answer
        answer = f"Courses that require {course_id} ({course_name}) as prerequisite:\n\n"
        answer += f"Total: {len(dependents)} courses\n\n"
        
        for i, dep in enumerate(dependents, 1):
            answer += f"{i}. {dep['id']}: {dep['name']}\n"
            
            if dep.get('name_vn'):
                answer += f"   Vietnamese: {dep['name_vn']}\n"
            
            answer += f"   Credits: {dep['total_credits']}\n"
            
            if dep.get('description'):
                desc = dep['description'][:150].strip()
                answer += f"   Description: {desc}...\n"
            
            answer += "\n"
        
        return course, answer
    
    def handle_semantic_query(self, query, top_k=5):
        """Handle general queries with semantic search + SLM"""
        print("🔍 Query Type: SEMANTIC SEARCH")
        print("   Using vector search + SLM\n")
        
        # Search
        results = self.search_courses(query, top_k)
        
        if not results:
            return "No relevant courses found"
        
        print(f"📚 Found {len(results)} courses:\n")
        
        # Display
        courses_info = []
        for i, result in enumerate(results, 1):
            payload = result.payload
            print(f"{i}. {payload['course_id']}: {payload['course_name']}")
            print(f"   Score: {result.score:.3f}")
            if payload.get('name_vn'):
                print(f"   Vietnamese: {payload['name_vn']}")
            print()
            
            courses_info.append({
                'id': payload['course_id'],
                'name': payload['course_name'],
                'name_vn': payload.get('name_vn', ''),
                'description': payload.get('description', ''),
                'credits': payload['credits_theory'] + payload['credits_lab'],
            })
        
        # Build context
        context = "Relevant courses:\n\n"
        for i, course in enumerate(courses_info, 1):
            context += f"{i}. {course['id']}: {course['name']}\n"
            if course['name_vn']:
                context += f"   Vietnamese: {course['name_vn']}\n"
            context += f"   Credits: {course['credits']}\n"
            if course['description']:
                context += f"   {course['description'][:150]}...\n"
            context += "\n"
        
        # Generate
        print(" Generating answer with SLM...\n")
        answer = self.slm.generate_answer(query, context, max_tokens=300)
        
        return answer
    
    def ask(self, query):
        """Main query handler with improved routing"""
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        print()
        
        # Classify query type
        query_type = self.classify_query(query)
        
        # Route to handler
        if query_type == 'prerequisite':
            course, answer = self.handle_prerequisite_query(query)
            
        elif query_type == 'dependent':
            course, answer = self.handle_dependent_query(query)
            
        else:  # semantic
            answer = self.handle_semantic_query(query)
        
        # Display
        print("=" * 60)
        print(" ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n")

# Main
def main():
    print("=" * 60)
    print(" HYBRID RAG SYSTEM v2.0")
    print("   Improved Query Classification")
    print("=" * 60)
    print()
    
    rag = HybridRAG()
    
    print("=" * 60)
    print(" INTERACTIVE MODE")
    print("=" * 60)
    print("Type questions (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input(" Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                rag.mysql.close()
                break
            
            if not query:
                continue
            
            print()
            rag.ask(query)
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            rag.mysql.close()
            break
        except Exception as e:
            print(f"\n Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()