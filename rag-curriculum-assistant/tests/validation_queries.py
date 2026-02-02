# tests/validation_queries.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mysql_connector import MySQLConnector
from sentence_transformers import SentenceTransformer
from data.qdrant_connector import QdrantConnector
from rag.slm_openrouter import OpenRouterSLM

class ValidationTest:
    def __init__(self):
        print("🔧 Initializing Validation System...")
        
        # MySQL for ground truth
        self.mysql = MySQLConnector()
        
        # RAG components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant = QdrantConnector()
        self.slm = OpenRouterSLM("meta-llama/llama-3.2-3b-instruct")
        
        print("✅ Ready!\n")
    
    # ========== GROUND TRUTH SQL QUERIES ==========
    
    def get_ai_courses_sql(self):
        """Ground truth: AI/ML courses from SQL"""
        query = """
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory + c.credit_lab as total_credits
        FROM course c
        WHERE 
            c.name LIKE '%artificial intelligence%'
            OR c.name LIKE '%machine learning%'
            OR c.name LIKE '%deep learning%'
            OR c.name LIKE '%AI%'
            OR c.name LIKE '%ML%'
            OR c.name_vn LIKE '%trí tuệ nhân tạo%'
            OR c.name_vn LIKE '%học máy%'
        ORDER BY c.id
        """
        return self.mysql.execute_query(query)
    
    def get_database_courses_sql(self):
        """Ground truth: Database courses from SQL"""
        query = """
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory + c.credit_lab as total_credits
        FROM course c
        WHERE 
            c.name LIKE '%database%'
            OR c.name_vn LIKE '%cơ sở dữ liệu%'
            OR c.name_vn LIKE '%dữ liệu%'
        ORDER BY c.id
        """
        return self.mysql.execute_query(query)
    
    def get_programming_courses_sql(self):
        """Ground truth: Programming courses from SQL"""
        query = """
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory + c.credit_lab as total_credits
        FROM course c
        WHERE 
            c.name LIKE '%programming%'
            OR c.name LIKE '%Python%'
            OR c.name_vn LIKE '%lập trình%'
        ORDER BY c.id
        """
        return self.mysql.execute_query(query)
    
    def get_prerequisites_sql(self, course_id):
        """Ground truth: Prerequisites for a course"""
        query = f"""
        SELECT DISTINCT
            c.id,
            c.name,
            pr.type as prerequisite_type
        FROM prerequisite pr
        JOIN course c ON pr.course_prerequisite_id = c.id
        WHERE pr.course_id = '{course_id}'
        ORDER BY c.id
        """
        return self.mysql.execute_query(query)
    
    def get_course_by_id_sql(self, course_id):
        """Ground truth: Get specific course details"""
        query = f"""
        SELECT 
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory,
            c.credit_lab,
            c.credit_theory + c.credit_lab as total_credits
        FROM course c
        WHERE c.id = '{course_id}'
        """
        results = self.mysql.execute_query(query)
        return results[0] if results else None
    
    def get_courses_by_program_sql(self, program_name):
        """Ground truth: Courses for a specific program"""
        query = f"""
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.credit_theory + c.credit_lab as total_credits,
            p.name as program_name
        FROM course c
        JOIN course_program cp ON c.id = cp.course_id
        JOIN program p ON cp.program_id = p.id
        WHERE p.name LIKE '%{program_name}%'
        ORDER BY c.id
        """
        return self.mysql.execute_query(query)
    
    # ========== RAG QUERY ==========
    
    def query_rag(self, question, top_k=5):
        """Get answer from RAG system"""
        # Generate embedding
        query_vector = self.embedding_model.encode(question).tolist()
        
        # Search Qdrant
        results = self.qdrant.search(
            collection_name="curriculum",
            query_vector=query_vector,
            limit=top_k * 3
        )
        
        # Deduplicate
        seen_courses = set()
        unique_results = []
        
        for result in results:
            course_id = result.payload['course_id']
            if course_id not in seen_courses:
                seen_courses.add(course_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        # Build context
        context = "Relevant courses:\n\n"
        retrieved_courses = []
        
        for i, result in enumerate(unique_results, 1):
            payload = result.payload
            context += f"{i}. {payload['course_id']}: {payload['course_name']}\n"
            if payload.get('name_vn'):
                context += f"   Vietnamese: {payload['name_vn']}\n"
            context += f"   Credits: {payload['credits_theory'] + payload['credits_lab']}\n"
            if payload.get('description'):
                context += f"   Description: {payload['description'][:100]}...\n"
            context += "\n"
            
            retrieved_courses.append(payload['course_id'])
        
        # Generate answer
        answer = self.slm.generate_answer(question, context, max_tokens=300)
        
        return {
            'answer': answer,
            'retrieved_courses': retrieved_courses,
            'context': context
        }
    
    # ========== VALIDATION TESTS ==========
    
    def test_ai_courses(self):
        """Test 1: AI/ML courses"""
        print("=" * 60)
        print("TEST 1: AI and Machine Learning Courses")
        print("=" * 60)
        print()
        
        question = "What AI and machine learning courses are available?"
        
        # Ground truth from SQL
        print("📊 GROUND TRUTH (SQL):")
        sql_results = self.get_ai_courses_sql()
        sql_course_ids = set()
        
        for course in sql_results:
            print(f"  • {course['id']}: {course['name']}")
            sql_course_ids.add(course['id'])
        
        print(f"\n  Total: {len(sql_results)} courses\n")
        
        # RAG answer
        print("🤖 RAG SYSTEM ANSWER:")
        rag_result = self.query_rag(question)
        print(f"  Retrieved: {rag_result['retrieved_courses']}")
        print(f"\n  Answer: {rag_result['answer']}\n")
        
        # Compare
        retrieved_set = set(rag_result['retrieved_courses'])
        correct = retrieved_set.intersection(sql_course_ids)
        
        precision = len(correct) / len(retrieved_set) if retrieved_set else 0
        recall = len(correct) / len(sql_course_ids) if sql_course_ids else 0
        
        print("📈 VALIDATION METRICS:")
        print(f"  Precision: {precision:.2%} ({len(correct)}/{len(retrieved_set)})")
        print(f"  Recall: {recall:.2%} ({len(correct)}/{len(sql_course_ids)})")
        print(f"  Correctly retrieved: {correct}")
        print()
        
        return {
            'test': 'AI Courses',
            'precision': precision,
            'recall': recall,
            'sql_count': len(sql_course_ids),
            'rag_count': len(retrieved_set)
        }
    
    def test_database_courses(self):
        """Test 2: Database courses"""
        print("=" * 60)
        print("TEST 2: Database Courses")
        print("=" * 60)
        print()
        
        question = "Show me all database courses"
        
        # Ground truth
        print("📊 GROUND TRUTH (SQL):")
        sql_results = self.get_database_courses_sql()
        sql_course_ids = set()
        
        for course in sql_results:
            print(f"  • {course['id']}: {course['name']}")
            sql_course_ids.add(course['id'])
        
        print(f"\n  Total: {len(sql_results)} courses\n")
        
        # RAG answer
        print("🤖 RAG SYSTEM ANSWER:")
        rag_result = self.query_rag(question)
        print(f"  Retrieved: {rag_result['retrieved_courses']}")
        print(f"\n  Answer: {rag_result['answer']}\n")
        
        # Compare
        retrieved_set = set(rag_result['retrieved_courses'])
        correct = retrieved_set.intersection(sql_course_ids)
        
        precision = len(correct) / len(retrieved_set) if retrieved_set else 0
        recall = len(correct) / len(sql_course_ids) if sql_course_ids else 0
        
        print("📈 VALIDATION METRICS:")
        print(f"  Precision: {precision:.2%} ({len(correct)}/{len(retrieved_set)})")
        print(f"  Recall: {recall:.2%} ({len(correct)}/{len(sql_course_ids)})")
        print(f"  Correctly retrieved: {correct}")
        print()
        
        return {
            'test': 'Database Courses',
            'precision': precision,
            'recall': recall,
            'sql_count': len(sql_course_ids),
            'rag_count': len(retrieved_set)
        }
    
    def test_specific_course(self):
        """Test 3: Specific course details"""
        print("=" * 60)
        print("TEST 3: Specific Course Details")
        print("=" * 60)
        print()
        
        course_id = "IT079"
        question = f"Tell me about course {course_id}"
        
        # Ground truth
        print("📊 GROUND TRUTH (SQL):")
        sql_result = self.get_course_by_id_sql(course_id)
        
        if sql_result:
            print(f"  Course: {sql_result['id']}: {sql_result['name']}")
            print(f"  Vietnamese: {sql_result.get('name_vn', 'N/A')}")
            print(f"  Credits: {sql_result['total_credits']}")
            print(f"  Description: {sql_result.get('description', 'N/A')[:100]}...")
        print()
        
        # RAG answer
        print("🤖 RAG SYSTEM ANSWER:")
        rag_result = self.query_rag(question)
        print(f"  Answer: {rag_result['answer']}\n")
        
        # Validation: Check if course ID mentioned in answer
        answer_has_id = course_id in rag_result['answer']
        answer_has_name = (sql_result['name'].lower() in rag_result['answer'].lower()) if sql_result else False
        
        print("📈 VALIDATION:")
        print(f"  Course ID mentioned: {'✅' if answer_has_id else '❌'}")
        print(f"  Course name mentioned: {'✅' if answer_has_name else '❌'}")
        print()
        
        return {
            'test': 'Specific Course',
            'id_mentioned': answer_has_id,
            'name_mentioned': answer_has_name
        }
    
    def test_prerequisites(self):
        """Test 4: Prerequisites"""
        print("=" * 60)
        print("TEST 4: Course Prerequisites")
        print("=" * 60)
        print()
        
        course_id = "IT079"
        question = f"What are the prerequisites for {course_id}?"
        
        # Ground truth
        print("📊 GROUND TRUTH (SQL):")
        sql_results = self.get_prerequisites_sql(course_id)
        
        if sql_results:
            for prereq in sql_results:
                print(f"  • {prereq['id']}: {prereq['name']} (Type: {prereq['prerequisite_type']})")
        else:
            print("  No prerequisites found in database")
        
        print(f"\n  Total: {len(sql_results)} prerequisites\n")
        
        # RAG answer
        print("🤖 RAG SYSTEM ANSWER:")
        rag_result = self.query_rag(question)
        print(f"  Answer: {rag_result['answer']}\n")
        
        # Note: This is hard to validate automatically
        print("📈 NOTE: Prerequisites require SQL integration (future work)")
        print()
        
        return {
            'test': 'Prerequisites',
            'sql_count': len(sql_results),
            'note': 'Manual validation needed'
        }
    
    def test_program_courses(self):
        """Test 5: Program-specific courses"""
        print("=" * 60)
        print("TEST 5: Data Science Program Courses")
        print("=" * 60)
        print()
        
        question = "List all courses in the Data Science program"
        
        # Ground truth
        print("📊 GROUND TRUTH (SQL):")
        sql_results = self.get_courses_by_program_sql("Data Science")
        sql_course_ids = set()
        
        print(f"  Found {len(sql_results)} courses:")
        for i, course in enumerate(sql_results[:10], 1):  # Show first 10
            print(f"  {i}. {course['id']}: {course['name']}")
            sql_course_ids.add(course['id'])
        
        if len(sql_results) > 10:
            print(f"  ... and {len(sql_results) - 10} more")
        
        print(f"\n  Total: {len(sql_results)} courses\n")
        
        # RAG answer
        print("🤖 RAG SYSTEM ANSWER:")
        rag_result = self.query_rag(question, top_k=10)
        print(f"  Retrieved: {rag_result['retrieved_courses']}")
        print(f"\n  Answer: {rag_result['answer']}\n")
        
        # Compare
        retrieved_set = set(rag_result['retrieved_courses'])
        correct = retrieved_set.intersection(sql_course_ids)
        
        precision = len(correct) / len(retrieved_set) if retrieved_set else 0
        recall = len(correct) / len(sql_course_ids) if sql_course_ids else 0
        
        print("📈 VALIDATION METRICS:")
        print(f"  Precision: {precision:.2%} ({len(correct)}/{len(retrieved_set)})")
        print(f"  Recall: {recall:.2%} ({len(correct)}/{len(sql_course_ids)})")
        print()
        
        return {
            'test': 'Program Courses',
            'precision': precision,
            'recall': recall,
            'sql_count': len(sql_course_ids),
            'rag_count': len(retrieved_set)
        }
    
    # ========== RUN ALL TESTS ==========
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "=" * 60)
        print("🧪 RUNNING ALL VALIDATION TESTS")
        print("=" * 60)
        print()
        
        results = []
        
        # Run tests
        results.append(self.test_ai_courses())
        results.append(self.test_database_courses())
        results.append(self.test_specific_course())
        results.append(self.test_prerequisites())
        results.append(self.test_program_courses())
        
        # Summary
        print("=" * 60)
        print("📊 SUMMARY REPORT")
        print("=" * 60)
        print()
        
        avg_precision = sum(r.get('precision', 0) for r in results if 'precision' in r) / len([r for r in results if 'precision' in r])
        avg_recall = sum(r.get('recall', 0) for r in results if 'recall' in r) / len([r for r in results if 'recall' in r])
        
        print("Overall Metrics:")
        print(f"  Average Precision: {avg_precision:.2%}")
        print(f"  Average Recall: {avg_recall:.2%}")
        print()
        
        print("Test Results:")
        for result in results:
            print(f"  • {result['test']}:")
            if 'precision' in result:
                print(f"    - Precision: {result['precision']:.2%}")
                print(f"    - Recall: {result['recall']:.2%}")
            if 'note' in result:
                print(f"    - {result['note']}")
        
        print()
        print("=" * 60)
        print("✅ VALIDATION COMPLETE!")
        print("=" * 60)
        
        # Cleanup
        self.mysql.close()

# Run
if __name__ == "__main__":
    validator = ValidationTest()
    validator.run_all_tests()