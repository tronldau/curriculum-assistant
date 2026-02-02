# data/mysql_connector.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from mysql.connector import Error
class Config:
    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "Bamboo0610@"
    MYSQL_DATABASE = "digit_curriculum"

class MySQLConnector:
    def __init__(self):
        try:
            self.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DATABASE
            )
            if self.connection.is_connected():
                print("✅ Connected to MySQL")
        except Error as e:
            print(f"❌ MySQL Connection Error: {e}")
            self.connection = None
    
    def execute_query(self, query):
        """Execute SELECT query and return results"""
        if not self.connection:
            print("❌ No database connection")
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"❌ Query Error: {e}")
            return []
    
    def get_all_courses(self):
        """Get all courses with basic info"""
        query = """
        SELECT 
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory,
            c.credit_lab,
            p.name as program_name
        FROM course c
        LEFT JOIN course_program cp ON c.id = cp.course_id
        LEFT JOIN program p ON cp.program_id = p.id
        """
        return self.execute_query(query)
    
    def get_prerequisites(self, course_id):
        """Get full prerequisites information for a course"""
        query = f"""
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.description,
            c.credit_theory,
            c.credit_lab,
            c.credit_theory + c.credit_lab as total_credits,
            c.course_level_id,
            ccr.relationship_id
        FROM course_course_relationship ccr
        JOIN course c ON ccr.course_id2 = c.id
        WHERE ccr.course_id1 = '{course_id}'
        ORDER BY c.id
        """
        return self.execute_query(query)

    def get_dependent_courses(self, course_id):
        """Get courses that require this course as prerequisite"""
        query = f"""
        SELECT DISTINCT
            c.id,
            c.name,
            c.name_vn,
            c.credit_theory + c.credit_lab as total_credits
        FROM course_course_relationship ccr
        JOIN course c ON ccr.course_id1 = c.id
        WHERE ccr.course_id2 = '{course_id}'
        ORDER BY c.id
        """
        return self.execute_query(query)

    def find_course_by_name(self, course_name):
        """Find course ID by partial name match"""
        query = f"""
        SELECT id, name, name_vn
        FROM course
        WHERE 
            name LIKE '%{course_name}%'
            OR name_vn LIKE '%{course_name}%'
            OR id LIKE '%{course_name}%'
        LIMIT 5
        """
        return self.execute_query(query)
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✅ MySQL connection closed")

# Test
if __name__ == "__main__":
    print("Testing MySQL Connection...\n")
    
    db = MySQLConnector()
    
    if db.connection:
        courses = db.get_all_courses()
        print(f"📊 Total courses: {len(courses)}")
        
        if courses:
            print("\n📖 Sample course:")
            print(f"   ID: {courses[0]['id']}")
            print(f"   Name: {courses[0]['name']}")
        
        db.close()
    else:
        print("❌ Failed to connect to database")