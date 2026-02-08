# rag/slm_openrouter.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

class OpenRouterSLM:
    def __init__(self, model_name="meta-llama/llama-3.2-3b-instruct"):
        """
        OpenRouter FREE SLMs:
        
        - meta-llama/llama-3.2-3b-instruct (3B - BEST, FREE)
        - meta-llama/llama-3.2-1b-instruct (1B - Faster, FREE)
        - microsoft/phi-3-mini-128k-instruct (3.8B - FREE)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY') 
        )
        self.model_name = model_name
        
        # Extract size
        size = "3B" if "3b" in model_name else "1B" if "1b" in model_name else "3.8B"
        
        print(f"🤖 Using OpenRouter (FREE)")
        print(f"   Model: {model_name}")
        print(f"   Size: {size} (SLM)\n")
    
    def generate_answer(self, query, context, max_tokens=300):
        """Generate answer using OpenRouter"""
        
        system_prompt = """You are a helpful curriculum assistant for International University (IU).

    CRITICAL: When asked about PREREQUISITES:
    1. If context has "Calculus 1" and user asks "prerequisites for Calculus 2" 
    → Answer: "Calculus 1 (MA001) is the prerequisite"
    2. If context has "Introduction to X" and user asks "prerequisites for Advanced X"
    → Infer: Introduction is prerequisite for Advanced
    3. Use course numbering logic: Course 1 typically required before Course 2
    4. Only say "Not found" if context has NO related courses at all

    Rules:
    - Be concise and factual
    - Cite course IDs (e.g., MA001, IT079)
    - Use bullet points for lists
    - INFER logical prerequisites from course titles and numbers
    - Be professional

    Example:
    Context: "MA001: Calculus 1"
    Query: "Prerequisites for Calculus 2?"
    Good answer: "Calculus 1 (MA001) is the prerequisite for Calculus 2"
    Bad answer: "Not found" ❌"""

        user_prompt = f"""Curriculum Information:
{context}

Question: {query}

Provide a clear, concise answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {e}"

# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING OPENROUTER SLM (FREE)")
    print("=" * 60)
    print()
    
    # Test with LLaMA 3.2 3B
    slm = OpenRouterSLM("meta-llama/llama-3.2-3b-instruct")
    
    context = """Course IT097: Introduction to Artificial Intelligence
Credits: 4 (3 theory + 1 lab)
Vietnamese: Nhập môn trí tuệ nhân tạo
Description: AI fundamentals, search algorithms, machine learning basics

Course CSAI301: Machine Learning
Credits: 4 (3 theory + 1 lab)
Description: Supervised and unsupervised learning algorithms"""
    
    query = "What AI courses are available?"
    
    print("📝 Query:", query)
    print("\n🤖 Generating answer...\n")
    
    answer = slm.generate_answer(query, context)
    
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print()