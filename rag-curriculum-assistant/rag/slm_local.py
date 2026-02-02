# rag/slm_local.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalSLM:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialize local Small Language Model
        
        For 4GB VRAM:
        - Qwen/Qwen2.5-1.5B-Instruct (RECOMMENDED, ~1.5GB VRAM)
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1GB VRAM)
        
        For 8GB+ VRAM:
        - Qwen/Qwen2.5-3B-Instruct (~3GB VRAM)
        - microsoft/Phi-3-mini-4k-instruct (~3.5GB VRAM)
        """
        print(f"🔧 Loading SLM: {model_name}")
        print("📥 First run will download model (~3GB)...")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"💻 Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model - simple approach for 1.5B
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # FP16 for speed
                device_map="auto",
                trust_remote_code=True
            )
            print(f"✅ Model loaded on GPU (FP16)\n")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            print(f"✅ Model loaded on CPU (FP32)\n")
    
    def generate_answer(self, query, context, max_tokens=300):
        """Generate answer using local SLM"""
        
        # Build prompt (Qwen format)
        system_prompt = """You are a helpful curriculum assistant for International University (IU).
Provide direct, concise answers based only on the given curriculum information.

Rules:
- Be concise and factual
- Cite course IDs (e.g., IT079, CSAI301)
- Use bullet points for lists
- If information is not provided, say "Not found in curriculum"
- Be professional"""

        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Curriculum Information:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after assistant tag)
        if "<|im_start|>assistant" in response:
            answer = response.split("<|im_start|>assistant")[-1].strip()
        else:
            answer = response.split("Question:")[-1].strip() if "Question:" in response else response
        
        return answer

# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING LOCAL SLM (Qwen 1.5B)")
    print("=" * 60)
    print()
    
    # Initialize
    slm = LocalSLM("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Test query
    context = """Course IT079: Principles of Database Management
Credits: 4 (3 theory + 1 lab)
Description: Introduction to database concepts, SQL, normalization

Course CSAI301: Machine Learning
Credits: 4 (3 theory + 1 lab)
Description: Supervised and unsupervised learning algorithms"""
    
    query = "What database courses are available?"
    
    print("📝 Query:", query)
    print("\n🤖 Generating answer...\n")
    
    answer = slm.generate_answer(query, context)
    
    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print()