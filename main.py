# ============================================
# GROUNDED QA API WITH GEMINI AI
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re
import uvicorn

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Gemini not installed. Install with: pip install google-generativeai")

# Create the app
app = FastAPI(title="Grounded QA API with Gemini")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DATA MODELS
# ============================================

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]

class QAResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool

# ============================================
# GEMINI QA SYSTEM
# ============================================

class GroundedQA:
    def __init__(self):
        self.threshold = 0.15
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini AI initialized successfully!")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                self.model = None
        else:
            self.model = None
            if not os.getenv("GEMINI_API_KEY"):
                print("⚠️ GEMINI_API_KEY not set. Using fallback matching.")
    
    def clean_text(self, text):
        """Clean text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity (fallback when Gemini not available)"""
        clean1 = self.clean_text(text1)
        clean2 = self.clean_text(text2)
        
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common = words1.intersection(words2)
        total = words1.union(words2)
        
        return len(common) / len(total) if total else 0.0
    
    def find_best_answer(self, question, chunks):
        """Find answer using Gemini or fallback"""
        
        if not chunks:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.0,
                "answerable": False
            }
        
        # Try using Gemini first
        if self.model:
            try:
                # Build context from chunks
                context = "\n".join([f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks])
                
                # Create prompt that forces grounded answers
                prompt = f"""
You are a strict QA assistant. Answer the question ONLY using information from the provided context chunks.
If the answer is not in the context, say "I don't know".

Context chunks:
{context}

Question: {question}

Rules:
1. Only use information from the context
2. Cite the chunk ID(s) you used
3. If unsure, say "I don't know"
4. Be concise

Answer:"""
                
                # Get response from Gemini
                response = self.model.generate_content(prompt)
                answer = response.text.strip()
                
                # Extract citations from the answer
                citations = []
                # Look for chunk IDs like [C1], [C2] etc.
                chunk_ids = re.findall(r'\[(C\d+)\]', answer)
                citations = list(set(chunk_ids))
                
                # Verify citations exist in provided chunks
                valid_ids = {chunk.chunk_id for chunk in chunks}
                citations = [c for c in citations if c in valid_ids]
                
                # Clean answer: remove citation markers if present
                clean_answer = re.sub(r'\[C\d+\]', '', answer).strip()
                
                # Check if it's "I don't know"
                if "I don't know" in clean_answer or clean_answer.lower().strip() == "i don't know":
                    return {
                        "answer": "I don't know",
                        "citations": [],
                        "confidence": 0.2,
                        "answerable": False
                    }
                
                # Calculate confidence (Gemini gives good answers)
                confidence = 0.85 if citations else 0.7
                
                return {
                    "answer": clean_answer,
                    "citations": citations if citations else [chunks[0].chunk_id],
                    "confidence": round(confidence, 3),
                    "answerable": True
                }
                
            except Exception as e:
                print(f"⚠️ Gemini error: {e}. Using fallback.")
                # Fall through to regular matching
        
        # ============================================
        # FALLBACK: Simple matching (without Gemini)
        # ============================================
        
        scored = []
        for chunk in chunks:
            similarity = self.calculate_similarity(question, chunk.text)
            scored.append({
                "chunk": chunk,
                "score": similarity
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        
        if best["score"] < self.threshold:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": min(best["score"], 0.3),
                "answerable": False
            }
        
        return {
            "answer": best["chunk"].text,
            "citations": [best["chunk"].chunk_id],
            "confidence": round(min(best["score"] * 1.2, 0.95), 3),
            "answerable": True
        }

# ============================================
# CREATE QA SYSTEM
# ============================================

qa_system = GroundedQA()

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Grounded QA API with Gemini",
        "gemini_available": GEMINI_AVAILABLE and qa_system.model is not None,
        "version": "2.0.0"
    }

@app.post("/grounded-answer", response_model=QAResponse)
async def grounded_answer(request: QARequest):
    try:
        if not request.question or not request.question.strip():
            return QAResponse(
                answer="I don't know",
                citations=[],
                confidence=0.0,
                answerable=False
            )
        
        result = qa_system.find_best_answer(request.question, request.chunks)
        
        if not result["answerable"]:
            result["answer"] = "I don't know"
            result["citations"] = []
            result["confidence"] = min(result["confidence"], 0.3)
        
        return QAResponse(**result)
    
    except Exception as e:
        print(f"Error: {e}")
        return QAResponse(
            answer="I don't know",
            citations=[],
            confidence=0.0,
            answerable=False
        )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GROUNDED QA API WITH GEMINI")
    print("="*60)
    print("📡 Server: http://localhost:8000")
    print("🔗 Endpoint: http://localhost:8000/grounded-answer")
    print("="*60)
    print("\n✅ Ready! Press CTRL+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)