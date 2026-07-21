# ============================================
# GROUNDED QA API - IMPROVED VERSION
# Better matching, more accurate answers
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
from difflib import SequenceMatcher
import uvicorn

# Create the app
app = FastAPI(title="Grounded QA API")

# Allow everyone to access (CORS)
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
# THE QA SYSTEM - IMPROVED MATCHING
# ============================================

class GroundedQA:
    def __init__(self):
        # Lower threshold to be more lenient
        self.threshold = 0.15  # Changed from 0.3 to 0.15
    
    def clean_text(self, text):
        """Clean text for better comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text):
        """Extract important words from text"""
        # Remove common words (stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'to',
                     'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                     'through', 'during', 'including', 'according', 'according'}
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return set(keywords)
    
    def calculate_similarity(self, text1, text2):
        """Calculate how similar two texts are (0 to 1)"""
        clean1 = self.clean_text(text1)
        clean2 = self.clean_text(text2)
        
        # Extract keywords
        keywords1 = self.extract_keywords(clean1)
        keywords2 = self.extract_keywords(clean2)
        
        # If either is empty, no similarity
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate keyword overlap
        common_keywords = keywords1.intersection(keywords2)
        total_keywords = keywords1.union(keywords2)
        
        # Jaccard similarity
        keyword_similarity = len(common_keywords) / len(total_keywords) if total_keywords else 0.0
        
        # Character sequence similarity
        char_similarity = SequenceMatcher(None, clean1, clean2).ratio()
        
        # Word overlap ratio (how many question words appear in chunk)
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        word_overlap = len(words1.intersection(words2)) / len(words1) if words1 else 0.0
        
        # Combine scores with weights
        # Weighted more towards keyword matching
        final_score = (keyword_similarity * 0.5) + (char_similarity * 0.2) + (word_overlap * 0.3)
        
        return round(final_score, 3)
    
    def find_best_answer(self, question, chunks):
        """Find the best answer from provided chunks"""
        
        if not chunks:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.0,
                "answerable": False
            }
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            similarity = self.calculate_similarity(question, chunk.text)
            scored_chunks.append({
                "chunk": chunk,
                "score": similarity,
                "text": chunk.text
            })
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Get the best match
        best = scored_chunks[0]
        best_score = best["score"]
        
        # Check if good enough to answer
        if best_score < self.threshold:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": min(best_score, 0.3),
                "answerable": False
            }
        
        # Calculate confidence
        confidence = round(min(best_score * 1.2, 0.95), 3)
        
        return {
            "answer": best["chunk"].text,
            "citations": [best["chunk"].chunk_id],
            "confidence": confidence,
            "answerable": True
        }

# Create the QA system
qa_system = GroundedQA()

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Grounded QA API is running!",
        "version": "1.0.0"
    }

@app.post("/grounded-answer", response_model=QAResponse)
async def grounded_answer(request: QARequest):
    """Main endpoint: Answer questions from provided chunks"""
    try:
        # Validate question
        if not request.question or not request.question.strip():
            return QAResponse(
                answer="I don't know",
                citations=[],
                confidence=0.0,
                answerable=False
            )
        
        # Get the answer
        result = qa_system.find_best_answer(request.question, request.chunks)
        
        # Ensure "I don't know" format
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

# ============================================
# RUN THE SERVER
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GROUNDED QA API - IMPROVED")
    print("="*60)
    print("📡 Server: http://localhost:8000")
    print("🔗 Endpoint: http://localhost:8000/grounded-answer")
    print("🔍 Health: http://localhost:8000/health")
    print("="*60)
    print("\n✅ Ready to answer questions!")
    print("Press CTRL+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)