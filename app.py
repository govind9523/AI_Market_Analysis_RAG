"""
Updated FastAPI app with integrated frontend
Add this to your app.py to serve the web interface
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles  
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Market Analyst",
    description="RAG system optimized for 8GB systems with Web Interface",
    version="1.0.0",
    docs_url="/api/docs",  # Move API docs to /api/docs
    redoc_url="/api/redoc"  # Move ReDoc to /api/redoc
)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    tool: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    tool_used: str
    confidence: float
    metadata: Dict[str, Any]

# Global variables (will be initialized on startup)
embedding_model = None
vector_store = None
llm_model = None
tools = {}
router = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global embedding_model, vector_store, llm_model, tools, router
    
    logger.info("🚀 Starting AI Market Analyst RAG System...")
    logger.info("💾 System optimized for 8GB RAM constraints")
    
    try:
        # Import modules
        from models.embeddings import EmbeddingModel
        from models.vector_store import VectorStore
        from models.llm import LLMModel
        from tools.qa_tool import QATool
        from tools.summarizer import SummarizerTool
        from tools.extractor import ExtractorTool
        from utils.document_loader import DocumentLoader
        from utils.router import AutonomousRouter
        
        # 1. Initialize embedding model
        logger.info("📊 Loading embedding model (all-MiniLM-L6-v2)...")
        embedding_model = EmbeddingModel()
        
        # 2. Load and process document
        logger.info("📄 Processing Innovate Inc. document...")
        doc_loader = DocumentLoader()
        
        doc_path = "data/innovate_inc_report.txt"
        if not Path(doc_path).exists():
            logger.error(f"❌ Document not found: {doc_path}")
            raise FileNotFoundError(f"Missing: {doc_path}")
        
        chunks = doc_loader.load_and_chunk(doc_path)
        
        # 3. Create vector store
        logger.info("🔍 Creating FAISS vector store...")
        vector_store = VectorStore(embedding_model)
        await vector_store.add_documents(chunks)
        
        # 4. Initialize LLM
        logger.info("🤖 Loading local Llama 3 8B GGUF model...")
        llm_model = LLMModel()
        
        # Test LLM connection
        try:
            test_response = await llm_model.generate("Hello", options={"num_predict": 5})
            logger.info("✅ LLM loaded and tested successfully")
        except Exception as e:
            logger.warning(f"⚠️ LLM loading or generation failed: {e}")
            logger.warning("Make sure the GGUF model file is available at the path specified in llm.py")
        
        # 5. Initialize tools
        logger.info("🛠️ Setting up RAG tools...")
        tools = {
            "qa": QATool(vector_store, llm_model),
            "summarize": SummarizerTool(vector_store, llm_model), 
            "extract": ExtractorTool(vector_store, llm_model)
        }
        
        # 6. Initialize router
        logger.info("🎯 Setting up autonomous routing...")
        router = AutonomousRouter(llm_model)
        
        logger.info("✅ System initialization complete!")
        logger.info("🌐 Web Interface: http://localhost:8000")
        logger.info("📖 API Docs: http://localhost:8000/api/docs")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# FRONTEND ROUTES
@app.get("/")
async def serve_frontend():
    """Serve the web interface"""
    # Check if frontend.html exists
    frontend_path = Path("frontend.html")
    if frontend_path.exists():
        return FileResponse("frontend.html")
    else:
        # Return a simple message if frontend not found
        return {
            "message": "AI Market Analyst RAG System",
            "status": "API is running",
            "web_interface": "Create frontend.html file to enable web interface",
            "api_docs": "http://localhost:8000/api/docs"
        }

# API ROUTES (with /api prefix to avoid conflicts)
@app.get("/api/status")
@app.get("/health")  # Keep backwards compatibility
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if all([embedding_model, vector_store, llm_model, len(tools) == 3]) else "initializing",
        "models_loaded": all([
            embedding_model is not None,
            vector_store is not None, 
            llm_model is not None,
            len(tools) == 3
        ]),
        "components": {
            "embedding_model": embedding_model is not None,
            "vector_store": vector_store is not None and vector_store.size > 0,
            "llm_model": llm_model is not None,
            "tools": len(tools) if tools else 0,
            "document_chunks": vector_store.size if vector_store else 0
        },
        "system_info": {
            "optimization": "8GB RAM optimized",
            "embedding_model": "all-MiniLM-L6-v2 (384 dim)",
            "llm_model": "Meta-Llama-3-8B-Instruct (local GGUF)",
            "vector_store": "FAISS (CPU)"
        }
    }

@app.post("/query", response_model=QueryResponse)
@app.post("/api/query", response_model=QueryResponse)  # Alternative API path
async def query_system(request: QueryRequest):
    """Main query endpoint with autonomous routing"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if system is ready
        if not all([embedding_model, vector_store, tools]):
            raise HTTPException(
                status_code=503, 
                detail="System not fully initialized. Check /health endpoint."
            )
        
        # Determine which tool to use
        if request.tool:
            # Explicit tool selection
            tool_name = request.tool.lower()
            if tool_name not in tools:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Tool '{tool_name}' not found. Available: {list(tools.keys())}"
                )
            selected_tool = tool_name
            confidence = 1.0
        else:
            # Autonomous routing
            try:
                selected_tool, confidence = await router.route_query(query)
            except Exception as e:
                logger.warning(f"Routing failed: {e}, defaulting to Q&A")
                selected_tool = "qa"
                confidence = 0.5
        
        # Execute the tool
        tool = tools[selected_tool]
        start_time = time.time()
        response = await tool.execute(query)
        execution_time = time.time() - start_time
        
        # Add timing and routing info
        if "metadata" not in response:
            response["metadata"] = {}
        
        response["metadata"].update({
            "execution_time": round(execution_time, 2),
            "selected_tool": selected_tool,
            "routing_method": "explicit" if request.tool else "autonomous",
            "query_length": len(query),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return QueryResponse(
            response=response["answer"],
            tool_used=selected_tool,
            confidence=confidence,
            metadata=response.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools")
@app.get("/tools")  # Keep backwards compatibility
async def list_tools():
    """List available tools"""
    return {
        "available_tools": {
            "qa": {
                "name": "Question & Answer",
                "description": "Answer specific questions about the market report",
                "example": "What is the market share of FutureFlow?",
                "icon": "❓"
            },
            "summarize": {
                "name": "Summarizer", 
                "description": "Provide concise summary of the entire report",
                "example": "Summarize the key findings",
                "icon": "📋"
            },
            "extract": {
                "name": "Data Extractor",
                "description": "Extract structured data in JSON format",
                "example": "Extract key financial data",
                "icon": "📊"
            }
        },
        "autonomous_routing": {
            "enabled": router is not None,
            "description": "AI automatically selects the best tool for your query"
        }
    }

@app.get("/api/demo")
@app.get("/demo")  # Keep backwards compatibility
async def demo_queries():
    """Demo queries to test the system"""
    return {
        "demo_queries": [
            {
                "query": "What is the market share of FutureFlow?",
                "tool": "qa",
                "expected": "Specific answer about FutureFlow's 15% market share",
                "category": "Question & Answer"
            },
            {
                "query": "Summarize the competitive landscape",
                "tool": "summarize", 
                "expected": "Summary of competitors and market positions",
                "category": "Summarization"
            },
            {
                "query": "Extract key financial data",
                "tool": "extract",
                "expected": "JSON with market size, CAGR, projections",
                "category": "Data Extraction"
            },
            {
                "query": "What are the main threats to Innovate Inc?",
                "tool": None,
                "expected": "Autonomous routing to appropriate tool",
                "category": "Autonomous Routing"
            },
            {
                "query": "How much is the projected market size by 2030?",
                "tool": None,
                "expected": "AI routes to Q&A tool for specific question",
                "category": "Autonomous Routing"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Market Analyst RAG System with Web Interface")
    logger.info("🌐 Web Interface: http://localhost:8000")
    logger.info("📖 API Documentation: http://localhost:8000/api/docs")
    logger.info("🎮 Demo Queries: http://localhost:8000/demo")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        reload=False,
        workers=1
    )