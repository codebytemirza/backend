from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
import sqlite3
import os
from pathlib import Path
from services.docx_service import DocxPineconeIngestionAgent
from services.audio_service import AudioPineconeService
from services.play_audio import AudioService  # Add this import
from services.rag_workflow import RAGService
import base64
from datetime import datetime
from typing import Optional
import uuid
import json
from langchain_core.messages import HumanMessage
from services.auth_service import (
    register_user,
    verify_user,
    login_user,
    reset_password_request,
    reset_password,
    admin_login,
    create_admin,
    change_admin_password,
    get_all_admins,
    toggle_admin_status,
    resend_verification_code
)

app = FastAPI()


# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database paths relative to the script location
DB_PATH = os.path.join(BASE_DIR, "prompt.db")
UPLOADS_DB_PATH = os.path.join(BASE_DIR, "uploads.db")

# Initialize paths
env_path = Path(__file__).parent / ".env"
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

# Initialize services
docx_agent = DocxPineconeIngestionAgent(str(env_path))
audio_service = AudioPineconeService(str(env_path))
rag_service = RAGService(str(env_path))
tts_service = AudioService()  # Add this line

# In-memory storage for background task results (in production, use Redis or database)
background_results = {}

# Models
class PromptModel(BaseModel):
    prompt: str

class QueryModel(BaseModel):
    query: str
    thread_id: Optional[str] = None
    use_web_search: bool = True

class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "coral"
    instructions: Optional[str] = None
    response_format: str = "mp3"
    
class RegisterRequest(BaseModel):
    email: str
    password: str

class VerifyRequest(BaseModel):
    email: str
    code: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ResetPasswordRequestModel(BaseModel):
    email: str

class ResetPasswordModel(BaseModel):
    email: str
    code: str
    new_password: str

class AdminLoginRequest(BaseModel):
    username: str  # Can be username or email
    password: str

class CreateAdminRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "admin"  # 'admin' or 'moderator'

class ChangeAdminPasswordRequest(BaseModel):
    current_password: str
    new_password: str

class AdminActionRequest(BaseModel):
    admin_id: int

# Database initialization functions
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def init_uploads_db():
    conn = sqlite3.connect(UPLOADS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_type TEXT NOT NULL,
            status TEXT NOT NULL,
            chunks INTEGER,
            duration REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize databases
init_db()
init_uploads_db()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Existing prompt endpoints
@app.post("/chngPrompt")
async def chng_prompt(data: PromptModel):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Delete existing prompt
    cursor.execute("DELETE FROM prompt")
    # Insert new prompt
    cursor.execute("INSERT INTO prompt (text) VALUES (?)", (data.prompt,))
    conn.commit()
    conn.close()
    return {"status": "success", "prompt": data.prompt}

@app.get("/getPrompt")
async def get_prompt():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM prompt LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return {"prompt": result[0] if result else ""}

# =============================================================================
# RAG CHAT ENDPOINTS
# =============================================================================

@app.post("/chat")
async def chat_endpoint(data: QueryModel):
    """
    Main chat endpoint that processes user queries through the RAG workflow.
    """
    try:
        # Generate thread ID if not provided
        if not data.thread_id:
            data.thread_id = f"chat_{uuid.uuid4().hex[:8]}"
        
        # Process the query through RAG workflow
        result = await rag_service.answer_query(
            query=data.query,
            thread_id=data.thread_id
        )
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "answer": result["answer"],
                    "query": result["query"],
                    "thread_id": data.thread_id,
                    "metadata": result["metadata"],
                    "timestamp": result["processed_at"]
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Failed to process query: {result['error']}",
                    "query": result["query"]
                }
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/chat-stream")
async def chat_stream_endpoint(data: QueryModel):
    """
    Streaming chat endpoint for real-time responses.
    """
    async def generate_response():
        try:
            # Generate thread ID if not provided
            if not data.thread_id:
                data.thread_id = f"stream_{uuid.uuid4().hex[:8]}"
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...', 'stage': 'start'})}\n\n"
            
            # Process the query
            result = await rag_service.answer_query(
                query=data.query,
                thread_id=data.thread_id
            )
            
            if result["success"]:
                # Send the complete answer
                yield f"data: {json.dumps({'type': 'answer', 'content': result['answer'], 'metadata': result['metadata']})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'thread_id': data.thread_id})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/chat-history/{thread_id}")
async def get_chat_history(thread_id: str, limit: int = 20):
    """
    Get chat history for a specific thread.
    """
    try:
        # This would integrate with your workflow's memory system
        # For now, return a placeholder response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "thread_id": thread_id,
                "messages": [],  # Would fetch from workflow memory
                "message_count": 0
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/query-vector-db")
async def query_vector_db_direct(data: QueryModel):
    """
    Direct query to vector database without full RAG workflow.
    Useful for testing or simple searches.
    """
    try:
        # Access the vector store from the RAG service
        vector_store = rag_service.workflow.vector_store
        
        if not vector_store:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Vector store not available"
                }
            )
        
        # Perform similarity search
        results = vector_store.similarity_search_with_score(
            data.query,
            k=5
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "query": data.query,
                "results": formatted_results,
                "result_count": len(formatted_results)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/analyze-query")
async def analyze_query_endpoint(data: QueryModel):
    """
    Analyze and decompose a user query without executing full RAG.
    Useful for understanding query complexity.
    """
    try:
        # Initialize workflow components
        workflow = rag_service.workflow
        
        # Create minimal state for analysis
        state = {
            "original_query": data.query,
            "decomposed_queries": [],
            "custom_prompt": workflow.get_custom_prompt()
        }
        
        # Run decomposition
        analyzed_state = workflow.decompose_query(state)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "original_query": data.query,
                "decomposed_queries": analyzed_state["decomposed_queries"],
                "complexity": "complex" if len(analyzed_state["decomposed_queries"]) > 1 else "simple",
                "sub_query_count": len(analyzed_state["decomposed_queries"]),
                "custom_prompt_available": len(state["custom_prompt"]) > 0
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/test-workflow-step")
async def test_workflow_step(step_name: str, data: dict):
    """
    Test individual workflow steps for debugging purposes.
    """
    try:
        workflow = rag_service.workflow
        
        # Create a test state
        test_state = {
            "original_query": data.get("query", "test query"),
            "decomposed_queries": data.get("decomposed_queries", []),
            "retrieved_contexts": data.get("retrieved_contexts", []),
            "ranked_contexts": data.get("ranked_contexts", []),
            "filtered_contexts": data.get("filtered_contexts", []),
            "web_search_results": data.get("web_search_results"),
            "final_context": data.get("final_context", ""),
            "is_context_sufficient": data.get("is_context_sufficient", False),
            "custom_prompt": data.get("custom_prompt", ""),
            "final_answer": data.get("final_answer", ""),
            "messages": []
        }
        
        # Execute specific step
        if step_name == "decompose_query":
            result_state = workflow.decompose_query(test_state)
        elif step_name == "retrieve_contexts":
            result_state = workflow.retrieve_contexts(test_state)
        elif step_name == "rank_contexts":
            result_state = workflow.rank_contexts(test_state)
        elif step_name == "filter_contexts":
            result_state = workflow.filter_contexts(test_state)
        elif step_name == "decide_sufficiency":
            result_state = workflow.decide_sufficiency(test_state)
        elif step_name == "web_search":
            result_state = workflow.web_search(test_state)
        else:
            raise ValueError(f"Unknown step: {step_name}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "step": step_name,
                "input_state": test_state,
                "output_state": result_state
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "step": step_name
            }
        )

@app.get("/workflow-status")
async def get_workflow_status():
    """
    Get the current status and configuration of the RAG workflow.
    """
    try:
        workflow = rag_service.workflow
        
        # Check component status
        status = {
            "workflow_initialized": workflow is not None,
            "vector_store_connected": workflow.vector_store is not None,
            "custom_prompt_available": len(workflow.get_custom_prompt()) > 0,
            "llm_models": {
                "decomposer": "gpt-4o-mini",
                "ranker": "gpt-4o-mini", 
                "filter": "gpt-4o-mini",
                "decision": "gpt-4o-mini",
                "answer": "gpt-4o"
            },
            "tools_available": {
                "tavily_search": workflow.tavily_api_key is not None,
                "vector_search": True
            },
            "database_paths": {
                "prompt_db": workflow.db_path,
                "checkpoint_db": workflow.checkpoint_db
            }
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "workflow_status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# Background task for processing large queries
async def process_large_query_background(query: str, thread_id: str, result_storage: dict):
    """Background task for processing computationally intensive queries."""
    try:
        result = await rag_service.answer_query(query, thread_id)
        result_storage[thread_id] = result
    except Exception as e:
        result_storage[thread_id] = {
            "success": False,
            "error": str(e),
            "query": query
        }

@app.post("/chat-async")
async def chat_async_endpoint(data: QueryModel, background_tasks: BackgroundTasks):
    """
    Process query asynchronously and return task ID for polling results.
    """
    try:
        # Generate thread ID if not provided
        if not data.thread_id:
            data.thread_id = f"async_{uuid.uuid4().hex[:8]}"
        
        # Add background task
        background_tasks.add_task(
            process_large_query_background,
            data.query,
            data.thread_id,
            background_results
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "thread_id": data.thread_id,
                "query": data.query,
                "message": "Query is being processed in background. Use /chat-result/{thread_id} to check status."
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/chat-result/{thread_id}")
async def get_chat_result(thread_id: str):
    """
    Get the result of an asynchronously processed query.
    """
    if thread_id not in background_results:
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "thread_id": thread_id,
                "message": "Query is still being processed. Please try again later."
            }
        )
    
    result = background_results[thread_id]
    
    # Clean up result from memory after retrieval
    del background_results[thread_id]
    
    if result["success"]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "thread_id": thread_id,
                "answer": result["answer"],
                "query": result["query"],
                "metadata": result["metadata"],
                "timestamp": result["processed_at"]
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "thread_id": thread_id,
                "message": result["error"],
                "query": result["query"]
            }
        )

# Health check for RAG system
@app.get("/rag-health")
async def rag_health_check():
    """
    Comprehensive health check for the RAG system.
    """
    try:
        health_status = {
            "rag_service": "healthy",
            "vector_db": "unknown",
            "llm_models": "unknown",
            "custom_prompt": "unknown",
            "web_search": "unknown",
            "overall": "healthy"
        }
        
        # Test vector database
        try:
            if rag_service.workflow.vector_store:
                test_results = rag_service.workflow.vector_store.similarity_search("test", k=1)
                health_status["vector_db"] = "healthy" if test_results else "no_data"
            else:
                health_status["vector_db"] = "not_connected"
        except Exception:
            health_status["vector_db"] = "unhealthy"
            health_status["overall"] = "degraded"
        
        # Test LLM models
        try:
            test_response = rag_service.workflow.decomposer_llm.invoke([HumanMessage(content="test")])
            health_status["llm_models"] = "healthy" if test_response else "unhealthy"
        except Exception:
            health_status["llm_models"] = "unhealthy"
            health_status["overall"] = "degraded"
        
        # Check custom prompt
        try:
            custom_prompt = rag_service.workflow.get_custom_prompt()
            health_status["custom_prompt"] = "available" if custom_prompt else "empty"
        except Exception:
            health_status["custom_prompt"] = "unavailable"
        
        # Test web search
        try:
            if rag_service.workflow.tavily_api_key:
                health_status["web_search"] = "available"
            else:
                health_status["web_search"] = "not_configured"
        except Exception:
            health_status["web_search"] = "unavailable"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": health_status["overall"],
                "components": health_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# =============================================================================
# EXISTING FILE UPLOAD ENDPOINTS
# =============================================================================

@app.post("/upload-docx")
async def upload_docx(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.docx'):
            raise HTTPException(status_code=400, detail="Only DOCX files are allowed")

        # Save file to data directory
        file_path = data_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the file
        result = docx_agent.process_docx_file(str(file_path))

        # Save to uploads database
        conn = sqlite3.connect(UPLOADS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO uploads (filename, file_type, upload_type, status, chunks)
            VALUES (?, ?, ?, ?, ?)
        """, (
            file.filename,
            'docx',
            'upload',
            'success' if result["success"] else 'failed',
            result.get("total_chunks", 0)
        ))
        conn.commit()
        conn.close()

        if not result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process file: {result['error']}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
                "details": {
                    "filename": file.filename,
                    "chunks_created": result["total_chunks"],
                    "tokens_processed": result["total_tokens"]
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), language: str = "en"):
    try:
        # Validate file type
        if not file.filename.endswith('.mp3'):
            raise HTTPException(status_code=400, detail="Only MP3 files are allowed")

        # Save file to data directory
        file_path = data_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the audio file
        result = audio_service.process_mp3_file(
            str(file_path),
            language=language,
            enhance_with_ai=True
        )

        # Save to uploads database
        conn = sqlite3.connect(UPLOADS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO uploads (filename, file_type, upload_type, status, chunks, duration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file.filename,
            'mp3',
            'upload',
            'success' if result["success"] else 'failed',
            result.get("total_chunks", 0),
            result.get("audio_duration", 0)
        ))
        conn.commit()
        conn.close()

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process audio: {result.get('error', 'Unknown error')}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Audio processed successfully",
                "details": {
                    "filename": file.filename,
                    "duration": result.get("audio_duration"),
                    "chunks_created": result.get("total_chunks"),
                    "tokens_processed": result.get("total_tokens"),
                    "language": result.get("language", "en")
                }
            }

        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/upload-recorded-audio")
async def upload_recorded_audio(
    audio_data: dict,
    language: str = "en"
):
    try:
        # Extract base64 audio data
        base64_audio = audio_data.get("audioData", "").split(",")[-1]
        filename = audio_data.get("filename", f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")

        # Process the base64 audio data
        result = audio_service.process_mp3_from_base64(
            base64_audio,
            filename=filename,
            language=language,
            enhance_with_ai=True
        )

        # Save to uploads database
        conn = sqlite3.connect(UPLOADS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO uploads (filename, file_type, upload_type, status, chunks, duration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            filename,
            'mp3',
            'recording',
            'success' if result["success"] else 'failed',
            result.get("total_chunks", 0),
            result.get("audio_duration", 0)
        ))
        conn.commit()
        conn.close()

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process audio: {result.get('error', 'Unknown error')}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Recorded audio processed successfully",
                "details": {
                    "filename": filename,
                    "duration": result.get("audio_duration"),
                    "chunks_created": result.get("total_chunks"),
                    "tokens_processed": result.get("total_tokens"),
                    "language": result.get("language", "en")
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/get-upload-history")
async def get_upload_history(limit: int = 20):
    conn = sqlite3.connect(UPLOADS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, file_type, upload_type, status, chunks, duration, created_at
        FROM uploads ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    results = cursor.fetchall()
    conn.close()
    
    return {
        "uploads": [
            {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "upload_type": row[3],
                "status": row[4],
                "chunks": row[5],
                "duration": row[6],
                "created_at": row[7]
            }
            for row in results
        ]
    }

@app.delete("/delete-upload/{upload_id}")
async def delete_upload(upload_id: int):
    try:
        conn = sqlite3.connect(UPLOADS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Upload record deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add these new endpoints before the existing upload endpoints
@app.post("/text-to-speech")
async def text_to_speech(data: TextToSpeechRequest):
    """Generate audio file from text."""
    try:
        result = await tts_service.generate_audio(
            text=data.text,
            voice=data.voice,
            instructions=data.instructions,
            response_format=data.response_format
        )
        
        if result["status"] == "success":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "file_path": result["file_path"],
                    "message": result["message"]
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-text-to-speech")
async def stream_text_to_speech(data: TextToSpeechRequest):
    """Stream audio data to client."""
    try:
        result = await tts_service.stream_audio(
            text=data.text,
            voice=data.voice,
            instructions=data.instructions,
            response_format="mp3"  # Using mp3 for better browser compatibility
        )
        
        if result["status"] == "success":
            return Response(
                content=result["audio_data"],
                media_type=result["content_type"],
                headers={
                    "Content-Disposition": "attachment;filename=speech.mp3"
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts-voices")
async def get_tts_voices():
    """Get list of available TTS voices."""
    try:
        voices = tts_service.get_supported_voices()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "voices": voices
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts-formats")
async def get_tts_formats():
    """Get list of supported audio formats."""
    try:
        formats = tts_service.get_supported_formats()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "formats": formats
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/auth/register")
async def register_endpoint(data: RegisterRequest):
    """Register a new user account."""
    try:
        result = register_user(data.email, data.password)
        
        if result['success']:
            return JSONResponse(
                status_code=201,
                content={
                    "status": "success",
                    "message": result['message'],
                    "data": result.get('data', {})
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Registration failed: {str(e)}"
            }
        )

@app.post("/auth/verify")
async def verify_endpoint(data: VerifyRequest):
    """Verify user email with verification code."""
    try:
        result = verify_user(data.email, data.code)
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message']
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Verification failed: {str(e)}"
            }
        )

@app.post("/auth/login")
async def login_endpoint(data: LoginRequest):
    """Authenticate user and create session."""
    try:
        result = login_user(data.email, data.password)
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message'],
                    "data": {
                        "user_id": result['data']['user_id'],
                        "email": result['data']['email'],
                        "token": result['data']['token']
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Login failed: {str(e)}"
            }
        )

@app.post("/auth/reset-password-request")
async def reset_password_request_endpoint(data: ResetPasswordRequestModel):
    """Request password reset code."""
    try:
        result = reset_password_request(data.email)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": result['message']
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Password reset request failed: {str(e)}"
            }
        )

@app.post("/auth/reset-password")
async def reset_password_endpoint(data: ResetPasswordModel):
    """Reset password with verification code."""
    try:
        result = reset_password(data.email, data.code, data.new_password)
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message']
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Password reset failed: {str(e)}"
            }
        )

@app.post("/auth/resend-code")
async def resend_verification_code_endpoint(data: dict):
    """Resend verification code to user email."""
    try:
        email = data.get("email")
        
        if not email:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Email is required"
                }
            )
        
        result = resend_verification_code(email)
        
        return JSONResponse(
            status_code=200 if result['success'] else 400,
            content={
                "status": "success" if result['success'] else "error",
                "message": result['message'],
                "data": result.get('data', {})
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Resend failed: {str(e)}"
            }
        )

@app.get("/auth/health")
async def auth_health_check():
    """Check authentication system health."""
    try:
        from services.auth_service import get_db_connection, SMTP_CONFIG
        
        health_status = {
            "database": "unknown",
            "smtp": "unknown",
            "overall": "healthy"
        }
        
        # Test database connection
        try:
            conn = get_db_connection()
            if conn and conn.is_connected():
                health_status["database"] = "healthy"
                conn.close()
            else:
                health_status["database"] = "unhealthy"
                health_status["overall"] = "degraded"
        except Exception:
            health_status["database"] = "unhealthy"
            health_status["overall"] = "degraded"
        
        # Check SMTP configuration
        if all([
            SMTP_CONFIG['email'],
            SMTP_CONFIG['password'],
            SMTP_CONFIG['server'],
            SMTP_CONFIG['port']
        ]):
            health_status["smtp"] = "configured"
        else:
            health_status["smtp"] = "not_configured"
            health_status["overall"] = "degraded"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": health_status["overall"],
                "components": health_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@app.post("/admin/login")
async def admin_login_endpoint(data: AdminLoginRequest):
    """
    Admin login endpoint.
    Accepts username or email with password.
    """
    try:
        result = admin_login(data.username, data.password)
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message'],
                    "data": result.get('data', {})
                }
            )
        else:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Admin login failed: {str(e)}"
            }
        )

@app.post("/admin/create")
async def create_admin_endpoint(data: CreateAdminRequest, admin_token: str = None):
    """
    Create new admin account (super_admin only).
    Requires admin_token in header or query parameter.
    
    Note: In production, implement proper token validation.
    For now, pass admin_id as a query parameter.
    """
    try:
        # TODO: Validate admin_token and extract admin_id
        # For now, you'll need to pass created_by_id
        # In production, decode the token to get admin_id
        
        result = create_admin(
            username=data.username,
            email=data.email,
            password=data.password,
            role=data.role,
            created_by_id=None  # Replace with decoded admin_id from token
        )
        
        if result['success']:
            return JSONResponse(
                status_code=201,
                content={
                    "status": "success",
                    "message": result['message'],
                    "data": result.get('data', {})
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Admin creation failed: {str(e)}"
            }
        )

@app.post("/admin/change-password")
async def change_admin_password_endpoint(data: ChangeAdminPasswordRequest, admin_id: int):
    """
    Change admin password.
    Requires admin_id (from decoded token in production).
    """
    try:
        result = change_admin_password(
            admin_id=admin_id,
            current_password=data.current_password,
            new_password=data.new_password
        )
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message']
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Password change failed: {str(e)}"
            }
        )

@app.get("/admin/list")
async def list_admins_endpoint(requesting_admin_id: int):
    """
    Get list of all admins (super_admin only).
    Requires requesting_admin_id (from decoded token in production).
    """
    try:
        result = get_all_admins(requesting_admin_id)
        
        if result['success']:
            # ✅ Convert datetime objects to strings
            admins = result.get('data', {}).get('admins', [])
            for admin in admins:
                if admin.get('created_at'):
                    admin['created_at'] = admin['created_at'].isoformat() if hasattr(admin['created_at'], 'isoformat') else str(admin['created_at'])
                if admin.get('last_login'):
                    admin['last_login'] = admin['last_login'].isoformat() if hasattr(admin['last_login'], 'isoformat') else str(admin['last_login'])
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message'],
                    "data": {
                        "admins": admins,
                        "total_count": result.get('data', {}).get('total_count', len(admins))
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to retrieve admins: {str(e)}"
            }
        )

@app.post("/admin/toggle-status")
async def toggle_admin_status_endpoint(data: AdminActionRequest, requesting_admin_id: int):
    """
    Activate or deactivate an admin account (super_admin only).
    Requires requesting_admin_id (from decoded token in production).
    """
    try:
        result = toggle_admin_status(
            admin_id=data.admin_id,
            requesting_admin_id=requesting_admin_id
        )
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": result['message']
                }
            )
        else:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "message": result['message']
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to toggle admin status: {str(e)}"
            }
        )

@app.get("/admin/dashboard-stats")
async def admin_dashboard_stats(admin_id: int):
    """
    Get dashboard statistics for admin panel.
    Returns user count, admin count, recent activity, etc.
    """
    try:
        from services.auth_service import get_db_connection
        
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        # Get total users
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()['total']
        
        # Get verified users
        cursor.execute("SELECT COUNT(*) as verified FROM users WHERE is_verified = TRUE")
        verified_users = cursor.fetchone()['verified']
        
        # Get total admins
        cursor.execute("SELECT COUNT(*) as total FROM admins WHERE is_active = TRUE")
        total_admins = cursor.fetchone()['total']
        
        # Get recent registrations (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) as recent 
            FROM users 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        recent_registrations = cursor.fetchone()['recent']
        
        # Get recent logins (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) as recent 
            FROM users 
            WHERE last_login >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """)
        recent_logins = cursor.fetchone()['recent']
        
        connection.close()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": {
                    "total_users": total_users,
                    "verified_users": verified_users,
                    "unverified_users": total_users - verified_users,
                    "total_admins": total_admins,
                    "recent_registrations_7d": recent_registrations,
                    "recent_logins_24h": recent_logins,
                    "timestamp": datetime.utcnow().isoformat()  # ✅ Convert to ISO string
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
@app.get("/admin/users")
async def admin_get_users(
    admin_id: int,
    page: int = 1,
    limit: int = 20,
    verified_only: bool = False
):
    """
    Get paginated list of users (admin only).
    """
    try:
        from services.auth_service import get_db_connection
        
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        # Verify admin exists
        cursor.execute("SELECT role FROM admins WHERE id = %s AND is_active = TRUE", (admin_id,))
        admin = cursor.fetchone()
        
        if not admin:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "message": "Unauthorized"
                }
            )
        
        offset = (page - 1) * limit
        
        # Build query
        where_clause = "WHERE is_verified = TRUE" if verified_only else ""
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) as total FROM users {where_clause}")
        total_count = cursor.fetchone()['total']
        
        # Get users
        cursor.execute(f"""
            SELECT id, email, is_verified, created_at, last_login
            FROM users
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        
        users = cursor.fetchall()
        
        # ✅ Convert datetime objects to ISO strings
        for user in users:
            if user.get('created_at'):
                user['created_at'] = user['created_at'].isoformat() if hasattr(user['created_at'], 'isoformat') else str(user['created_at'])
            if user.get('last_login') and user['last_login'] is not None:
                user['last_login'] = user['last_login'].isoformat() if hasattr(user['last_login'], 'isoformat') else str(user['last_login'])
            else:
                user['last_login'] = None
        
        connection.close()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": {
                    "users": users,
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total": total_count,
                        "total_pages": (total_count + limit - 1) // limit
                    }
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
        
@app.delete("/admin/delete-user/{user_id}")
async def admin_delete_user(user_id: int, requesting_admin_id: int):
    """
    Delete a user account (admin only).
    """
    try:
        from services.auth_service import get_db_connection
        
        connection = get_db_connection()
        if not connection:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = connection.cursor(dictionary=True)
        
        # Verify admin has permission
        cursor.execute(
            "SELECT role FROM admins WHERE id = %s AND is_active = TRUE",
            (requesting_admin_id,)
        )
        admin = cursor.fetchone()
        
        if not admin or admin['role'] not in ['super_admin', 'admin']:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "message": "Unauthorized. Only admins can delete users."
                }
            )
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        
        if cursor.rowcount == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "User not found"
                }
            )
        
        connection.commit()
        connection.close()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "User deleted successfully"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.get("/test-smtp")
async def test_smtp_endpoint():
    """Test SMTP connection configuration."""
    try:
        from services.auth_service import test_smtp_connection
        result = test_smtp_connection()
        
        return JSONResponse(
            status_code=200 if result['success'] else 500,
            content=result
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )