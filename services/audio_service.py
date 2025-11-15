#!/usr/bin/env python3
"""
Audio to Pinecone Voice Service

This service accepts MP3 voice input, converts it to text using OpenAI's Whisper,
processes the text as markdown, and stores it in Pinecone with proper chunking and metadata.
"""

import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import tiktoken
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

# OpenAI client for audio processing
from openai import OpenAI


class AudioPineconeService:
    """
    Service for processing MP3 voice input and storing in Pinecone vector database.
    """
    
    def __init__(self, env_path: str = ".env"):
        """Initialize the service with environment configuration."""
        # Convert relative path to absolute using the backend directory
        if not Path(env_path).is_absolute():
            env_path = Path(__file__).parent.parent / env_path
        
        print(f"Loading environment from: {env_path}")
        load_dotenv(str(env_path))
        
        # Load configuration from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.embedding_metric = os.getenv("EMBEDDING_METRIC", "cosine")
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX")
        self.pinecone_env = os.getenv("PINECONE_ENV")
        self.pinecone_host = os.getenv("PINECONE_HOST")
        self.pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
        
        # Initialize components
        self._init_openai_client()
        self._init_pinecone()
        self._init_embeddings()
        self._init_text_splitter()
        self._init_tokenizer()
        
    def _init_openai_client(self):
        """Initialize OpenAI client for voice processing."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
    def _init_pinecone(self):
        """Initialize Pinecone client and index."""
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
            
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Create index if it doesn't exist
        if not self.pc.has_index(self.pinecone_index_name):
            print(f"Creating Pinecone index: {self.pinecone_index_name}")
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=self.embedding_dimensions,
                metric=self.embedding_metric,
                spec=ServerlessSpec(
                    cloud=self.pinecone_cloud,
                    region=self.pinecone_region
                ),
            )
        
        self.index = self.pc.Index(self.pinecone_index_name)
        
    def _init_embeddings(self):
        """Initialize OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.openai_api_key
        )
        
    def _init_text_splitter(self):
        """Initialize text splitter for chunking documents."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,  # 700 tokens approximately
            chunk_overlap=100,  # 100 tokens overlap
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def convert_mp3_to_text(self, audio_file_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Convert MP3 audio file to text using OpenAI's Whisper.
        
        Args:
            audio_file_path: Path to the MP3 audio file
            language: Language code for transcription (default: "en")
            
        Returns:
            Dictionary containing transcription results and metadata
        """
        print(f"Converting MP3 to text: {audio_file_path}")
        
        try:
            # Verify file exists and is MP3
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            if audio_path.suffix.lower() != '.mp3':
                raise ValueError(f"Only MP3 files are supported. Got: {audio_path.suffix}")
            
            # Open and transcribe the audio file
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",  # Get detailed metadata
                    temperature=0.0  # More deterministic results
                )
            
            transcription_result = {
                "text": transcript.text,
                "language": transcript.language if hasattr(transcript, 'language') else language,
                "duration": transcript.duration if hasattr(transcript, 'duration') else None,
                "segments": transcript.segments if hasattr(transcript, 'segments') else None,
                "source_file": str(audio_path),
                "transcribed_at": datetime.utcnow().isoformat(),
                "word_count": len(transcript.text.split()),
                "character_count": len(transcript.text)
            }
            
            print(f"Transcription complete: {transcription_result['word_count']} words, {transcription_result['character_count']} characters")
            
            return transcription_result
            
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            raise
    
    def convert_mp3_from_base64(self, audio_base64: str, filename: str = None, language: str = "en") -> Dict[str, Any]:
        """
        Convert base64 encoded MP3 audio to text using OpenAI's Whisper.
        
        Args:
            audio_base64: Base64 encoded MP3 audio data
            filename: Optional filename for the audio (used in metadata)
            language: Language code for transcription (default: "en")
            
        Returns:
            Dictionary containing transcription results and metadata
        """
        print("Converting base64 MP3 to text")
        
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(audio_base64)
            
            # Create temporary file-like object
            import io
            audio_buffer = io.BytesIO(audio_data)
            audio_buffer.name = filename or f"audio_{uuid.uuid4().hex[:8]}.mp3"
            
            # Transcribe the audio
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
                language=language,
                response_format="verbose_json",
                temperature=0.0
            )
            
            transcription_result = {
                "text": transcript.text,
                "language": transcript.language if hasattr(transcript, 'language') else language,
                "duration": transcript.duration if hasattr(transcript, 'duration') else None,
                "segments": transcript.segments if hasattr(transcript, 'segments') else None,
                "source_file": filename or "base64_audio.mp3",
                "transcribed_at": datetime.utcnow().isoformat(),
                "word_count": len(transcript.text.split()),
                "character_count": len(transcript.text),
                "input_type": "base64"
            }
            
            print(f"Transcription complete: {transcription_result['word_count']} words")
            
            return transcription_result
            
        except Exception as e:
            print(f"Error transcribing base64 audio: {str(e)}")
            raise
    
    def enhance_transcription_with_markdown(self, transcription_result: Dict[str, Any]) -> str:
        """
        Use ChatOpenAI to enhance the transcription with proper markdown formatting.
        
        Args:
            transcription_result: Result from audio transcription
            
        Returns:
            Enhanced text with markdown formatting
        """
        print("Enhancing transcription with markdown formatting")
        
        try:
            # Initialize ChatOpenAI for text enhancement
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.openai_api_key
            )
            
            # Create enhancement prompt
            enhancement_prompt = f"""
Please convert the following voice transcription into well-formatted markdown text. 
Apply appropriate formatting including:
- Headers for main topics and sections
- Lists for enumerated items
- Bold/italic for emphasis
- Proper paragraph breaks
- Code blocks if technical content is mentioned
- Quotes if the speaker is quoting someone

Preserve all the original meaning and content while making it more readable and structured.

Original transcription:
{transcription_result['text']}
"""
            
            response = llm.invoke([("user", enhancement_prompt)])
            enhanced_text = response.content
            
            print(f"Text enhancement complete: {len(enhanced_text)} characters")
            
            return enhanced_text
            
        except Exception as e:
            print(f"Error enhancing transcription: {str(e)}")
            # Fallback to original text with basic markdown
            return f"# Voice Transcription\n\n{transcription_result['text']}"
    
    def extract_section_from_content(self, content: str, chunk_index: int) -> str:
        """Extract section information from markdown content."""
        lines = content.split('\n')
        
        # Look for markdown headers
        for line in lines:
            if line.startswith('#'):
                return line.strip().replace('#', '').strip()
        
        # Fallback: use first line or chunk index
        first_line = lines[0].strip() if lines else ""
        if len(first_line) > 0 and len(first_line) < 100:
            return first_line
            
        return f"Voice Section {chunk_index + 1}"
    
    def create_audio_metadata(
        self, 
        transcription_result: Dict[str, Any], 
        chunk_index: int, 
        chunk_text: str
    ) -> Dict[str, Any]:
        """
        Create structured metadata for each voice transcription chunk.
        
        Args:
            transcription_result: Original transcription data
            chunk_index: Index of the current chunk
            chunk_text: Text content of the chunk
            
        Returns:
            Dictionary containing structured metadata
        """
        source_file = transcription_result.get("source_file", "unknown_audio.mp3")
        title = os.path.splitext(os.path.basename(source_file))[0]
        section = self.extract_section_from_content(chunk_text, chunk_index)
        
        metadata = {
            "id": str(uuid.uuid4()),
            "title": title,
            "chunk_index": chunk_index,
            "source": source_file,
            "source_type": "voice_transcription",
            "section": section,
            "text": chunk_text,
            "created_at": datetime.utcnow().isoformat(),
            "token_count": self._count_tokens(chunk_text),
            "original_language": transcription_result.get("language", "en"),
            "audio_duration": transcription_result.get("duration"),
            "transcribed_at": transcription_result.get("transcribed_at"),
            "word_count": transcription_result.get("word_count"),
            "input_type": transcription_result.get("input_type", "file")
        }
        
        return metadata
    
    def chunk_transcription(self, enhanced_text: str, transcription_result: Dict[str, Any]) -> List[Document]:
        """
        Split enhanced transcription into chunks with structured metadata.
        
        Args:
            enhanced_text: Enhanced markdown text
            transcription_result: Original transcription data
            
        Returns:
            List of chunked documents with metadata
        """
        # Split the text into chunks
        chunks = self.text_splitter.split_text(enhanced_text)
        
        print(f"Split transcription into {len(chunks)} chunks")
        
        chunked_docs = []
        for chunk_index, chunk_text in enumerate(chunks):
            # Create structured metadata for this chunk
            metadata = self.create_audio_metadata(
                transcription_result, chunk_index, chunk_text
            )
            
            # Create a new document with the chunk and metadata
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            
            chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def store_in_pinecone(self, chunked_docs: List[Document]) -> None:
        """
        Store chunked documents in Pinecone vector database.
        
        Args:
            chunked_docs: List of chunked documents with metadata
        """
        print(f"Storing {len(chunked_docs)} voice chunks in Pinecone...")
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        
        # Extract IDs from metadata
        ids = [doc.metadata["id"] for doc in chunked_docs]
        
        # Add documents to vector store
        vector_store.add_documents(documents=chunked_docs, ids=ids)
        
        print(f"Successfully stored {len(chunked_docs)} chunks in Pinecone index: {self.pinecone_index_name}")
    
    def process_mp3_file(self, audio_file_path: str, language: str = "en", enhance_with_ai: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline to process an MP3 file and store in Pinecone.
        
        Args:
            audio_file_path: Path to the MP3 audio file
            language: Language code for transcription
            enhance_with_ai: Whether to enhance transcription with AI formatting
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Step 1: Convert MP3 to text
            transcription_result = self.convert_mp3_to_text(audio_file_path, language)
            
            # Step 2: Enhance with markdown formatting (optional)
            if enhance_with_ai:
                enhanced_text = self.enhance_transcription_with_markdown(transcription_result)
            else:
                enhanced_text = f"# Voice Transcription\n\n{transcription_result['text']}"
            
            # Step 3: Chunk the enhanced text
            chunked_docs = self.chunk_transcription(enhanced_text, transcription_result)
            
            # Step 4: Store in Pinecone
            self.store_in_pinecone(chunked_docs)
            
            # Return summary
            result = {
                "success": True,
                "source_file": audio_file_path,
                "transcription_length": len(transcription_result['text']),
                "enhanced_length": len(enhanced_text),
                "total_chunks": len(chunked_docs),
                "total_tokens": sum(doc.metadata["token_count"] for doc in chunked_docs),
                "audio_duration": transcription_result.get("duration"),
                "language": transcription_result.get("language"),
                "index_name": self.pinecone_index_name,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            print(f"\n{'='*60}")
            print("AUDIO PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"File: {result['source_file']}")
            print(f"Duration: {result.get('audio_duration', 'Unknown')}s")
            print(f"Language: {result.get('language', 'Unknown')}")
            print(f"Transcription: {result['transcription_length']} chars")
            print(f"Enhanced: {result['enhanced_length']} chars")
            print(f"Chunks created: {result['total_chunks']}")
            print(f"Total tokens: {result['total_tokens']}")
            print(f"Stored in index: {result['index_name']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "source_file": audio_file_path,
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
            print(f"Error processing {audio_file_path}: {str(e)}")
            return error_result
    
    def process_mp3_from_base64(self, audio_base64: str, filename: str = None, language: str = "en", enhance_with_ai: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline to process base64 MP3 audio and store in Pinecone.
        
        Args:
            audio_base64: Base64 encoded MP3 audio data
            filename: Optional filename for metadata
            language: Language code for transcription
            enhance_with_ai: Whether to enhance transcription with AI formatting
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Step 1: Convert base64 MP3 to text
            transcription_result = self.convert_mp3_from_base64(audio_base64, filename, language)
            
            # Step 2: Enhance with markdown formatting (optional)
            if enhance_with_ai:
                enhanced_text = self.enhance_transcription_with_markdown(transcription_result)
            else:
                enhanced_text = f"# Voice Transcription\n\n{transcription_result['text']}"
            
            # Step 3: Chunk the enhanced text
            chunked_docs = self.chunk_transcription(enhanced_text, transcription_result)
            
            # Step 4: Store in Pinecone
            self.store_in_pinecone(chunked_docs)
            
            # Return summary
            result = {
                "success": True,
                "source_file": filename or "base64_audio.mp3",
                "transcription_length": len(transcription_result['text']),
                "enhanced_length": len(enhanced_text),
                "total_chunks": len(chunked_docs),
                "total_tokens": sum(doc.metadata["token_count"] for doc in chunked_docs),
                "audio_duration": transcription_result.get("duration"),
                "language": transcription_result.get("language"),
                "index_name": self.pinecone_index_name,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            print(f"\n{'='*60}")
            print("BASE64 AUDIO PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Duration: {result.get('audio_duration', 'Unknown')}s")
            print(f"Language: {result.get('language', 'Unknown')}")
            print(f"Transcription: {result['transcription_length']} chars")
            print(f"Enhanced: {result['enhanced_length']} chars")
            print(f"Chunks created: {result['total_chunks']}")
            print(f"Total tokens: {result['total_tokens']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "source_file": filename or "base64_audio.mp3",
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }
            print(f"Error processing base64 audio: {str(e)}")
            return error_result
    
    def process_directory(self, directory_path: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Process all MP3 files in a directory.
        
        Args:
            directory_path: Path to directory containing MP3 files
            language: Language code for transcription
            
        Returns:
            List of processing results for each file
        """
        directory = Path(directory_path)
        mp3_files = list(directory.glob("*.mp3"))
        
        if not mp3_files:
            print(f"No MP3 files found in {directory_path}")
            return []
        
        print(f"Found {len(mp3_files)} MP3 files to process")
        
        results = []
        for mp3_file in mp3_files:
            print(f"\n{'='*60}")
            print(f"Processing: {mp3_file.name}")
            print(f"{'='*60}")
            
            result = self.process_mp3_file(str(mp3_file), language)
            results.append(result)
        
        return results
    
    def query_voice_content(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Query the Pinecone index for voice transcription content.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of matching documents with metadata
        """
        # Add voice content filter if not specified
        if filter_dict is None:
            filter_dict = {"source_type": "voice_transcription"}
        elif "source_type" not in filter_dict:
            filter_dict["source_type"] = "voice_transcription"
        
        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        
        results = vector_store.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            })
        
        return formatted_results


def main():
    """
    Main function to demonstrate the audio processing pipeline.
    """
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Audio to Pinecone Voice Service")
    print("="*50)
    
    try:
        # Use absolute paths relative to the backend directory
        backend_dir = Path(__file__).parent.parent
        env_path = backend_dir / ".env"
        data_dir = backend_dir / "data"
        
        print(f"Loading environment from: {env_path}")
        service = AudioPineconeService(str(env_path))
        
        # Process files in the data directory
        if not data_dir.exists():
            data_dir.mkdir(exist_ok=True)
            print(f"Created data directory: {data_dir}")
            print("Place your MP3 files in the data directory and run again.")
            return
        
        # Process all MP3 files in the directory
        results = service.process_directory(str(data_dir))
        
        # Display summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"Total files processed: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            total_chunks = sum(r["total_chunks"] for r in successful)
            total_tokens = sum(r["total_tokens"] for r in successful)
            total_duration = sum(r.get("audio_duration", 0) or 0 for r in successful)
            
            print(f"Total chunks created: {total_chunks}")
            print(f"Total tokens processed: {total_tokens}")
            print(f"Total audio duration: {total_duration:.2f} seconds")
        
        # Example query (uncomment to test)
        # print(f"\n{'='*60}")
        # print("EXAMPLE VOICE QUERY")
        # print(f"{'='*60}")
        # query_results = service.query_voice_content("your search query here", k=3)
        # for i, result in enumerate(query_results, 1):
        #     print(f"\n--- Result {i} (Score: {result['similarity_score']:.4f}) ---")
        #     print(f"Title: {result['metadata']['title']}")
        #     print(f"Duration: {result['metadata'].get('audio_duration', 'Unknown')}s")
        #     print(f"Language: {result['metadata'].get('original_language', 'Unknown')}")
        #     print(f"Content: {result['content'][:200]}...")
        
    except Exception as e:
        print(f"Error initializing service: {str(e)}")
        print("\nMake sure:")
        print("1. Your .env file contains all required environment variables")
        print("2. You have installed the required dependencies:")
        print("   pip install langchain-openai langchain-pinecone openai")
        print("   pip install tiktoken python-dotenv pinecone-client")


if __name__ == "__main__":
    main()