#!/usr/bin/env python3
"""
DOCX to Pinecone Data Ingestion Agent

This program loads DOCX files, converts them to markdown using Docling,
splits them into chunks with proper metadata, and stores embeddings in Pinecone.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import tiktoken
from dotenv import load_dotenv

# LangChain imports
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec


class DocxPineconeIngestionAgent:
    """
    Agent for ingesting DOCX files into Pinecone vector database.
    """
    
    def __init__(self, env_path: str = ".env"):
        """Initialize the agent with environment configuration."""
        load_dotenv(env_path)
        
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
        self._init_pinecone()
        self._init_embeddings()
        self._init_text_splitter()
        self._init_tokenizer()
        
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
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
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
    
    def load_docx_as_markdown(self, file_path: str) -> List[Document]:
        """
        Load a DOCX file and convert it to markdown format.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of LangChain Document objects containing markdown content
        """
        print(f"Loading DOCX file: {file_path}")
        
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.MARKDOWN
        )
        
        docs = loader.load()
        print(f"Loaded {len(docs)} document(s) from DOCX file")
        
        return docs
    
    def extract_section_from_content(self, content: str, chunk_index: int) -> str:
        """
        Extract section information from markdown content.
        Simple heuristic: look for headers or use chunk position.
        """
        lines = content.split('\n')
        
        # Look for markdown headers
        for line in lines:
            if line.startswith('#'):
                return line.strip().replace('#', '').strip()
        
        # Fallback: use first line or chunk index
        first_line = lines[0].strip() if lines else ""
        if len(first_line) > 0 and len(first_line) < 100:
            return first_line
            
        return f"Section {chunk_index + 1}"
    
    def create_structured_metadata(
        self, 
        doc: Document, 
        chunk_index: int, 
        chunk_text: str,
        source_file: str
    ) -> Dict[str, Any]:
        """Create structured metadata for each chunk."""
        title = os.path.splitext(os.path.basename(source_file))[0]
        section = self.extract_section_from_content(chunk_text, chunk_index)
        
        # Use datetime.now(UTC) instead of utcnow()
        from datetime import timezone
        
        # Create flattened metadata structure
        metadata = {
            "id": str(uuid.uuid4()),
            "title": title,
            "chunk_index": chunk_index,
            "source": source_file,
            "section": section,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "token_count": self._count_tokens(chunk_text)
        }
        
        # Add original metadata as flattened key-value pairs
        if hasattr(doc, 'metadata'):
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"original_{key}"] = value
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    metadata[f"original_{key}"] = value
        
        return metadata
    
    def chunk_documents(self, docs: List[Document], source_file: str) -> List[Document]:
        """
        Split documents into chunks with structured metadata.
        
        Args:
            docs: List of documents to chunk
            source_file: Source file path
            
        Returns:
            List of chunked documents with metadata
        """
        chunked_docs = []
        
        for doc in docs:
            # Split the document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            print(f"Split document into {len(chunks)} chunks")
            
            for chunk_index, chunk_text in enumerate(chunks):
                # Create structured metadata for this chunk
                metadata = self.create_structured_metadata(
                    doc, chunk_index, chunk_text, source_file
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
        print(f"Storing {len(chunked_docs)} chunks in Pinecone...")
        
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
    
    def process_docx_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete pipeline to process a DOCX file and store in Pinecone.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Step 1: Load DOCX and convert to markdown
            docs = self.load_docx_as_markdown(file_path)
            
            # Step 2: Chunk documents with structured metadata
            chunked_docs = self.chunk_documents(docs, file_path)
            
            # Step 3: Store in Pinecone
            self.store_in_pinecone(chunked_docs)
            
            # Return summary
            result = {
                "success": True,
                "source_file": file_path,
                "total_chunks": len(chunked_docs),
                "total_tokens": sum(doc.metadata["token_count"] for doc in chunked_docs),
                "index_name": self.pinecone_index_name,
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"\n{'='*60}")
            print("PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"File: {result['source_file']}")
            print(f"Chunks created: {result['total_chunks']}")
            print(f"Total tokens: {result['total_tokens']}")
            print(f"Stored in index: {result['index_name']}")
            print(f"Processed at: {result['processed_at']}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "source_file": file_path,
                "error": str(e),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            print(f"Error processing {file_path}: {str(e)}")
            return error_result
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all DOCX files in a directory.
        
        Args:
            directory_path: Path to directory containing DOCX files
            
        Returns:
            List of processing results for each file
        """
        directory = Path(directory_path)
        docx_files = list(directory.glob("*.docx"))
        
        if not docx_files:
            print(f"No DOCX files found in {directory_path}")
            return []
        
        print(f"Found {len(docx_files)} DOCX files to process")
        
        results = []
        for docx_file in docx_files:
            print(f"\n{'='*60}")
            print(f"Processing: {docx_file.name}")
            print(f"{'='*60}")
            
            result = self.process_docx_file(str(docx_file))
            results.append(result)
        
        return results
    
    def query_index(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Query the Pinecone index for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of matching documents with metadata
        """
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
    Main function to demonstrate the DOCX ingestion pipeline.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("DOCX to Pinecone Data Ingestion Agent")
    print("="*50)
    
    try:
        # Use absolute paths relative to the backend directory
        backend_dir = Path(__file__).parent.parent  # Go up one level to backend/
        env_path = backend_dir / ".env"
        data_dir = backend_dir / "data"
        
        print(f"Loading environment from: {env_path}")
        agent = DocxPineconeIngestionAgent(str(env_path))
        
        if not data_dir.exists():
            data_dir.mkdir(exist_ok=True)
            print(f"Created data directory: {data_dir}")
            print("Place your DOCX files in the data directory and run again.")
            return
        
        # Process all DOCX files in the directory
        results = agent.process_directory(str(data_dir))
        
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
            print(f"Total chunks created: {total_chunks}")
            print(f"Total tokens processed: {total_tokens}")
        
        # Example query (uncomment to test)
        # print(f"\n{'='*60}")
        # print("EXAMPLE QUERY")
        # print(f"{'='*60}")
        # query_results = agent.query_index("your search query here", k=3)
        # for i, result in enumerate(query_results, 1):
        #     print(f"\n--- Result {i} (Score: {result['similarity_score']:.4f}) ---")
        #     print(f"Title: {result['metadata']['title']}")
        #     print(f"Section: {result['metadata']['section']}")
        #     print(f"Content: {result['content'][:200]}...")
        
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        print("\nMake sure:")
        print("1. Your .env file contains all required environment variables")
        print("2. You have installed the required dependencies:")
        print("   pip install langchain-docling langchain-pinecone langchain-openai")
        print("   pip install tiktoken python-dotenv pinecone-client")


if __name__ == "__main__":
    main()