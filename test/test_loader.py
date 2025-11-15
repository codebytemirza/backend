#!/usr/bin/env python3
"""
DOCX to Markdown Loader using Docling

This program loads a DOCX file and converts it to markdown format using Docling.
It displays the markdown content with proper formatting.
"""

import os
from pathlib import Path
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType


def load_docx_as_markdown(file_path):
    """
    Load a DOCX file and convert it to markdown format.
    
    Args:
        file_path (str): Path to the DOCX file (local file or URL)
    
    Returns:
        list: List of LangChain Document objects containing markdown content
    """
    print(f"Loading DOCX file: {file_path}")
    
    # Initialize the DoclingLoader with markdown export type
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.MARKDOWN  # Export as markdown format
    )
    
    # Load the documents
    docs = loader.load()
    
    return docs


def display_markdown_content(docs):
    """
    Display the markdown content from the loaded documents.
    
    Args:
        docs (list): List of LangChain Document objects
    """
    print("\n" + "="*60)
    print("MARKDOWN CONTENT")
    print("="*60)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content Length: {len(doc.page_content)} characters")
        
        # Display metadata if available
        if doc.metadata:
            print("\nMetadata:")
            for key, value in doc.metadata.items():
                if key != "pk":  # Skip internal keys
                    print(f"  {key}: {value}")
        
        print("\n--- Content ---")
        print(doc.page_content)
        print("\n" + "-"*40)


def main():
    """
    Main function to demonstrate DOCX to markdown conversion.
    """
    # Use absolute path and create data directory if it doesn't exist
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / "sample.docx"  # Change to your actual DOCX filename

    # Single file existence check
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        print("\nTo use this program:")
        print(f"1. Place your DOCX file in: {data_dir}")
        print("2. Update the file_path variable with your filename")
        print("\nExample usage:")
        print(f"  file_path = '{data_dir}/your_document.docx'")
        return
    
    try:
        # Load the DOCX file as markdown
        docs = load_docx_as_markdown(str(file_path))
        
        # Display the markdown content
        display_markdown_content(docs)
        
        print(f"\nSuccessfully processed {len(docs)} document(s)")
        
    except Exception as e:
        print(f"Error processing the DOCX file: {str(e)}")
        print("\nMake sure:")
        print("1. The file path is correct")
        print("2. The file is a valid DOCX document")
        print("3. You have installed the required dependencies:")
        print("   pip install langchain-docling")


def load_and_save_markdown(input_file, output_file=None):
    """
    Load a DOCX file and save the markdown content to a file.
    
    Args:
        input_file (str): Path to input DOCX file
        output_file (str): Path to output markdown file (optional)
    """
    # Convert input_file to Path object if it's a string
    input_path = Path(input_file)
    
    if output_file is None:
        # Save output in the same directory as input file
        output_file = input_path.with_suffix('.md')
    
    try:
        docs = load_docx_as_markdown(str(input_path))
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(docs):
                if i > 0:
                    f.write("\n\n---\n\n")
                f.write(doc.page_content)
        
        print(f"Markdown content saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving markdown: {str(e)}")


if __name__ == "__main__":
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("DOCX to Markdown Converter using Docling")
    print("="*50)
    
    # Run the main function
    main()
    
    # Example of saving to file (uncomment to use):
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "sample.docx"
    output_file = input_file.with_suffix('.md')
    load_and_save_markdown(input_file, output_file)