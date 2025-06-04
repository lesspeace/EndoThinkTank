import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pypdf # For PDF reading
import re    # For regular expressions (to find PMID and clean text)
from uuid import uuid4 # For generating unique chunk IDs

# --- Configuration ---
DATA_DIR = "data" # Folder where your PDF files are located
FAISS_INDEX_PATH = "endometriosis_faiss.index" # File to save the FAISS index
CHUNKS_DATA_PATH = "endometriosis_chunks_with_ids.json" # File to save chunks with IDs

# Choose an embedding model. 'all-MiniLM-L6-v2' is a good balance of size and performance.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Chunking parameters (you can experiment with these later)
# Slightly smaller chunk size to ensure more focused content within each chunk.
# Higher overlap helps maintain context across chunks.
CHUNK_SIZE = 300 # Character count
CHUNK_OVERLAP = 50

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file using pypdf."""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or "" # Use .extract_text() and handle None
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    """
    Performs basic cleaning on extracted text to improve chunk quality.
    - Removes common headers/footers (e.g., page numbers, journal names).
    - Replaces multiple newlines with single ones.
    - Removes hyphenation from words split across lines.
    - Removes common bibliography markers or stray characters.
    """
    # Remove common patterns like page numbers (e.g., "1", "10", "1 of 20")
    cleaned_text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # Remove lines with only numbers
    cleaned_text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove common journal/document specific headers/footers if they are consistent
    # Example: if you notice "Int. J. Mol. Sci. 2023, 24, 4254" repeating, add a regex for it
    # cleaned_text = re.sub(r'Int\. J\. Mol\. Sci\.\d{4},\s*\d+,\s*\d+', '', cleaned_text)
    
    # Remove hyphenation from words broken across lines (e.g., "endo-metriosis" -> "endometriosis")
    cleaned_text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', cleaned_text)
    
    # Replace multiple newlines with a single newline (or space if you prefer flat text)
    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text) # Replace multiple blank lines with one
    
    # Remove bibliography numbers like [1], [2,3], [4-6] if they appear as standalone
    cleaned_text = re.sub(r'\[\d+(?:,\d+)*\]|\s*\[\d+\-\d+\]', '', cleaned_text)

    # Remove non-alphanumeric leading/trailing characters from lines
    cleaned_text = re.sub(r'^\W+|\W+$', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def get_chunks_from_paragraphs(text, source_metadata, chunk_size_chars, chunk_overlap_chars):
    """
    Splits text into chunks, trying to respect paragraph boundaries.
    Args:
        text (str): The full cleaned text of the document.
        source_metadata (dict): Metadata for the source document.
        chunk_size_chars (int): Desired character length of each chunk.
        chunk_overlap_chars (int): Overlap between chunks in characters.
    Returns:
        list: A list of dictionaries, each representing a chunk.
    """
    chunks = []
    if not text:
        return []

    # Split text into paragraphs based on double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk_text = ""
    for paragraph in paragraphs:
        # If adding the next paragraph makes the current chunk too large,
        # or if the current chunk is empty and this is the first paragraph
        if len(current_chunk_text) + len(paragraph) + 2 > chunk_size_chars and current_chunk_text: # +2 for newline
            chunks.append({
                "id": str(uuid4()),
                "text": current_chunk_text,
                "source": source_metadata
            })
            # For overlap, take the end of the previous chunk or start of new paragraph
            # This simple overlap is char-based; for more complex, consider LangChain
            current_chunk_text = current_chunk_text[-(chunk_size_chars - chunk_overlap_chars):] 
            if not current_chunk_text: # Ensure it's not empty if overlap is small
                current_chunk_text = paragraph[:chunk_overlap_chars] # Just start with some chars from new paragraph
        
        current_chunk_text += ("\n\n" + paragraph if current_chunk_text else paragraph)

    # Add the last chunk if it's not empty
    if current_chunk_text:
        chunks.append({
            "id": str(uuid4()),
            "text": current_chunk_text,
            "source": source_metadata
        })
    
    return chunks

def extract_pmid_from_text(text):
    """Attempts to extract a PubMed ID (PMID) from text using regex."""
    # Common PMID patterns: e.g., "PMID: 12345678", "PMID 12345678", "(PMID: 12345678)"
    match = re.search(r'PMID:\s*(\d+)|PMID\s+(\d+)|PubMed\s+ID:\s*(\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None

def extract_title_from_pdf_metadata(reader):
    """Attempts to extract title from PDF metadata."""
    if reader.metadata and reader.metadata.title:
        return reader.metadata.title
    return None

def create_and_save_vector_db_from_pdfs():
    all_chunks = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and put your PDFs inside.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}'. Please place your PDFs there.")
        return

    print(f"Found {len(pdf_files)} PDF files in '{DATA_DIR}'. Processing...")

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"({i+1}/{len(pdf_files)}) Processing {pdf_file}...")
        
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            print(f"Skipping {pdf_file} due to empty or unextractable text.")
            continue
        
        # --- Improved Metadata Extraction ---
        pmid = extract_pmid_from_text(full_text[:2000]) # Search in the first 2000 chars for PMID
        
        # Try to get title from PDF metadata first
        pdf_reader = pypdf.PdfReader(pdf_path) # Re-open reader to access metadata directly
        title = extract_title_from_pdf_metadata(pdf_reader)
        if not title: # Fallback to filename if no title in metadata
             title = f"Document: {pdf_file}"
        
        source_metadata = {
            "filename": pdf_file,
            "pmid": pmid if pmid else "N/A",
            "title": title
        }
        # --- End Improved Metadata Extraction ---

        # Clean the extracted text before chunking
        cleaned_full_text = clean_text(full_text)
        if not cleaned_full_text:
            print(f"Skipping {pdf_file} due to empty text after cleaning.")
            continue

        file_chunks = get_chunks_from_paragraphs(cleaned_full_text, source_metadata, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(file_chunks)
        print(f"   Extracted {len(file_chunks)} chunks from {pdf_file}.")

    if not all_chunks:
        print("No chunks generated from any PDFs. Exiting.")
        return

    print(f"\nTotal chunks generated from all PDFs: {len(all_chunks)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")

    chunk_texts = [chunk['text'] for chunk in all_chunks]

    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    print("Embeddings generated.")

    embeddings = embeddings.astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    print(f"FAISS index created and {index.ntotal} embeddings added.")

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    with open(CHUNKS_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"All chunks data saved to {CHUNKS_DATA_PATH}")

    print("\nPDF Processing, Embedding, and Vector Database creation complete!")

if __name__ == "__main__":
    create_and_save_vector_db_from_pdfs()