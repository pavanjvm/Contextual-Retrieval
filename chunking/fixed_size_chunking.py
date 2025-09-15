import tiktoken
from pypdf import PdfReader
from typing import Generator, List, Dict
tokenizer = tiktoken.get_encoding("cl100k_base")
from loguru import logger
from tqdm import tqdm

def pdf_to_chunks(pdf_path: str, chunk_size: int, overlap: int, cross_page: bool) -> Generator[List[Dict[str, str]], None, None]:
    reader = PdfReader(pdf_path)
    buffer_tokens = []
    batch_chunks = []
    batch_token_count = 0
    current_page = 0
    max_tokens = 200000

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        tokens = tokenizer.encode(text)

        if cross_page:
            buffer_tokens.extend(tokens)
            if not buffer_tokens or page_num < current_page:
                current_page = page_num
        else:
            buffer_tokens = tokens
            current_page = page_num

        # Process chunks
        while len(buffer_tokens) >= chunk_size:
            chunk_tokens = buffer_tokens[:chunk_size]
            chunk_len = len(chunk_tokens)

            # If adding this chunk would overflow → yield current batch
            if batch_token_count + chunk_len > max_tokens:
                if batch_chunks:
                    yield batch_chunks.copy()
                batch_chunks = []
                batch_token_count = 0

            # Add chunk to current batch
            chunk_text = tokenizer.decode(chunk_tokens)
            batch_chunks.append({
                "page": current_page + 1,
                "chunk": chunk_text
            })
            batch_token_count += chunk_len

            # Slide the buffer
            buffer_tokens = buffer_tokens[chunk_size - overlap:]

    # Handle leftover tokens
    if buffer_tokens:
        chunk_text = tokenizer.decode(buffer_tokens)
        batch_chunks.append({
            "page": current_page + 1,
            "chunk": chunk_text

        })

    # Yield final batch
    if batch_chunks:
        yield batch_chunks


