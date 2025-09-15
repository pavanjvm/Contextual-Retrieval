import tiktoken
from pypdf import PdfReader
from typing import Generator, List, Dict
tokenizer = tiktoken.get_encoding("cl100k_base")
from loguru import logger
from tqdm import tqdm
def pdf_to_chunks(pdf_path:str, chunk_size:int, overlap:int, cross_page:bool, batch_size:int) -> Generator[List[Dict[str, str]], None, None]:
    reader = PdfReader(pdf_path)
    buffer_tokens = []
    batch_chunks = []
    # logger.info("Processing pages")
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        tokens = tokenizer.encode(text)
        
        if cross_page:
            buffer_tokens.extend(tokens)
        else:
            buffer_tokens = tokens
        
        # Process chunks
        while len(buffer_tokens) >= chunk_size:
            chunk_tokens = buffer_tokens[:chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            batch_chunks.append(chunk_text)
            # Remove processed tokens, keep overlap
            buffer_tokens = buffer_tokens[chunk_size - overlap:]
            
            # Yield batch when ready
            if len(batch_chunks) >= batch_size:
                yield batch_chunks
                batch_chunks = []
    
    # Remaining tokens only for non-cross-page
    if buffer_tokens and not cross_page:
        chunk_text = tokenizer.decode(buffer_tokens)
        batch_chunks.append(chunk_text)
    
    # Yield final batch
    if batch_chunks:
        yield batch_chunks


