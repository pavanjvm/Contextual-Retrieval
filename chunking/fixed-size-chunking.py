import tiktoken
from pypdf import PdfReader

tokenizer = tiktoken.get_encoding("cl100k_base")

def fixed_chunk_pdf(pdf_path, chunk_size, overlap, cross_page, batch_size):
    reader = PdfReader(pdf_path)
    buffer_tokens = []
    batch_chunks = []
    
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
            batch_chunks.append({
                "page": page_num + 1,
                "text": chunk_text
            })
            # Remove processed tokens, keep overlap
            buffer_tokens = buffer_tokens[chunk_size - overlap:]
            
            # Yield batch when ready
            if len(batch_chunks) >= batch_size:
                yield batch_chunks
                batch_chunks = []
    
    # Remaining tokens only for non-cross-page
    if buffer_tokens and not cross_page:
        chunk_text = tokenizer.decode(buffer_tokens)
        batch_chunks.append({
            "page": page_num + 1,
            "text": chunk_text
        })
    
    # Yield final batch
    if batch_chunks:
        yield batch_chunks