import tiktoken
from pypdf import PdfReader

tokenizer = tiktoken.get_encoding("cl100k_base")

def fixed_chunk_pdf(pdf_path, chunk_size=500, overlap=100):
    reader = PdfReader(pdf_path)
    chunks = []
    buffer_tokens = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        tokens = tokenizer.encode(text)

        for token in tokens:
            buffer_tokens.append(token)
            if len(buffer_tokens) >= chunk_size:
                # take chunk and leave overlap for next one
                chunk_tokens = buffer_tokens[:chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append({
                    "page": page_num + 1,
                    "text": chunk_text
                })
                buffer_tokens = buffer_tokens[chunk_size - overlap:]  # keep overlap

    # handle leftovers
    if buffer_tokens:
        chunks.append({
            "page": page_num + 1,
            "text": tokenizer.decode(buffer_tokens)
        })

    return chunks

# Example usage
pdf_file = "large_doc.pdf"
chunks = fixed_chunk_pdf(pdf_file, chunk_size=500, overlap=100)

print(f"Total chunks: {len(chunks)}")
print(chunks[0])  # preview first chunk
