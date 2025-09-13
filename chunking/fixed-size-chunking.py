import tiktoken
from pypdf import PdfReader

tokenizer = tiktoken.get_encoding("cl100k_base")

def fixed_chunk_pdf(pdf_path, chunk_size=500, overlap=100, cross_page=False):
    reader = PdfReader(pdf_path)
    chunks = []
    buffer_tokens = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        tokens = tokenizer.encode(text)

        if cross_page:
            # accumulate across pages
            buffer_tokens.extend(tokens)
        else:
            # reset each page
            buffer_tokens = tokens  

        # slice into chunks directly
        for i in range(0, len(buffer_tokens), chunk_size - overlap):
            chunk_tokens = buffer_tokens[i : i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append({
                "page": page_num + 1,
                "text": chunk_text
            })

    return chunks
