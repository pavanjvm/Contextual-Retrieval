import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Document
import uuid
from embedding.openai_embedder import get_embeddings
from loguru import logger
from tqdm import tqdm
from chunking.fixed_size_chunking import pdf_to_chunks
import time

qdrant = QdrantClient(
    url=os.getenv("url"), 
    api_key=os.getenv("api_key"),
    cloud_inference=True
)


if not qdrant.collection_exists(collection_name="hybrid-rag"):
    qdrant.create_collection(
            collection_name="hybrid-rag",
            vectors_config={
                "dense": VectorParams(
                    size=1024, 
                    distance=Distance.COSINE
                )
            },  # size and distance are model dependent
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )



def hybrid_search(query_text:str, limit=5):
    """
    Simple hybrid search - returns most similar vectors for LLM context
    """
    # Get dense embedding for the query
    query_embeddings = get_embeddings([query_text])
    
    
    # Perform hybrid search using FusionQuery
    search_result = qdrant.query_points(
        collection_name="hybrid-rag",
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
        ),
        prefetch=[
            # Dense vector search (semantic similarity)
            models.Prefetch(
                query=query_embeddings.data[0].embedding,
                using="dense",
                limit=limit * 2  # Get more candidates for better fusion
            ),
            # Sparse vector search (keyword matching)
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model="qdrant/bm25"
                ),
                using="sparse", 
                limit=limit * 2
            ),
        ],
        limit=limit
    )
    
    return search_result.points



def store_chunks_as_embeddings(batch_chunks:list,batch_upload_size:int,j:int):

    points = []
    texts = [d["chunk"] for d in batch_chunks]
    logger.info(f"\ntotal chunks in batch {j+1} : {len(texts)} ")
    response = get_embeddings(texts)

    for i,text in enumerate(tqdm(texts,desc = f"batch {j+1}", unit = "chunk",ncols = 100)):  #add tqdm heregit
        point = PointStruct(
        id=uuid.uuid4().hex,  
        payload={"text":text},
        vector={
            "dense": response.data[i].embedding
            ,
            "sparse": Document(
                text=texts[i],
                model="qdrant/bm25"
                )
            }
        )
        points.append(point)
    start_time = time.time()
    qdrant.upload_points(
    collection_name= "hybrid-rag", 
    points=points, 
    batch_size=batch_upload_size
    )
    elapsed = time.time() - start_time
    logger.info(
        f"\nuploaded {i+1} chunks successfully in batch {j+1} "
        f"(upload time: {elapsed:.2f} sec)"
    )
    
def pdf_to_embeddings(pdf_path:str, chunk_size:int, overlap:int, cross_page:bool):
    start_time = time.time() 
    chunks = pdf_to_chunks(pdf_path, chunk_size, overlap, cross_page)
    for i, chunks in enumerate(chunks):
        store_chunks_as_embeddings(chunks,30,i)
    elapsed = time.time() - start_time
    logger.info(f"\nProcessed all chunks and stored them in the DB (took {elapsed:.2f} seconds)")
    


pdf_to_embeddings("AIE.pdf",500,100,True)