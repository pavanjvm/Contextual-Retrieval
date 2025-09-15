import openai
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
def get_embeddings(texts:list):
    
    # logger.info ("creating embeddings")
    response = openai.embeddings.create(
            model="text-embedding-3-large",
            dimensions = 1024,
            input=texts
        )
    return response