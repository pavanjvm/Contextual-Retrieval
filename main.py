from openai import OpenAI
from dotenv import load_dotenv
from chatbot.agent import Agent
load_dotenv()
client = OpenAI()

system_instruction = '''
You are an AI chat assistant capable of retrieving relevant documents using your available tools.
When a user asks a question, begin with a concise checklist (3-7 bullets) outlining your process,
such as analyzing the user's question, formulating an effective search query, and selecting retrieval methods.
Before sending a query to any retrieval tool, briefly state the purpose of the call and the minimal inputs used. 
The retrieval tool employs BM25 for sparse retrieval and cosine similarity for dense vector search. 
After each retrieval, validate in 1-2 lines that the returned documents align with the user's intent, 
and decide on the next step or re-query if needed. Ensure each query is optimized for the most relevant document results.
'''

tools = [
    {
        "type":"function",
        "name":"knowledge_retriever",
        "description":"Performs a hybrid search using semantic embeddings and keyword matching to retrieve the most relevant documents for a query.",
        "parameters":{
            "type":"object",
            "properties":{
                "query_text":{
                    "type":"string",
                    "description":"the text query to search for"
                },
                "limit":{
                    "type":"integer",
                    "description":"the number of documents needed from the tool to answer the user query",
                }
                },
                "required":["query_text"]
            }
        }
]
my_agent = Agent(client = client,model = "gpt-5",instruction = system_instruction,tools = tools,memory_limit = 10)

my_agent.run()
