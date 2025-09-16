from openai import OpenAI
from dotenv import load_dotenv
from chatbot.agent import Agent
load_dotenv()
client = OpenAI()

system_instruction = '''
You are a helpful AI assistant with access to a knowledge base containing detailed information about the book "The Richest Man in Babylon." Your goal is to answer user questions accurately, using the KB whenever needed. 

Guidelines:
- Make tool calls only if somethings is ask about the book.
- Summarize and synthesize information from the KB rather than copying verbatim.
- Provide clear, concise, and actionable explanations, including examples or lessons from the book where appropriate.
- If the KB does not contain sufficient information, indicate that clearly and avoid making assumptions.
- Use simple, engaging language suitable for users seeking financial wisdom and practical advice.
- Support your answers with citations from the KB when relevant.

Your response flow:
1. Receive user query.
2. Formulate an optimized query for the KB.
3. Call the retriever tool with the query.
4. Analyze retrieved documents.
5. Generate a concise, helpful answer based on KB content.
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
