import asyncio
from langchain_openai import ChatOpenAI
from uqlm import BlackBoxUQ
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompts = ["hii"]

bb_scorer = BlackBoxUQ(
    llm=llm,
    scorers=["semantic_negentropy", "noncontradiction"]
)

async def score_response(prompts):
    result = await bb_scorer.generate_and_score(prompts = prompts,num_responses = 5)
    return result


async def main():
    # Pretend this is the AI's final answer from your RAG system
    ai_answer = "The customer is interested in cloud migration services."

    confidence = await score_response(prompts)

    print(confidence.to_dict())
asyncio.run(main())
