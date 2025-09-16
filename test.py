from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client  = OpenAI()

result = client.responses.create(model = "gpt-5", input = "hey hiie")
print(result.output_text)