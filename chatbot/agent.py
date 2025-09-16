import json
from embedding.embedding_handler import knowledge_retriever
class Agent:
    def __init__(self,client,model,instruction,tools,memory_limit):
        self.client = client
        self.model = model
        self.memory_limit = memory_limit
        self.instruction = instruction
        self.tools = tools
        self.messages = []
        if self.instruction:
            self.messages.append({"role":"system","content":self.instruction})
        
    def run(self):
        while True:
            message = input("you: ")
            if message.lower() == "quit" or message.lower() == "exit":
                return
            self.messages.append({"role":"user","content":message})
            result = self.execute()
            for item in result:
                if item.type == "output_message":
                    self.messages.append({"role":"assistant","content":item.text})
                    print("assistan: ",item.text)
                elif item.type == "function_call":
                    if item.name == "hybrid_search":
                        output = knowledge_retriever(json.loads(item.arguments))
                        self.messages.append({"type":"function_call_output",
                                                "call_id":item.call_id,
                                                "output":json.dumps({
                                                    "knowledge_retriever":output})})
            if len(self.messages) > self.memory_limit:
                self.messages = self.messages[-self.memory_limit:]


    def execute(self):
        response = self.client.responses.create(
            model = self.model,
            tools = self.tools,
            input = self.messages
        )
        return response.output