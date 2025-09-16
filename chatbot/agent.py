import json
from embedding.embedding_handler import knowledge_retriever
from typing import List,Dict,Any
from openai import OpenAI
class Agent:
    def __init__(self,client:OpenAI,model:str,instruction:str,tools:List[Any],memory_limit:int):
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
            # print(result)
            for item in result.output:
                if getattr(item, "content", None):
                    for content in item.content:
                        self.messages.append({"role":"assistant","content":content.text})
                        print("assistant_o: ",content.text)
                elif item.type == "function_call":
                    self.messages.append({"type":"function_call","name":item.name,"arguments":item.arguments,"call_id":item.call_id})
                    if item.name == "knowledge_retriever":
                        output = knowledge_retriever(item.arguments)
                        serializable_output = [
                                                {
                                                    "id": point.id,
                                                    "version": point.version,
                                                    "score": point.score,
                                                    "text": point.payload.get("text") if point.payload else None
                                                }
                                                for point in output
                                            ]
                        print(serializable_output)
                        self.messages.append({"type":"function_call_output",
                                                "call_id":item.call_id,
                                                "output":json.dumps({
                                                    "knowledge_retriever":serializable_output})})
                        tool_resp = self.execute()
                        print("assistant_f: ",tool_resp.output_text)
            
                        
            if len(self.messages) > self.memory_limit:
                self.messages = self.messages[-self.memory_limit:]


    def execute(self):
        # print(self.messages)
        response = self.client.responses.create(
            model = self.model,
            tools = self.tools,
            input = self.messages
        )
        return response