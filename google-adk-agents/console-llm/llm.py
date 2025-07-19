import sys
sys.path.append('../sharepoint-agent')
from agent import SharePointAgent
from transformers import pipeline
import os

class ConsoleLLM:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline('text-generation', model=model_name)
        self.sharepoint_agent = None

    def connect_to_sharepoint(self, site_url, client_id, client_secret):
        try:
            self.sharepoint_agent = SharePointAgent(site_url, client_id, client_secret)
            print("Successfully connected to SharePoint.")
        except Exception as e:
            print(f"Error connecting to SharePoint: {e}")

    def start_chat(self):
        print("Console LLM initialized. Type 'exit' to end the session.")
        print("First, connect to SharePoint using the command: connect <site_url> <client_id> <client_secret>")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            if user_input.startswith("connect "):
                parts = user_input.split(" ")
                if len(parts) == 4:
                    self.connect_to_sharepoint(parts[1], parts[2], parts[3])
                else:
                    print("Usage: connect <site_url> <client_id> <client_secret>")
                continue

            if self.sharepoint_agent:
                # This is a conceptual integration. A real implementation would have more
                # sophisticated interaction between the LLM and the agent.
                if user_input.startswith("analyze "):
                    response = self.sharepoint_agent.call(user_input)
                else:
                    response = self.generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
            else:
                response = "Please connect to SharePoint first."

            print(f"LLM: {response}")

if __name__ == "__main__":
    llm = ConsoleLLM()
    llm.start_chat()
