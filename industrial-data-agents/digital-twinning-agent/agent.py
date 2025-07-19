import adk

class DigitalTwinningAgent(adk.Agent):
    def __init__(self):
        super().__init__()
        self.knowledge_base = {
            "data requirements": "For a digital twin, you typically need real-time sensor data (e.g., temperature, pressure), historical operational data, 3D models of the asset, and maintenance records.",
            "simulation": "Digital twin simulations often involve physics-based models, finite element analysis, and computational fluid dynamics to predict asset behavior and performance.",
            "benefits": "The benefits of digital twinning include predictive maintenance, improved operational efficiency, and the ability to test 'what-if' scenarios in a virtual environment without affecting the physical asset."
        }

    def call(self, prompt: str) -> str:
        prompt = prompt.lower()
        for keyword, response in self.knowledge_base.items():
            if keyword in prompt:
                return response
        return "I can provide information on data requirements, simulation, and benefits of digital twinning. Please ask me about one of these topics."

if __name__ == "__main__":
    agent = DigitalTwinningAgent()

    prompt1 = "What are the data requirements for a digital twin?"
    response1 = agent.call(prompt1)
    print(f"Prompt: {prompt1}\nResponse: {response1}\n")

    prompt2 = "Tell me about the benefits."
    response2 = agent.call(prompt2)
    print(f"Prompt: {prompt2}\nResponse: {response2}\n")

    prompt3 = "How do I build one?"
    response3 = agent.call(prompt3)
    print(f"Prompt: {prompt3}\nResponse: {response3}\n")
