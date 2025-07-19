import unittest
from agent import DigitalTwinningAgent

class TestDigitalTwinningAgent(unittest.TestCase):

    def test_data_requirements_prompt(self):
        agent = DigitalTwinningAgent()
        prompt = "What are the data requirements for a digital twin?"
        response = agent.call(prompt)
        self.assertIn("real-time sensor data", response)

    def test_simulation_prompt(self):
        agent = DigitalTwinningAgent()
        prompt = "Tell me about simulation."
        response = agent.call(prompt)
        self.assertIn("physics-based models", response)

    def test_benefits_prompt(self):
        agent = DigitalTwinningAgent()
        prompt = "What are the benefits?"
        response = agent.call(prompt)
        self.assertIn("predictive maintenance", response)

    def test_unknown_prompt(self):
        agent = DigitalTwinningAgent()
        prompt = "How do I build one?"
        response = agent.call(prompt)
        self.assertIn("I can provide information", response)

if __name__ == '__main__':
    unittest.main()
