import unittest
from unittest.mock import patch
from llm import ConsoleLLM

class TestConsoleLLM(unittest.TestCase):

    @patch('llm.pipeline')
    def test_llm_initialization(self, mock_pipeline):
        try:
            llm = ConsoleLLM()
            self.assertIsNotNone(llm)
        except Exception as e:
            self.fail(f"ConsoleLLM initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
