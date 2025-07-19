import unittest
from unittest.mock import patch, MagicMock
from agent import IndustrialEquipmentAgent

class TestIndustrialEquipmentAgent(unittest.TestCase):

    @patch('agent.ClientContext')
    @patch('agent.AuthenticationContext')
    def test_get_document_found(self, mock_auth_context, mock_client_context):
        mock_ctx = MagicMock()
        mock_client_context.return_value = mock_ctx
        mock_file = MagicMock()
        mock_file.properties = {"Name": "Pump-123.pdf"}
        mock_file_content_response = MagicMock()
        mock_file_content_response.value = b"Pump manual content"
        mock_file.open_binary.return_value = mock_file_content_response
        mock_ctx.web.lists.get_by_title.return_value.root_folder.files = [mock_file]

        agent = IndustrialEquipmentAgent("url", "id", "secret")
        content = agent.get_document("Manuals", "Pump-123.pdf")

        self.assertEqual(content, "Pump manual content")

    @patch('agent.ClientContext')
    @patch('agent.AuthenticationContext')
    def test_get_document_not_found(self, mock_auth_context, mock_client_context):
        mock_ctx = MagicMock()
        mock_client_context.return_value = mock_ctx
        mock_ctx.web.lists.get_by_title.return_value.root_folder.files = []

        agent = IndustrialEquipmentAgent("url", "id", "secret")
        result = agent.get_document("Manuals", "Pump-123.pdf")

        self.assertTrue(result.endswith("not found in library 'Manuals'."))

if __name__ == '__main__':
    unittest.main()
