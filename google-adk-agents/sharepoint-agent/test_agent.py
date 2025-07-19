import unittest
from unittest.mock import patch, MagicMock
from agent import SharePointAgent

class TestSharePointAgent(unittest.TestCase):

    @patch('agent.ClientContext')
    @patch('agent.AuthenticationContext')
    def test_read_file_success(self, mock_auth_context, mock_client_context):
        # This is a simplified test that mocks the SharePoint client.
        # A more comprehensive test would involve more detailed mocking of the SharePoint objects.
        mock_ctx = MagicMock()
        mock_client_context.return_value = mock_ctx

        mock_file = MagicMock()
        mock_file.properties = {"Name": "Data Architecture.docx"}

        # Mocking the file content response
        mock_file_content_response = MagicMock()
        mock_file_content_response.value = b"This is a test document."
        mock_file.open_binary.return_value = mock_file_content_response

        mock_ctx.web.lists.get_by_title.return_value.root_folder.files = [mock_file]

        agent = SharePointAgent("https://example.sharepoint.com", "id", "secret")
        prompt = "Analyze the 'Data Architecture.docx' document in the 'Shared Documents' library."
        result = agent.call(prompt)

        self.assertTrue(result.startswith("Successfully read file"))

    @patch('agent.ClientContext')
    @patch('agent.AuthenticationContext')
    def test_read_file_not_found(self, mock_auth_context, mock_client_context):
        mock_ctx = MagicMock()
        mock_client_context.return_value = mock_ctx
        mock_ctx.web.lists.get_by_title.return_value.root_folder.files = []

        agent = SharePointAgent("https://example.sharepoint.com", "id", "secret")
        prompt = "Analyze the 'Data Architecture.docx' document in the 'Shared Documents' library."
        result = agent.call(prompt)

        self.assertTrue(result.endswith("not found in library 'Shared Documents'."))

if __name__ == '__main__':
    unittest.main()
