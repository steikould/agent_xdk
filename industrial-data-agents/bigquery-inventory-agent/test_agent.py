import unittest
from unittest.mock import patch, MagicMock
from agent import BigQueryInventoryAgent

class TestBigQueryInventoryAgent(unittest.TestCase):

    @patch('agent.bigquery.Client')
    @patch('agent.AuthenticationContext')
    @patch('agent.ClientContext')
    @patch('agent.pipeline')
    def test_list_bigquery_inventory(self, mock_pipeline, mock_client_context, mock_auth_context, mock_bigquery_client):
        # Mock BigQuery client
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "my_dataset"
        mock_bigquery_client.return_value.list_datasets.return_value = [mock_dataset]
        mock_table = MagicMock()
        mock_table.table_id = "my_table"
        mock_bigquery_client.return_value.list_tables.return_value = [mock_table]

        agent = BigQueryInventoryAgent("test-project", "url", "id", "secret")
        inventory = agent.list_bigquery_inventory()

        self.assertIn("my_dataset", inventory)
        self.assertIn("my_table", inventory["my_dataset"])

    @patch('agent.bigquery.Client')
    @patch('agent.AuthenticationContext')
    @patch('agent.ClientContext')
    @patch('agent.pipeline')
    def test_get_sharepoint_document_found(self, mock_pipeline, mock_client_context, mock_auth_context, mock_bigquery_client):
        # Mock SharePoint client
        mock_ctx = MagicMock()
        mock_client_context.return_value = mock_ctx
        mock_file = MagicMock()
        mock_file.properties = {"Name": "MyDoc.docx"}
        mock_file_content_response = MagicMock()
        mock_file_content_response.value = b"Test content"
        mock_file.open_binary.return_value = mock_file_content_response
        mock_ctx.web.lists.get_by_title.return_value.root_folder.files = [mock_file]

        agent = BigQueryInventoryAgent("test-project", "url", "id", "secret")
        content = agent.get_sharepoint_document("Docs", "MyDoc.docx")

        self.assertEqual(content, "Test content")

if __name__ == '__main__':
    unittest.main()
