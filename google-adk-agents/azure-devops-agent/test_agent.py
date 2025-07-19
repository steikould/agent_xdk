import unittest
from unittest.mock import patch, MagicMock
from agent import AzureDevOpsAgent

class TestAzureDevOpsAgent(unittest.TestCase):

    @patch('agent.Connection')
    def test_create_work_item_success(self, mock_connection):
        mock_work_item_tracking_client = MagicMock()
        mock_connection.return_value.clients.get_work_item_tracking_client.return_value = mock_work_item_tracking_client

        mock_work_item = MagicMock()
        mock_work_item.id = 123
        mock_work_item_tracking_client.create_work_item.return_value = mock_work_item

        agent = AzureDevOpsAgent("https://dev.azure.com/org", "project", "pat")
        prompt = "Create a new user story for 'Implement user authentication'."
        result = agent.call(prompt)

        self.assertEqual(result, "Successfully created work item 123 of type 'user story' with title 'Implement user authentication'.")
        mock_work_item_tracking_client.create_work_item.assert_called_once()

    @patch('agent.Connection')
    def test_create_work_item_failure(self, mock_connection):
        mock_work_item_tracking_client = MagicMock()
        mock_connection.return_value.clients.get_work_item_tracking_client.return_value = mock_work_item_tracking_client
        mock_work_item_tracking_client.create_work_item.side_effect = Exception("Test error")

        agent = AzureDevOpsAgent("https://dev.azure.com/org", "project", "pat")
        prompt = "Create a new user story for 'Implement user authentication'."
        result = agent.call(prompt)

        self.assertTrue(result.startswith("Error creating work item:"))

if __name__ == '__main__':
    unittest.main()
