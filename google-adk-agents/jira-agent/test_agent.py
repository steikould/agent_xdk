import unittest
from unittest.mock import patch, MagicMock
from agent import JiraAgent

class TestJiraAgent(unittest.TestCase):

    @patch('agent.JIRA')
    def test_create_issue_success(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira.return_value = mock_jira_instance

        mock_issue = MagicMock()
        mock_issue.key = "PROJ-123"
        mock_jira_instance.create_issue.return_value = mock_issue

        agent = JiraAgent("https://jira.example.com", "user", "token", "PROJ")
        prompt = "Create a new story for 'Implement user authentication'."
        result = agent.call(prompt)

        self.assertEqual(result, "Successfully created issue PROJ-123 of type 'Story' with summary 'Implement user authentication'.")
        mock_jira_instance.create_issue.assert_called_once()

    @patch('agent.JIRA')
    def test_create_issue_failure(self, mock_jira):
        mock_jira_instance = MagicMock()
        mock_jira.return_value = mock_jira_instance
        mock_jira_instance.create_issue.side_effect = Exception("Test error")

        agent = JiraAgent("https://jira.example.com", "user", "token", "PROJ")
        prompt = "Create a new story for 'Implement user authentication'."
        result = agent.call(prompt)

        self.assertTrue(result.startswith("Error creating issue:"))

if __name__ == '__main__':
    unittest.main()
