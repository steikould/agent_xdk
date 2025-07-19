import adk
import os
from jira import JIRA

class JiraAgent(adk.Agent):
    def __init__(self, server_url, username, api_token, project_key):
        super().__init__()
        self.server_url = server_url
        self.username = username
        self.api_token = api_token
        self.project_key = project_key
        self.jira = JIRA(server=self.server_url, basic_auth=(self.username, self.api_token))

    def call(self, prompt: str) -> str:
        """
        Parses a prompt to create an issue in Jira.
        Example prompt: "Create a new story for 'Implement user authentication'."
        """
        try:
            # This is a very basic parser. A more sophisticated implementation
            # would use a language model to extract the issue type and summary.
            parts = prompt.split("'")
            summary = parts[1]
            issue_type = prompt.split(" for ")[0].split(" a new ")[1].capitalize()

            issue_dict = {
                'project': {'key': self.project_key},
                'summary': summary,
                'issuetype': {'name': issue_type},
            }
            new_issue = self.jira.create_issue(fields=issue_dict)
            return f"Successfully created issue {new_issue.key} of type '{issue_type}' with summary '{summary}'."
        except Exception as e:
            return f"Error creating issue: {e}"

if __name__ == "__main__":
    # Example usage:
    # You would need to set these environment variables with your Jira details.
    server_url = os.environ.get("JIRA_SERVER_URL")
    username = os.environ.get("JIRA_USERNAME")
    api_token = os.environ.get("JIRA_API_TOKEN")
    project_key = os.environ.get("JIRA_PROJECT_KEY")

    if not all([server_url, username, api_token, project_key]):
        print("Please set the JIRA_SERVER_URL, JIRA_USERNAME, JIRA_API_TOKEN, and JIRA_PROJECT_KEY environment variables.")
    else:
        agent = JiraAgent(server_url, username, api_token, project_key)
        result = agent.call("Create a new story for 'Implement user authentication'.")
        print(result)
