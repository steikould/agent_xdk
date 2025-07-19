import adk
import os
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from azure.devops.v6_0.work_item_tracking.models import JsonPatchOperation

class AzureDevOpsAgent(adk.Agent):
    def __init__(self, org_url, project_name, personal_access_token):
        super().__init__()
        self.org_url = org_url
        self.project_name = project_name
        self.personal_access_token = personal_access_token
        self.credentials = BasicAuthentication('', self.personal_access_token)
        self.connection = Connection(base_url=self.org_url, creds=self.credentials)
        self.work_item_tracking_client = self.connection.clients.get_work_item_tracking_client()

    def call(self, prompt: str) -> str:
        """
        Parses a prompt to create a work item in Azure DevOps.
        Example prompt: "Create a new user story for 'Implement user authentication'."
        """
        try:
            # This is a very basic parser. A more sophisticated implementation
            # would use a language model to extract the work item type and title.
            parts = prompt.split("'")
            title = parts[1]
            work_item_type = prompt.split(" for ")[0].split(" a new ")[1]

            patch_document = [
                JsonPatchOperation(
                    op="add",
                    path="/fields/System.Title",
                    value=title
                )
            ]

            work_item = self.work_item_tracking_client.create_work_item(
                document=patch_document,
                project=self.project_name,
                type=work_item_type
            )

            return f"Successfully created work item {work_item.id} of type '{work_item_type}' with title '{title}'."
        except Exception as e:
            return f"Error creating work item: {e}"


if __name__ == "__main__":
    # Example usage:
    # You would need to set these environment variables with your Azure DevOps details.
    org_url = os.environ.get("AZURE_DEVOPS_ORG_URL")
    project_name = os.environ.get("AZURE_DEVOPS_PROJECT_NAME")
    personal_access_token = os.environ.get("AZURE_DEVOPS_PAT")

    if not all([org_url, project_name, personal_access_token]):
        print("Please set the AZURE_DEVOPS_ORG_URL, AZURE_DEVOPS_PROJECT_NAME, and AZURE_DEVOPS_PAT environment variables.")
    else:
        agent = AzureDevOpsAgent(org_url, project_name, personal_access_token)
        result = agent.call("Create a new user story for 'Implement user authentication'.")
        print(result)
