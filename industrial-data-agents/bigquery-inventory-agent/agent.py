import adk
import os
from google.cloud import bigquery
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from transformers import pipeline

class BigQueryInventoryAgent(adk.Agent):
    def __init__(self, project_id, sharepoint_site_url, sharepoint_client_id, sharepoint_client_secret, llm_model="gpt2"):
        super().__init__()
        self.bigquery_client = bigquery.Client(project=project_id)

        # SharePoint connection
        self.sharepoint_site_url = sharepoint_site_url
        self.sharepoint_client_id = sharepoint_client_id
        self.sharepoint_client_secret = sharepoint_client_secret
        self.sharepoint_ctx = self._connect_to_sharepoint()

        # LLM
        self.llm = pipeline('text-generation', model=llm_model)

    def _connect_to_sharepoint(self):
        try:
            ctx_auth = AuthenticationContext(self.sharepoint_site_url)
            ctx_auth.acquire_token_for_app(client_id=self.sharepoint_client_id, client_secret=self.sharepoint_client_secret)
            ctx = ClientContext(self.sharepoint_site_url, ctx_auth)
            return ctx
        except Exception as e:
            print(f"Error connecting to SharePoint: {e}")
            return None

    def list_bigquery_inventory(self):
        inventory = {}
        datasets = list(self.bigquery_client.list_datasets())
        for dataset in datasets:
            dataset_id = dataset.dataset_id
            inventory[dataset_id] = []
            tables = list(self.bigquery_client.list_tables(dataset_id))
            for table in tables:
                inventory[dataset_id].append(table.table_id)
        return inventory

    def get_sharepoint_document(self, library_name, file_name):
        if not self.sharepoint_ctx:
            return "SharePoint connection not available."
        try:
            library = self.sharepoint_ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.sharepoint_ctx.load(files)
            self.sharepoint_ctx.execute_query()

            for file in files:
                if file.properties["Name"] == file_name:
                    file_content_response = file.open_binary(self.sharepoint_ctx)
                    self.sharepoint_ctx.execute_query()
                    return file_content_response.value.decode('utf-8')
            return f"File '{file_name}' not found in library '{library_name}'."
        except Exception as e:
            return f"Error reading file from SharePoint: {e}"

    def call(self, prompt: str) -> str:
        # This is a conceptual implementation. A real implementation would
        # have more sophisticated interaction between the components.
        if "inventory" in prompt:
            return str(self.list_bigquery_inventory())
        elif "sharepoint" in prompt:
            # Example prompt: "sharepoint 'Shared Documents' 'MyDoc.docx'"
            parts = prompt.split("'")
            library_name = parts[1]
            file_name = parts[3]
            doc_content = self.get_sharepoint_document(library_name, file_name)
            # Use the LLM to summarize or reason about the document
            llm_prompt = f"Summarize the following document:\n\n{doc_content}"
            summary = self.llm(llm_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
            return summary
        else:
            return self.llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

if __name__ == "__main__":
    # Example usage:
    project_id = os.environ.get("GCP_PROJECT_ID")
    sharepoint_site_url = os.environ.get("SHAREPOINT_SITE_URL")
    sharepoint_client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
    sharepoint_client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")

    if not all([project_id, sharepoint_site_url, sharepoint_client_id, sharepoint_client_secret]):
        print("Please set the required environment variables.")
    else:
        agent = BigQueryInventoryAgent(project_id, sharepoint_site_url, sharepoint_client_id, sharepoint_client_secret)
        # Example 1: Get BigQuery inventory
        inventory_result = agent.call("show me the bigquery inventory")
        print(inventory_result)
        # Example 2: Analyze a SharePoint document
        sharepoint_result = agent.call("analyze sharepoint 'Shared Documents' 'Technical Spec.docx'")
        print(sharepoint_result)
        # Example 3: Interact with the LLM
        llm_result = agent.call("What is the purpose of this agent?")
        print(llm_result)
