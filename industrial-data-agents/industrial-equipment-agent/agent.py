import adk
import os
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

class IndustrialEquipmentAgent(adk.Agent):
    def __init__(self, site_url, client_id, client_secret):
        super().__init__()
        self.site_url = site_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.ctx = self._connect_to_sharepoint()

    def _connect_to_sharepoint(self):
        try:
            ctx_auth = AuthenticationContext(self.site_url)
            ctx_auth.acquire_token_for_app(client_id=self.client_id, client_secret=self.client_secret)
            ctx = ClientContext(self.site_url, ctx_auth)
            return ctx
        except Exception as e:
            print(f"Error connecting to SharePoint: {e}")
            return None

    def get_document(self, library_name, file_name):
        if not self.ctx:
            return "SharePoint connection not available."
        try:
            library = self.ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            for file in files:
                if file.properties["Name"] == file_name:
                    file_content_response = file.open_binary(self.ctx)
                    self.ctx.execute_query()
                    return file_content_response.value.decode('utf-8')
            return f"File '{file_name}' not found in library '{library_name}'."
        except Exception as e:
            return f"Error reading file from SharePoint: {e}"

    def call(self, prompt: str) -> str:
        # Example prompt: "get 'Equipment Manuals' 'Pump-123.pdf'"
        try:
            parts = prompt.split("'")
            library_name = parts[1]
            file_name = parts[3]
            return self.get_document(library_name, file_name)
        except Exception as e:
            return f"Invalid prompt format. Use: get '<library_name>' '<file_name>'. Error: {e}"

if __name__ == "__main__":
    # Example usage:
    site_url = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_URL")
    client_id = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_ID")
    client_secret = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_SECRET")

    if not all([site_url, client_id, client_secret]):
        print("Please set the required environment variables for the industrial equipment SharePoint.")
    else:
        agent = IndustrialEquipmentAgent(site_url, client_id, client_secret)
        result = agent.call("get 'Maintenance Records' 'Compressor-A45.docx'")
        print(result)
