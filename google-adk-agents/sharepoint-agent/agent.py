import adk
import os
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

class SharePointAgent(adk.Agent):
    def __init__(self, site_url, client_id, client_secret):
        super().__init__()
        self.site_url = site_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.ctx_auth = AuthenticationContext(self.site_url)
        self.ctx_auth.acquire_token_for_app(client_id=self.client_id, client_secret=self.client_secret)
        self.ctx = ClientContext(self.site_url, self.ctx_auth)

    def call(self, prompt: str) -> str:
        """
        Parses a prompt to read a file from SharePoint.
        Example prompt: "Analyze the 'Data Architecture' document in the 'Shared Documents' library."
        """
        try:
            # This is a very basic parser. A more sophisticated implementation
            # would use a language model to extract the file name and library.
            parts = prompt.split("'")
            file_name = parts[1]
            library_name = parts[3]

            library = self.ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            file_content = None
            for file in files:
                if file.properties["Name"] == file_name:
                    file_content_response = file.open_binary(self.ctx)
                    self.ctx.execute_query()
                    file_content = file_content_response.value.decode('utf-8')
                    break

            if file_content:
                # In a real implementation, you would perform some analysis on the file content.
                # For now, we'll just return the first 200 characters.
                return f"Successfully read file '{file_name}'.\n\nContent preview:\n{file_content[:200]}..."
            else:
                return f"File '{file_name}' not found in library '{library_name}'."

        except Exception as e:
            return f"Error reading file: {e}"

if __name__ == "__main__":
    # Example usage:
    # You would need to set these environment variables with your SharePoint details.
    site_url = os.environ.get("SHAREPOINT_SITE_URL")
    client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
    client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")

    if not all([site_url, client_id, client_secret]):
        print("Please set the SHAREPOINT_SITE_URL, SHAREPOINT_CLIENT_ID, and SHAREPOINT_CLIENT_SECRET environment variables.")
    else:
        agent = SharePointAgent(site_url, client_id, client_secret)
        result = agent.call("Analyze the 'Data Architecture' document in the 'Shared Documents' library.")
        print(result)
