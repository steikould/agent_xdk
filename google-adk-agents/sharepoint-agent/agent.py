import os
import re
import mimetypes
from typing import Optional, Dict, List, Any
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from office365.sharepoint.lists.list import List as SPList

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner


class SharePointAgent:
    """
    A comprehensive SharePoint document management agent using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self, site_url: str, client_id: str, client_secret: str):
        """
        Initialize the SharePoint Agent.
        
        Args:
            site_url (str): The SharePoint site URL
            client_id (str): The client ID for app authentication
            client_secret (str): The client secret for app authentication
        """
        self.site_url = site_url
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Initialize SharePoint connection
        self.ctx = self._connect_to_sharepoint()
        
        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _connect_to_sharepoint(self) -> ClientContext:
        """
        Establish connection to SharePoint.
        
        Returns:
            ClientContext: The SharePoint client context
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Initialize authentication context
            ctx_auth = AuthenticationContext(self.site_url)
            token_acquired = ctx_auth.acquire_token_for_app(
                client_id=self.client_id, 
                client_secret=self.client_secret
            )
            
            if not token_acquired:
                raise ConnectionError("Failed to acquire authentication token")
            
            # Initialize client context
            ctx = ClientContext(self.site_url, ctx_auth)
            
            # Test connection
            web = ctx.web
            ctx.load(web)
            ctx.execute_query()
            
            print(f"✅ Successfully connected to SharePoint site: {web.properties['Title']}")
            return ctx
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to SharePoint: {e}")

    def _get_library(self, library_name: str) -> SPList:
        """
        Get a SharePoint library by name.
        
        Args:
            library_name (str): The name of the library
            
        Returns:
            SPList: The SharePoint list object
        """
        try:
            library = self.ctx.web.lists.get_by_title(library_name)
            self.ctx.load(library)
            self.ctx.execute_query()
            return library
        except Exception as e:
            raise ValueError(f"Library '{library_name}' not found: {e}")

    def read_file_content(self, file_name: str, library_name: str) -> str:
        """
        Read content from a SharePoint file.
        
        Args:
            file_name (str): The name of the file
            library_name (str): The name of the library
            
        Returns:
            str: The file content or error message
        """
        try:
            library = self._get_library(library_name)
            
            # Try exact match first
            file_url = f"{library_name}/{file_name}"
            try:
                file = self.ctx.web.get_file_by_server_relative_url(file_url)
                self.ctx.load(file)
                self.ctx.execute_query()
                
                # Download file content
                file_content = file.download(self.ctx).content
                
                # Try to decode as text
                try:
                    content = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = file_content.decode('utf-8-sig')  # Handle BOM
                    except UnicodeDecodeError:
                        content = f"[Binary file - {len(file_content)} bytes]"
                
                return content
                
            except Exception:
                # If exact match fails, search through files
                files = library.root_folder.files
                self.ctx.load(files)
                self.ctx.execute_query()
                
                for file in files:
                    if file.properties["Name"].lower() == file_name.lower():
                        file_content = file.download(self.ctx).content
                        try:
                            return file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                return file_content.decode('utf-8-sig')
                            except UnicodeDecodeError:
                                return f"[Binary file - {len(file_content)} bytes]"
                
                raise FileNotFoundError(f"File '{file_name}' not found")
                
        except Exception as e:
            raise Exception(f"Error reading file '{file_name}': {e}")

    def list_files(self, library_name: str, max_files: int = 50) -> List[Dict[str, Any]]:
        """
        List files in a SharePoint library.
        
        Args:
            library_name (str): The name of the library
            max_files (int): Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        try:
            library = self._get_library(library_name)
            files = library.root_folder.files
            self.ctx.load(files)
            self.ctx.execute_query()
            
            file_list = []
            for i, file in enumerate(files):
                if i >= max_files:
                    break
                    
                file_info = {
                    'name': file.properties.get("Name", "Unknown"),
                    'size': file.properties.get("Length", 0),
                    'modified': file.properties.get("TimeLastModified", "Unknown"),
                    'url': file.properties.get("ServerRelativeUrl", ""),
                    'created': file.properties.get("TimeCreated", "Unknown"),
                    'author': file.properties.get("Author", "Unknown")
                }
                file_list.append(file_info)
            
            return file_list
            
        except Exception as e:
            raise Exception(f"Error listing files in '{library_name}': {e}")

    def search_files(self, search_term: str, library_name: str) -> List[Dict[str, Any]]:
        """
        Search for files containing a term in their name.
        
        Args:
            search_term (str): The term to search for
            library_name (str): The name of the library
            
        Returns:
            List of matching file information dictionaries
        """
        try:
            all_files = self.list_files(library_name, max_files=200)
            matching_files = []
            
            for file_info in all_files:
                if search_term.lower() in file_info['name'].lower():
                    matching_files.append(file_info)
            
            return matching_files
            
        except Exception as e:
            raise Exception(f"Error searching files: {e}")

    def get_library_info(self, library_name: str) -> Dict[str, Any]:
        """
        Get information about a SharePoint library.
        
        Args:
            library_name (str): The name of the library
            
        Returns:
            Dict containing library information
        """
        try:
            library = self._get_library(library_name)
            
            return {
                'title': library.properties.get('Title', 'Unknown'),
                'description': library.properties.get('Description', 'No description'),
                'item_count': library.properties.get('ItemCount', 'Unknown'),
                'created': library.properties.get('Created', 'Unknown'),
                'last_modified': library.properties.get('LastItemModifiedDate', 'Unknown'),
                'template_type': library.properties.get('BaseTemplate', 'Unknown')
            }
            
        except Exception as e:
            raise Exception(f"Error getting library info: {e}")

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with SharePoint tools."""
        
        def read_file_tool(file_name: str, library_name: str = "Shared Documents") -> str:
            """
            Read and preview a file from SharePoint.
            
            Args:
                file_name (str): Name of the file to read
                library_name (str): Name of the SharePoint library (default: "Shared Documents")
                
            Returns:
                str: File content preview or error message
            """
            try:
                content = self.read_file_content(file_name, library_name)
                
                if content.startswith("[Binary file"):
                    return f"📁 Successfully accessed binary file '{file_name}' from '{library_name}'.\n\n{content}"
                
                # Provide content preview
                preview_length = min(1000, len(content))
                preview = content[:preview_length]
                if len(content) > preview_length:
                    preview += "..."
                
                return f"📄 Successfully read file '{file_name}' from '{library_name}'\n" \
                       f"Content length: {len(content):,} characters\n\n" \
                       f"Content preview:\n{preview}"
                       
            except Exception as e:
                return f"❌ Error reading file '{file_name}': {e}"

        def list_files_tool(library_name: str = "Shared Documents", max_files: str = "20") -> str:
            """
            List files in a SharePoint library.
            
            Args:
                library_name (str): Name of the SharePoint library (default: "Shared Documents")
                max_files (str): Maximum number of files to return (default: "20")
                
            Returns:
                str: Formatted list of files or error message
            """
            try:
                max_files_int = int(max_files)
                files = self.list_files(library_name, max_files=max_files_int)
                
                if not files:
                    return f"📁 No files found in library '{library_name}'"
                
                result = f"📁 Files in '{library_name}' ({len(files)} files shown):\n\n"
                for file_info in files:
                    size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
                    result += f"📄 {file_info['name']}\n"
                    result += f"   • Size: {size_mb:.2f} MB\n"
                    result += f"   • Modified: {file_info['modified']}\n"
                    result += f"   • Created: {file_info['created']}\n\n"
                
                return result.strip()
                
            except Exception as e:
                return f"❌ Error listing files in '{library_name}': {e}"

        def search_files_tool(search_term: str, library_name: str = "Shared Documents") -> str:
            """
            Search for files in a SharePoint library.
            
            Args:
                search_term (str): Term to search for in file names
                library_name (str): Name of the SharePoint library (default: "Shared Documents")
                
            Returns:
                str: Search results or error message
            """
            try:
                matching_files = self.search_files(search_term, library_name)
                
                if not matching_files:
                    return f"🔍 No files found matching '{search_term}' in '{library_name}'"
                
                result = f"🔍 Files matching '{search_term}' in '{library_name}' ({len(matching_files)} found):\n\n"
                for file_info in matching_files:
                    size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
                    result += f"📄 {file_info['name']}\n"
                    result += f"   • Size: {size_mb:.2f} MB\n"
                    result += f"   • Modified: {file_info['modified']}\n\n"
                
                return result.strip()
                
            except Exception as e:
                return f"❌ Error searching files: {e}"

        def get_library_info_tool(library_name: str = "Shared Documents") -> str:
            """
            Get information about a SharePoint library.
            
            Args:
                library_name (str): Name of the SharePoint library (default: "Shared Documents")
                
            Returns:
                str: Library information or error message
            """
            try:
                info = self.get_library_info(library_name)
                
                result = f"📚 Library Information: '{library_name}'\n\n"
                result += f"• Title: {info['title']}\n"
                result += f"• Description: {info['description']}\n"
                result += f"• Item Count: {info['item_count']}\n"
                result += f"• Created: {info['created']}\n"
                result += f"• Last Modified: {info['last_modified']}\n"
                result += f"• Template Type: {info['template_type']}\n"
                
                return result
                
            except Exception as e:
                return f"❌ Error getting library info: {e}"

        def analyze_file_tool(file_name: str, library_name: str = "Shared Documents") -> str:
            """
            Analyze a file's content and provide insights.
            
            Args:
                file_name (str): Name of the file to analyze
                library_name (str): Name of the SharePoint library (default: "Shared Documents")
                
            Returns:
                str: File analysis or error message
            """
            try:
                content = self.read_file_content(file_name, library_name)
                
                if content.startswith("[Binary file"):
                    return f"📊 File Analysis: '{file_name}'\n\n" \
                           f"Type: Binary file\n" \
                           f"Cannot analyze content of binary files.\n" \
                           f"{content}"
                
                # Basic content analysis
                lines = content.split('\n')
                words = content.split()
                chars = len(content)
                
                # Try to determine file type
                file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                
                result = f"📊 File Analysis: '{file_name}'\n\n"
                result += f"📁 Library: {library_name}\n"
                result += f"📄 File Extension: .{file_ext}\n"
                result += f"📏 Statistics:\n"
                result += f"   • Characters: {chars:,}\n"
                result += f"   • Words: {len(words):,}\n"
                result += f"   • Lines: {len(lines):,}\n\n"
                
                # Content preview
                preview_length = min(500, len(content))
                preview = content[:preview_length]
                if len(content) > preview_length:
                    preview += "..."
                
                result += f"📋 Content Preview:\n{preview}\n\n"
                
                # Simple insights
                if file_ext in ['txt', 'md']:
                    result += f"💡 Insights: This appears to be a text document.\n"
                elif file_ext in ['csv']:
                    result += f"💡 Insights: This appears to be a CSV data file.\n"
                elif file_ext in ['json']:
                    result += f"💡 Insights: This appears to be a JSON data file.\n"
                elif file_ext in ['xml']:
                    result += f"💡 Insights: This appears to be an XML document.\n"
                else:
                    result += f"💡 Insights: File type: {file_ext.upper()}\n"
                
                return result
                
            except Exception as e:
                return f"❌ Error analyzing file '{file_name}': {e}"

        # Create the agent with comprehensive SharePoint capabilities
        agent = Agent(
            name="sharepoint_document_agent",
            model="gemini-2.0-flash-exp",
            description="A comprehensive SharePoint document management agent with advanced file operations",
            instruction="""You are a SharePoint specialist agent with extensive capabilities for document management and analysis. 

🔧 **Your Capabilities:**

📄 **File Operations:**
- Read and preview documents from SharePoint libraries
- Analyze file content and provide insights
- Handle both text and binary files gracefully

📁 **Library Management:**
- List files in any SharePoint library with detailed metadata
- Search for files by name patterns
- Get comprehensive library information and statistics

🔍 **Search & Discovery:**
- Find files matching specific search terms
- Browse library contents with size and modification details
- Identify file types and content characteristics

📊 **Analysis Features:**
- Content analysis with word/line/character counts
- File type detection and insights
- Content previews for quick understanding

💡 **Usage Tips:**
- Default library is "Shared Documents" if not specified
- I can handle various file formats (text, CSV, JSON, XML, etc.)
- Search is case-insensitive and matches partial file names
- Binary files are detected and handled appropriately

🌟 **Example Commands:**
- "Read the file 'report.txt' from 'Documents'"
- "List files in 'Shared Documents'"
- "Search for 'budget' files in 'Finance Library'"
- "Analyze the content of 'data.csv'"
- "Get information about 'Project Files' library"

How can I help you with your SharePoint documents today?""",
            tools=[
                read_file_tool,
                list_files_tool,
                search_files_tool,
                get_library_info_tool,
                analyze_file_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "sharepoint_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the SharePoint agent using ADK's session and runner system.
        
        Args:
            message (str): User message
            app_name (str): Application name for session management
            user_id (str): User identifier
            session_id (str): Session identifier
            
        Returns:
            str: Agent response
        """
        try:
            # Create session service and session
            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name=app_name, 
                user_id=user_id, 
                session_id=session_id
            )
            
            # Create runner
            runner = Runner(
                agent=self.agent, 
                app_name=app_name, 
                session_service=session_service
            )
            
            # Create content for the message
            content = {
                'role': 'user',
                'parts': [{'text': message}]
            }
            
            # Run the agent
            events = runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=content
            )
            
            # Collect the response
            response_parts = []
            async for event in events:
                if event.is_final_response():
                    for part in event.content.parts:
                        if hasattr(part, 'text'):
                            response_parts.append(part.text)
            
            return '\n'.join(response_parts) if response_parts else "No response generated."
            
        except Exception as e:
            return f"❌ Error during chat: {e}"

    def chat_sync(self, message: str) -> str:
        """
        Synchronous wrapper for chat method.
        
        Args:
            message (str): User message
            
        Returns:
            str: Agent response
        """
        import asyncio
        
        try:
            # Create new event loop if none exists
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.chat(message))
        except Exception as e:
            return f"❌ Error: {e}"


def main():
    """Example usage of the SharePoint Agent with Google ADK."""
    
    # Get configuration from environment variables
    site_url = os.environ.get("SHAREPOINT_SITE_URL")
    client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
    client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")
    
    if not all([site_url, client_id, client_secret]):
        print("❌ Please set the required environment variables:")
        print("   - SHAREPOINT_SITE_URL")
        print("   - SHAREPOINT_CLIENT_ID") 
        print("   - SHAREPOINT_CLIENT_SECRET")
        return

    print("🚀 Initializing SharePoint Agent with Google ADK...")
    
    try:
        # Create the agent
        agent = SharePointAgent(site_url, client_id, client_secret)
        
        # Test conversations
        test_messages = [
            "List files in 'Shared Documents'",
            "Get information about 'Shared Documents' library",
            "Search for 'report' in 'Shared Documents'",
            "Read the file 'meeting-notes.txt'",
            "Analyze the content of 'data.csv' from 'Documents'",
            "What are your capabilities?"
        ]
        
        print("\n" + "="*80)
        print("🧪 TESTING SHAREPOINT AGENT WITH GOOGLE ADK")
        print("="*80)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n[Test {i}] 💬 User: {message}")
            print("-" * 60)
            
            response = agent.chat_sync(message)
            print(f"🤖 Agent: {response}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to initialize SharePoint agent: {e}")


if __name__ == "__main__":
    main()