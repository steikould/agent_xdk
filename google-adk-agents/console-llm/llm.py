import sys
import os
import asyncio
from typing import Optional, Dict, Any
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk import types


class ConsoleLLM:
    """
    An interactive console application that combines Google ADK with SharePoint integration
    for document management and AI-powered conversations.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Console LLM with Google ADK.
        
        Args:
            model_name (str): The name of the model to use (default: gemini-2.0-flash-exp)
        """
        self.model_name = model_name
        self.sharepoint_ctx = None
        self.sharepoint_connected = False
        self.current_session_id = "console_session_001"
        
        # Initialize the ADK agent
        self.agent = self._create_agent()
        
        # Initialize session service
        self.session_service = InMemorySessionService()
        
        print(f"✅ Console LLM initialized with Google ADK")
        print(f"🤖 Using model: {model_name}")

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with SharePoint and text generation capabilities."""
        
        def connect_sharepoint_tool(site_url: str, client_id: str, client_secret: str) -> str:
            """
            Connect to SharePoint using the provided credentials.
            
            Args:
                site_url (str): The SharePoint site URL
                client_id (str): The client ID for authentication
                client_secret (str): The client secret for authentication
                
            Returns:
                str: Connection result message
            """
            try:
                # Initialize authentication context
                ctx_auth = AuthenticationContext(site_url)
                token_acquired = ctx_auth.acquire_token_for_app(
                    client_id=client_id, 
                    client_secret=client_secret
                )
                
                if not token_acquired:
                    return "❌ Failed to acquire SharePoint authentication token"
                
                # Initialize client context
                ctx = ClientContext(site_url, ctx_auth)
                
                # Test connection
                web = ctx.web
                ctx.load(web)
                ctx.execute_query()
                
                # Store connection
                self.sharepoint_ctx = ctx
                self.sharepoint_connected = True
                
                return f"✅ Successfully connected to SharePoint site: {web.properties.get('Title', 'Unknown Site')}"
                
            except Exception as e:
                self.sharepoint_connected = False
                return f"❌ Failed to connect to SharePoint: {e}"

        def disconnect_sharepoint_tool() -> str:
            """
            Disconnect from SharePoint.
            
            Returns:
                str: Disconnection message
            """
            self.sharepoint_ctx = None
            self.sharepoint_connected = False
            return "✅ Disconnected from SharePoint"

        def sharepoint_status_tool() -> str:
            """
            Get SharePoint connection status.
            
            Returns:
                str: Connection status information
            """
            if self.sharepoint_connected and self.sharepoint_ctx:
                try:
                    web = self.sharepoint_ctx.web
                    return f"🟢 SharePoint Status: Connected\n" \
                           f"📍 Site: {web.properties.get('Title', 'Unknown')}\n" \
                           f"🔗 URL: {web.properties.get('Url', 'Unknown')}"
                except:
                    return "🟡 SharePoint Status: Connection may be stale"
            else:
                return "🔴 SharePoint Status: Not connected"

        def read_sharepoint_file_tool(file_name: str, library_name: str = "Shared Documents") -> str:
            """
            Read a file from SharePoint.
            
            Args:
                file_name (str): Name of the file to read
                library_name (str): Name of the SharePoint library
                
            Returns:
                str: File content or error message
            """
            if not self.sharepoint_connected:
                return "❌ Not connected to SharePoint. Use connect_sharepoint_tool first."
            
            try:
                library = self.sharepoint_ctx.web.lists.get_by_title(library_name)
                files = library.root_folder.files
                self.sharepoint_ctx.load(files)
                self.sharepoint_ctx.execute_query()

                for file in files:
                    if file.properties["Name"].lower() == file_name.lower():
                        file_content = file.download(self.sharepoint_ctx).content
                        
                        # Try to decode as text
                        try:
                            content = file_content.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                content = file_content.decode('utf-8-sig')
                            except UnicodeDecodeError:
                                return f"📁 Binary file detected: {file_name} ({len(file_content)} bytes)"
                        
                        # Provide content preview
                        preview_length = min(1500, len(content))
                        preview = content[:preview_length]
                        if len(content) > preview_length:
                            preview += "..."
                        
                        return f"📄 File: {file_name} (from {library_name})\n" \
                               f"📊 Size: {len(content):,} characters\n\n" \
                               f"Content:\n{preview}"
                
                return f"❌ File '{file_name}' not found in library '{library_name}'"
                
            except Exception as e:
                return f"❌ Error reading SharePoint file: {e}"

        def list_sharepoint_files_tool(library_name: str = "Shared Documents", max_files: str = "20") -> str:
            """
            List files in a SharePoint library.
            
            Args:
                library_name (str): Name of the SharePoint library
                max_files (str): Maximum number of files to list
                
            Returns:
                str: List of files or error message
            """
            if not self.sharepoint_connected:
                return "❌ Not connected to SharePoint. Use connect_sharepoint_tool first."
            
            try:
                max_files_int = int(max_files)
                library = self.sharepoint_ctx.web.lists.get_by_title(library_name)
                files = library.root_folder.files
                self.sharepoint_ctx.load(files)
                self.sharepoint_ctx.execute_query()
                
                if not files:
                    return f"📁 No files found in library '{library_name}'"
                
                result = f"📁 Files in '{library_name}' (showing up to {max_files_int}):\n\n"
                
                for i, file in enumerate(files):
                    if i >= max_files_int:
                        break
                    
                    name = file.properties.get("Name", "Unknown")
                    size = file.properties.get("Length", 0)
                    modified = file.properties.get("TimeLastModified", "Unknown")
                    size_mb = size / (1024 * 1024) if size > 0 else 0
                    
                    result += f"📄 {name}\n"
                    result += f"   📊 Size: {size_mb:.2f} MB\n"
                    result += f"   📅 Modified: {modified}\n\n"
                
                if len(files) > max_files_int:
                    result += f"... and {len(files) - max_files_int} more files"
                
                return result.strip()
                
            except Exception as e:
                return f"❌ Error listing SharePoint files: {e}"

        def search_sharepoint_files_tool(search_term: str, library_name: str = "Shared Documents") -> str:
            """
            Search for files in SharePoint by name.
            
            Args:
                search_term (str): Term to search for in file names
                library_name (str): Name of the SharePoint library
                
            Returns:
                str: Search results or error message
            """
            if not self.sharepoint_connected:
                return "❌ Not connected to SharePoint. Use connect_sharepoint_tool first."
            
            try:
                library = self.sharepoint_ctx.web.lists.get_by_title(library_name)
                files = library.root_folder.files
                self.sharepoint_ctx.load(files)
                self.sharepoint_ctx.execute_query()
                
                matching_files = []
                for file in files:
                    name = file.properties.get("Name", "")
                    if search_term.lower() in name.lower():
                        matching_files.append({
                            'name': name,
                            'size': file.properties.get("Length", 0),
                            'modified': file.properties.get("TimeLastModified", "Unknown")
                        })
                
                if not matching_files:
                    return f"🔍 No files found matching '{search_term}' in '{library_name}'"
                
                result = f"🔍 Found {len(matching_files)} file(s) matching '{search_term}':\n\n"
                
                for file_info in matching_files:
                    size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
                    result += f"📄 {file_info['name']}\n"
                    result += f"   📊 Size: {size_mb:.2f} MB\n"
                    result += f"   📅 Modified: {file_info['modified']}\n\n"
                
                return result.strip()
                
            except Exception as e:
                return f"❌ Error searching SharePoint files: {e}"

        def get_help_tool() -> str:
            """
            Get help information about available commands.
            
            Returns:
                str: Help information
            """
            help_text = """
🆘 **Console LLM Help - Available Commands**

🔗 **SharePoint Connection:**
- Connect to SharePoint: "Connect to SharePoint with URL [url], client ID [id], and secret [secret]"
- Check status: "What's my SharePoint connection status?"
- Disconnect: "Disconnect from SharePoint"

📁 **SharePoint Operations:**
- List files: "List files in 'Shared Documents'"
- Read file: "Read file 'document.txt' from 'Documents'"
- Search files: "Search for 'report' in 'Shared Documents'"

💬 **General Conversation:**
- Ask any question or have a conversation
- Get explanations, analysis, or creative content
- The AI will automatically choose SharePoint tools when needed

🌟 **Smart Features:**
- Automatic SharePoint operation detection
- Context-aware responses
- Rich formatting and emojis
- Error handling and helpful suggestions

📝 **Example Conversations:**
- "Connect to my SharePoint site"
- "Show me all documents in the main library"
- "Read the latest meeting notes"
- "Find all files with 'budget' in the name"
- "Explain quantum computing to me"
- "Help me write a project proposal"

Type 'exit' to end the session.
            """
            return help_text

        # Create the agent with all tools
        agent = Agent(
            name="console_llm_agent",
            model=self.model_name,
            description="An interactive console AI assistant with SharePoint integration capabilities",
            instruction="""You are an intelligent console assistant that combines general AI conversation capabilities with SharePoint document management.

🎯 **Your Primary Capabilities:**

1. **SharePoint Integration:**
   - Connect to SharePoint sites using credentials
   - Read, list, and search documents
   - Provide file content analysis and insights
   - Handle both text and binary files gracefully

2. **General AI Assistant:**
   - Answer questions on any topic
   - Provide explanations and analysis
   - Help with writing, coding, and creative tasks
   - Engage in natural conversations

3. **Smart Operation Detection:**
   - Automatically detect when users want SharePoint operations
   - Provide helpful guidance for SharePoint commands
   - Seamlessly switch between SharePoint tasks and general conversation

🔧 **Usage Patterns:**
- When users mention SharePoint, documents, files, or libraries, prioritize SharePoint tools
- For general questions, provide direct helpful responses
- Always check SharePoint connection status before attempting operations
- Provide clear, actionable error messages and suggestions

🌟 **Communication Style:**
- Be helpful, friendly, and informative
- Use emojis for visual clarity and engagement
- Provide step-by-step guidance when needed
- Offer proactive suggestions and tips

Remember: You're both a SharePoint specialist AND a general AI assistant. Adapt your responses based on what the user needs!""",
            tools=[
                connect_sharepoint_tool,
                disconnect_sharepoint_tool,
                sharepoint_status_tool,
                read_sharepoint_file_tool,
                list_sharepoint_files_tool,
                search_sharepoint_files_tool,
                get_help_tool
            ]
        )
        
        return agent

    async def chat_async(self, message: str) -> str:
        """
        Chat with the agent asynchronously.
        
        Args:
            message (str): User message
            
        Returns:
            str: Agent response
        """
        try:
            # Create or get session
            session = await self.session_service.create_session(
                app_name="console_llm",
                user_id="console_user",
                session_id=self.current_session_id
            )
            
            # Create runner
            runner = Runner(
                agent=self.agent,
                app_name="console_llm",
                session_service=self.session_service
            )
            
            # Create content
            content = types.Content(
                role='user',
                parts=[types.Part(text=message)]
            )
            
            # Run the agent
            events = runner.run_async(
                user_id="console_user",
                session_id=self.current_session_id,
                new_message=content
            )
            
            # Collect response
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
        try:
            # Get or create event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.chat_async(message))
        except Exception as e:
            return f"❌ Error: {e}"

    def process_special_commands(self, user_input: str) -> Optional[str]:
        """
        Process special console commands that don't need the agent.
        
        Args:
            user_input (str): User input
            
        Returns:
            Optional[str]: Response if it's a special command, None otherwise
        """
        input_lower = user_input.lower().strip()
        
        if input_lower == 'exit':
            return "EXIT_COMMAND"
        elif input_lower in ['help', 'h', '?']:
            return self.chat_sync("Show me help information")
        elif input_lower in ['status', 'st']:
            return self.chat_sync("What's my SharePoint connection status?")
        elif input_lower == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            return "🧹 Screen cleared!"
        
        return None

    def start_chat(self):
        """Start the interactive chat session."""
        print("\n" + "="*80)
        print("🤖 CONSOLE LLM WITH GOOGLE ADK & SHAREPOINT")
        print("="*80)
        print(f"✨ Welcome! I'm your AI assistant with SharePoint capabilities.")
        print(f"🔗 Model: {self.model_name}")
        print(f"💡 Type 'help' for commands, 'exit' to quit")
        print(f"🚀 Ready to help with SharePoint documents and general questions!")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for special commands
                special_response = self.process_special_commands(user_input)
                if special_response:
                    if special_response == "EXIT_COMMAND":
                        print("\n👋 Goodbye! Thanks for using Console LLM!")
                        break
                    else:
                        print(f"🤖 Assistant: {special_response}")
                        continue
                
                # Process with the agent
                print("🤔 Thinking...")
                response = self.chat_sync(user_input)
                print(f"🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation interrupted. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("💡 Try typing 'help' for assistance or 'exit' to quit.")
                continue

    def run_batch_commands(self, commands: list):
        """
        Run a batch of commands for testing.
        
        Args:
            commands (list): List of commands to execute
        """
        print("\n🧪 Running batch commands for testing...")
        
        for i, command in enumerate(commands, 1):
            print(f"\n[Command {i}] 💬 {command}")
            print("-" * 60)
            response = self.chat_sync(command)
            print(f"🤖 {response}")
            print()


def main():
    """Main function to run the Console LLM."""
    # Parse command line arguments
    model_name = "gemini-2.0-flash-exp"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    # Check for test mode
    test_mode = "--test" in sys.argv
    
    try:
        # Create the console LLM
        console_llm = ConsoleLLM(model_name=model_name)
        
        if test_mode:
            # Run test commands
            test_commands = [
                "Hello! What can you help me with?",
                "Show me help information",
                "What's my SharePoint connection status?",
                "Explain the benefits of cloud storage",
                "How do I connect to SharePoint?"
            ]
            console_llm.run_batch_commands(test_commands)
        else:
            # Start interactive chat
            console_llm.start_chat()
            
    except Exception as e:
        print(f"❌ Failed to initialize Console LLM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()