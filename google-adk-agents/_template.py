"""
Template for creating new Google ADK agents
Copy this template when creating new agents to ensure consistency
"""
import os
from typing import Optional, Dict, Any, List

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner


class YourSystemAgent:
    """
    A comprehensive [System Name] agent using Google's Agent Development Kit (ADK).
    
    This agent provides [brief description of functionality].
    """
    
    def __init__(self, connection_param1: str, connection_param2: str, **kwargs):
        """
        Initialize the [System Name] Agent.
        
        Args:
            connection_param1 (str): Description of parameter
            connection_param2 (str): Description of parameter
            **kwargs: Additional optional parameters
        """
        self.connection_param1 = connection_param1
        self.connection_param2 = connection_param2
        
        # Initialize system connection
        self.client = self._connect_to_system()
        
        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _connect_to_system(self) -> Any:
        """
        Establish connection to [System Name].
        
        Returns:
            Client instance for the system
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Initialize your system client here
            # client = YourSystemClient(self.connection_param1, self.connection_param2)
            
            # Test connection
            # client.test_connection()
            
            print(f"✅ Successfully connected to [System Name]")
            # return client
            return None  # Replace with actual client
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to [System Name]: {e}")

    def system_operation_1(self, param1: str, param2: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a specific system operation.
        
        Args:
            param1 (str): Description
            param2 (str, optional): Description
            
        Returns:
            Dict containing operation results
        """
        try:
            # Implement your system operation
            result = {
                "success": True,
                "data": f"Operation completed with {param1}"
            }
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Operation failed: {e}"
            }

    def system_operation_2(self, query: str) -> List[Dict[str, Any]]:
        """
        Another system operation (e.g., search).
        
        Args:
            query (str): Search query
            
        Returns:
            List of results
        """
        try:
            # Implement search/list operation
            results = [
                {"id": "1", "name": "Result 1"},
                {"id": "2", "name": "Result 2"}
            ]
            return results
            
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with tools."""
        
        def tool_1(param1: str, param2: str = "default") -> str:
            """
            First tool description - what it does.
            
            Args:
                param1 (str): Description of parameter
                param2 (str): Description with default value
                
            Returns:
                str: Formatted response message
            """
            result = self.system_operation_1(param1, param2)
            
            if result["success"]:
                response = f"✅ Successfully completed operation\n"
                response += f"📝 Result: {result['data']}\n"
                return response
            else:
                return f"❌ {result['error']}"

        def tool_2(search_query: str) -> str:
            """
            Second tool description - search functionality.
            
            Args:
                search_query (str): What to search for
                
            Returns:
                str: Formatted search results
            """
            results = self.system_operation_2(search_query)
            
            if results and "error" not in results[0]:
                response = f"🔍 Found {len(results)} results:\n\n"
                for item in results:
                    response += f"• {item['name']} (ID: {item['id']})\n"
                return response
            elif results and "error" in results[0]:
                return f"❌ {results[0]['error']}"
            else:
                return "🔍 No results found"

        def get_help_tool() -> str:
            """
            Get help information about available commands.
            
            Returns:
                str: Help information
            """
            return """
🆘 **[System Name] Agent Help**

🎯 **Available Operations:**

1. **Tool 1 Name**
   - Description of what it does
   - Example: "Do operation with parameter X"

2. **Tool 2 Name**
   - Description of search/list functionality
   - Example: "Search for items matching 'query'"

💡 **Tips:**
- Tip 1 about using the agent
- Tip 2 about best practices
- Tip 3 about common use cases

📝 **Examples:**
- "Create a new item with name 'Test'"
- "Search for all items created today"
- "Get help" (shows this message)
"""

        # Create the agent
        agent = Agent(
            name="your_system_agent",
            model="gemini-2.0-flash-exp",
            description="A comprehensive [System Name] integration agent",
            instruction="""You are a [System Name] specialist agent with the following capabilities:

🎯 **Core Functions:**
- Function 1: What it does
- Function 2: What it does
- Function 3: What it does

💡 **How to Use Me:**
1. Natural language requests are supported
2. I'll help you navigate [System Name] operations
3. Ask for help anytime to see available commands

🔧 **Technical Details:**
- Connected to: [System details]
- Capabilities: [List key capabilities]
- Limitations: [Any limitations to be aware of]

How can I help you with [System Name] today?""",
            tools=[
                tool_1,
                tool_2,
                get_help_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "your_system_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the agent using ADK's session and runner system.
        
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
    """Example usage of the [System Name] Agent."""
    
    # Get configuration from environment variables
    param1 = os.environ.get("YOUR_SYSTEM_PARAM1")
    param2 = os.environ.get("YOUR_SYSTEM_PARAM2")
    
    if not all([param1, param2]):
        print("❌ Please set the required environment variables:")
        print("   - YOUR_SYSTEM_PARAM1")
        print("   - YOUR_SYSTEM_PARAM2")
        return

    print("🚀 Initializing [System Name] Agent with Google ADK...")
    
    try:
        # Create the agent
        agent = YourSystemAgent(param1, param2)
        
        # Test conversations
        test_messages = [
            "Get help",
            "Perform operation with test parameter",
            "Search for recent items",
            "What can you do?"
        ]
        
        print("\n" + "="*80)
        print("🧪 TESTING [SYSTEM NAME] AGENT WITH GOOGLE ADK")
        print("="*80)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n[Test {i}] 💬 User: {message}")
            print("-" * 60)
            
            response = agent.chat_sync(message)
            print(f"🤖 Agent: {response}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")


if __name__ == "__main__":
    main()