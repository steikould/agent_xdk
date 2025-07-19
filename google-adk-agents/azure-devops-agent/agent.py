import os
import re
from typing import Dict, List, Any, Optional
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from azure.devops.v6_0.work_item_tracking.models import JsonPatchOperation
from azure.devops.v6_0.work_item_tracking.models import WorkItemStateColor

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk import types


class AzureDevOpsAgent:
    """
    A comprehensive Azure DevOps work item management agent using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self, org_url: str, project_name: str, personal_access_token: str):
        """
        Initialize the Azure DevOps Agent.
        
        Args:
            org_url (str): Azure DevOps organization URL
            project_name (str): Project name
            personal_access_token (str): Personal access token
        """
        self.org_url = org_url
        self.project_name = project_name
        self.personal_access_token = personal_access_token
        
        # Initialize Azure DevOps connection
        self.connection = self._connect_to_azure_devops()
        self.work_item_client = self.connection.clients.get_work_item_tracking_client()
        
        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _connect_to_azure_devops(self) -> Connection:
        """
        Establish connection to Azure DevOps.
        
        Returns:
            Connection: The Azure DevOps connection
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            credentials = BasicAuthentication('', self.personal_access_token)
            connection = Connection(base_url=self.org_url, creds=credentials)
            
            # Test connection by getting project info
            core_client = connection.clients.get_core_client()
            project = core_client.get_project(self.project_name)
            
            print(f"✅ Successfully connected to Azure DevOps")
            print(f"   Organization: {self.org_url}")
            print(f"   Project: {project.name} ({project.id})")
            
            return connection
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Azure DevOps: {e}")

    def create_work_item(self, work_item_type: str, title: str, description: str = None, 
                        assigned_to: str = None, priority: int = None, tags: str = None) -> Dict[str, Any]:
        """
        Create a work item in Azure DevOps.
        
        Args:
            work_item_type (str): Type of work item (User Story, Task, Bug, etc.)
            title (str): Work item title
            description (str, optional): Work item description
            assigned_to (str, optional): Assignee email/username
            priority (int, optional): Priority level (1-4)
            tags (str, optional): Comma-separated tags
            
        Returns:
            Dict containing work item creation result
        """
        try:
            # Build patch document
            patch_document = [
                JsonPatchOperation(
                    op="add",
                    path="/fields/System.Title",
                    value=title
                )
            ]
            
            # Add optional fields
            if description:
                patch_document.append(JsonPatchOperation(
                    op="add",
                    path="/fields/System.Description",
                    value=description
                ))
            
            if assigned_to:
                patch_document.append(JsonPatchOperation(
                    op="add",
                    path="/fields/System.AssignedTo",
                    value=assigned_to
                ))
            
            if priority:
                patch_document.append(JsonPatchOperation(
                    op="add",
                    path="/fields/Microsoft.VSTS.Common.Priority",
                    value=priority
                ))
            
            if tags:
                patch_document.append(JsonPatchOperation(
                    op="add",
                    path="/fields/System.Tags",
                    value=tags
                ))
            
            # Create the work item
            work_item = self.work_item_client.create_work_item(
                document=patch_document,
                project=self.project_name,
                type=work_item_type
            )
            
            return {
                "success": True,
                "work_item_id": work_item.id,
                "work_item_type": work_item_type,
                "title": title,
                "description": description,
                "url": work_item.url,
                "web_url": f"{self.org_url}/{self.project_name}/_workitems/edit/{work_item.id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating work item: {e}"
            }

    def get_work_item(self, work_item_id: int) -> Dict[str, Any]:
        """
        Get work item details.
        
        Args:
            work_item_id (int): Work item ID
            
        Returns:
            Dict containing work item information
        """
        try:
            work_item = self.work_item_client.get_work_item(work_item_id)
            
            fields = work_item.fields
            
            return {
                "success": True,
                "id": work_item.id,
                "title": fields.get("System.Title", "No title"),
                "work_item_type": fields.get("System.WorkItemType", "Unknown"),
                "state": fields.get("System.State", "Unknown"),
                "assigned_to": fields.get("System.AssignedTo", {}).get("displayName", "Unassigned") if fields.get("System.AssignedTo") else "Unassigned",
                "created_by": fields.get("System.CreatedBy", {}).get("displayName", "Unknown") if fields.get("System.CreatedBy") else "Unknown",
                "created_date": fields.get("System.CreatedDate", "Unknown"),
                "changed_date": fields.get("System.ChangedDate", "Unknown"),
                "description": fields.get("System.Description", "No description"),
                "priority": fields.get("Microsoft.VSTS.Common.Priority", "Not set"),
                "tags": fields.get("System.Tags", "No tags"),
                "url": work_item.url,
                "web_url": f"{self.org_url}/{self.project_name}/_workitems/edit/{work_item.id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving work item {work_item_id}: {e}"
            }

    def search_work_items(self, query: str = None, work_item_type: str = None, 
                         assigned_to: str = None, state: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for work items using WIQL.
        
        Args:
            query (str, optional): Custom WIQL query
            work_item_type (str, optional): Filter by work item type
            assigned_to (str, optional): Filter by assignee
            state (str, optional): Filter by state
            max_results (int): Maximum number of results
            
        Returns:
            List of work item dictionaries
        """
        try:
            if query:
                wiql_query = query
            else:
                # Build basic query
                conditions = [f"[System.TeamProject] = '{self.project_name}'"]
                
                if work_item_type:
                    conditions.append(f"[System.WorkItemType] = '{work_item_type}'")
                if assigned_to:
                    conditions.append(f"[System.AssignedTo] = '{assigned_to}'")
                if state:
                    conditions.append(f"[System.State] = '{state}'")
                
                wiql_query = f"SELECT [System.Id] FROM WorkItems WHERE {' AND '.join(conditions)} ORDER BY [System.ChangedDate] DESC"
            
            # Execute query
            wiql = {"query": wiql_query}
            result = self.work_item_client.query_by_wiql(wiql, top=max_results)
            
            if not result.work_items:
                return []
            
            # Get work item IDs
            work_item_ids = [item.id for item in result.work_items]
            
            # Get detailed work item information
            work_items = self.work_item_client.get_work_items(work_item_ids)
            
            results = []
            for work_item in work_items:
                fields = work_item.fields
                results.append({
                    "id": work_item.id,
                    "title": fields.get("System.Title", "No title"),
                    "work_item_type": fields.get("System.WorkItemType", "Unknown"),
                    "state": fields.get("System.State", "Unknown"),
                    "assigned_to": fields.get("System.AssignedTo", {}).get("displayName", "Unassigned") if fields.get("System.AssignedTo") else "Unassigned",
                    "created_date": fields.get("System.CreatedDate", "Unknown"),
                    "priority": fields.get("Microsoft.VSTS.Common.Priority", "Not set"),
                    "web_url": f"{self.org_url}/{self.project_name}/_workitems/edit/{work_item.id}"
                })
            
            return results
            
        except Exception as e:
            return [{"error": f"Error searching work items: {e}"}]

    def update_work_item(self, work_item_id: int, **fields) -> Dict[str, Any]:
        """
        Update a work item.
        
        Args:
            work_item_id (int): Work item ID to update
            **fields: Fields to update
            
        Returns:
            Dict containing update result
        """
        try:
            patch_document = []
            
            # Map common field names to Azure DevOps field names
            field_mapping = {
                'title': '/fields/System.Title',
                'description': '/fields/System.Description',
                'assigned_to': '/fields/System.AssignedTo',
                'state': '/fields/System.State',
                'priority': '/fields/Microsoft.VSTS.Common.Priority',
                'tags': '/fields/System.Tags'
            }
            
            for field, value in fields.items():
                if field in field_mapping:
                    patch_document.append(JsonPatchOperation(
                        op="add",
                        path=field_mapping[field],
                        value=value
                    ))
            
            if not patch_document:
                return {
                    "success": False,
                    "error": "No valid fields provided for update"
                }
            
            # Update the work item
            updated_work_item = self.work_item_client.update_work_item(
                document=patch_document,
                id=work_item_id
            )
            
            return {
                "success": True,
                "work_item_id": work_item_id,
                "updated_fields": list(fields.keys()),
                "web_url": f"{self.org_url}/{self.project_name}/_workitems/edit/{work_item_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error updating work item {work_item_id}: {e}"
            }

    def get_project_info(self) -> Dict[str, Any]:
        """
        Get project information and available work item types.
        
        Returns:
            Dict containing project information
        """
        try:
            core_client = self.connection.clients.get_core_client()
            project = core_client.get_project(self.project_name)
            
            # Get work item types
            work_item_types = self.work_item_client.get_work_item_types(self.project_name)
            
            return {
                "success": True,
                "project_name": project.name,
                "project_id": project.id,
                "description": project.description,
                "url": project.url,
                "work_item_types": [wit.name for wit in work_item_types],
                "web_url": f"{self.org_url}/{self.project_name}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting project info: {e}"
            }

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with Azure DevOps tools."""
        
        def create_work_item_tool(work_item_type: str, title: str, description: str = "", 
                                assigned_to: str = "", priority: str = "", tags: str = "") -> str:
            """
            Create a new work item in Azure DevOps.
            
            Args:
                work_item_type (str): Type of work item (User Story, Task, Bug, Feature, Epic)
                title (str): Work item title
                description (str): Detailed description (optional)
                assigned_to (str): Assignee email/username (optional)
                priority (str): Priority level 1-4 (optional)
                tags (str): Comma-separated tags (optional)
                
            Returns:
                str: Creation result message
            """
            # Convert priority to int if provided
            priority_int = None
            if priority:
                try:
                    priority_int = int(priority)
                except ValueError:
                    return f"❌ Invalid priority '{priority}'. Please use 1-4."
            
            result = self.create_work_item(
                work_item_type=work_item_type,
                title=title,
                description=description if description else None,
                assigned_to=assigned_to if assigned_to else None,
                priority=priority_int,
                tags=tags if tags else None
            )
            
            if result["success"]:
                response = f"✅ Successfully created {result['work_item_type']}: #{result['work_item_id']}\n"
                response += f"📝 Title: {result['title']}\n"
                if result['description']:
                    response += f"📄 Description: {result['description']}\n"
                response += f"🔗 Web URL: {result['web_url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def get_work_item_tool(work_item_id: str) -> str:
            """
            Get detailed information about a work item.
            
            Args:
                work_item_id (str): The work item ID
                
            Returns:
                str: Work item details or error message
            """
            try:
                work_item_id_int = int(work_item_id)
            except ValueError:
                return f"❌ Invalid work item ID '{work_item_id}'. Please provide a valid number."
            
            result = self.get_work_item(work_item_id_int)
            
            if result["success"]:
                response = f"📋 Work Item Details: #{result['id']}\n\n"
                response += f"📝 Title: {result['title']}\n"
                response += f"🏷️  Type: {result['work_item_type']}\n"
                response += f"📊 State: {result['state']}\n"
                response += f"👤 Assigned To: {result['assigned_to']}\n"
                response += f"👤 Created By: {result['created_by']}\n"
                response += f"📅 Created: {result['created_date']}\n"
                response += f"📅 Last Changed: {result['changed_date']}\n"
                response += f"⚡ Priority: {result['priority']}\n"
                response += f"🏷️  Tags: {result['tags']}\n"
                response += f"📄 Description: {result['description']}\n"
                response += f"🔗 Web URL: {result['web_url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def search_work_items_tool(work_item_type: str = "", assigned_to: str = "", 
                                 state: str = "", max_results: str = "10") -> str:
            """
            Search for work items in the project.
            
            Args:
                work_item_type (str): Filter by work item type (optional)
                assigned_to (str): Filter by assignee (optional)
                state (str): Filter by state (optional)
                max_results (str): Maximum number of results (default: 10)
                
            Returns:
                str: Search results or error message
            """
            try:
                max_results_int = int(max_results)
            except ValueError:
                max_results_int = 10
            
            results = self.search_work_items(
                work_item_type=work_item_type if work_item_type else None,
                assigned_to=assigned_to if assigned_to else None,
                state=state if state else None,
                max_results=max_results_int
            )
            
            if results and not results[0].get("error"):
                response = f"🔍 Work Items Found ({len(results)} results):\n\n"
                for item in results:
                    response += f"📋 #{item['id']}: {item['title']}\n"
                    response += f"   🏷️ Type: {item['work_item_type']} | 📊 State: {item['state']}\n"
                    response += f"   👤 Assigned: {item['assigned_to']} | ⚡ Priority: {item['priority']}\n"
                    response += f"   📅 Created: {item['created_date']}\n"
                    response += f"   🔗 {item['web_url']}\n\n"
                return response.strip()
            elif results and results[0].get("error"):
                return f"❌ {results[0]['error']}"
            else:
                return f"🔍 No work items found matching the criteria"

        def update_work_item_tool(work_item_id: str, title: str = "", description: str = "", 
                                assigned_to: str = "", state: str = "", priority: str = "", tags: str = "") -> str:
            """
            Update an existing work item.
            
            Args:
                work_item_id (str): The work item ID to update
                title (str): New title (optional)
                description (str): New description (optional)
                assigned_to (str): New assignee (optional)
                state (str): New state (optional)
                priority (str): New priority 1-4 (optional)
                tags (str): New tags (optional)
                
            Returns:
                str: Update result message
            """
            try:
                work_item_id_int = int(work_item_id)
            except ValueError:
                return f"❌ Invalid work item ID '{work_item_id}'. Please provide a valid number."
            
            update_fields = {}
            if title:
                update_fields['title'] = title
            if description:
                update_fields['description'] = description
            if assigned_to:
                update_fields['assigned_to'] = assigned_to
            if state:
                update_fields['state'] = state
            if priority:
                try:
                    update_fields['priority'] = int(priority)
                except ValueError:
                    return f"❌ Invalid priority '{priority}'. Please use 1-4."
            if tags:
                update_fields['tags'] = tags
            
            if not update_fields:
                return "❌ No fields specified for update. Please provide at least one field to update."
            
            result = self.update_work_item(work_item_id_int, **update_fields)
            
            if result["success"]:
                response = f"✅ Successfully updated work item #{result['work_item_id']}\n"
                response += f"📝 Updated fields: {', '.join(result['updated_fields'])}\n"
                response += f"🔗 Web URL: {result['web_url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def get_project_info_tool() -> str:
            """
            Get information about the current Azure DevOps project.
            
            Returns:
                str: Project information
            """
            result = self.get_project_info()
            
            if result["success"]:
                response = f"📁 Project Information\n\n"
                response += f"📝 Name: {result['project_name']}\n"
                response += f"🔑 ID: {result['project_id']}\n"
                response += f"📄 Description: {result['description'] or 'No description'}\n"
                response += f"🔗 Web URL: {result['web_url']}\n\n"
                response += f"🏷️  Available Work Item Types:\n"
                for wit in result['work_item_types']:
                    response += f"   • {wit}\n"
                return response.strip()
            else:
                return f"❌ {result['error']}"

        # Create the agent with comprehensive Azure DevOps capabilities
        agent = Agent(
            name="azure_devops_agent",
            model="gemini-2.0-flash-exp",
            description="A comprehensive Azure DevOps work item management agent with full CRUD operations",
            instruction=f"""You are an Azure DevOps specialist agent connected to project '{self.project_name}' with comprehensive work item management capabilities.

🎯 **Your Core Capabilities:**

🆕 **Work Item Creation:**
- Create User Stories, Tasks, Bugs, Features, Epics, and other work item types
- Set priorities, descriptions, assignments, and tags
- Support for detailed work item specifications

🔍 **Work Item Management:**
- Search and retrieve work items with flexible filtering
- Get detailed work item information including all metadata
- List work items by type, assignee, state, or custom criteria

✏️ **Work Item Updates:**
- Update titles, descriptions, assignments, states, and priorities
- Change work item properties and track modifications
- Bulk updates and state transitions

📊 **Project Operations:**
- Get project information and available work item types
- Access work item URLs for direct navigation in Azure DevOps
- View project statistics and metadata

🚀 **Advanced Features:**
- WIQL (Work Item Query Language) support for complex searches
- Rich formatting with emojis and structured responses
- Error handling with helpful troubleshooting messages
- Integration with Azure DevOps web interface

💡 **Usage Examples:**
- "Create a new user story for 'User registration feature'"
- "Get details for work item 1234"
- "Search for bugs assigned to john@company.com"
- "Update work item 5678 state to 'In Progress'"
- "Show me all active tasks in the project"
- "Get project information and available work item types"

🔧 **Technical Notes:**
- Connected to: {self.org_url}
- Project: {self.project_name}
- Supports all standard Azure DevOps work item types
- WIQL queries supported for advanced searching
- Direct web links provided for easy navigation

How can I help you manage your Azure DevOps work items today?""",
            tools=[
                create_work_item_tool,
                get_work_item_tool,
                search_work_items_tool,
                update_work_item_tool,
                get_project_info_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "azure_devops_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the Azure DevOps agent using ADK's session and runner system.
        
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
            content = types.Content(
                role='user', 
                parts=[types.Part(text=message)]
            )
            
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
    """Example usage of the Azure DevOps Agent with Google ADK."""
    
    # Get configuration from environment variables
    org_url = os.environ.get("AZURE_DEVOPS_ORG_URL")
    project_name = os.environ.get("AZURE_DEVOPS_PROJECT_NAME")
    personal_access_token = os.environ.get("AZURE_DEVOPS_PAT")
    
    if not all([org_url, project_name, personal_access_token]):
        print("❌ Please set the required environment variables:")
        print("   - AZURE_DEVOPS_ORG_URL")
        print("   - AZURE_DEVOPS_PROJECT_NAME")
        print("   - AZURE_DEVOPS_PAT")
        return

    print("🚀 Initializing Azure DevOps Agent with Google ADK...")
    
    try:
        # Create the agent
        agent = AzureDevOpsAgent(org_url, project_name, personal_access_token)
        
        # Test conversations
        test_messages = [
            "Show me information about the current project",
            "Create a new user story for 'Implement user authentication system'",
            "Search for all active work items in the project",
            "Get details for work item 1 if it exists",
            "What are your capabilities?",
            "Create a bug report for 'Login page crashes on mobile' with priority 1"
        ]
        
        print("\n" + "="*80)
        print("🧪 TESTING AZURE DEVOPS AGENT WITH GOOGLE ADK")
        print("="*80)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n[Test {i}] 💬 User: {message}")
            print("-" * 60)
            
            response = agent.chat_sync(message)
            print(f"🤖 Agent: {response}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to initialize Azure DevOps agent: {e}")


if __name__ == "__main__":
    main()