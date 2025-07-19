import os
import re
from typing import Optional, Dict, Any, List
from jira import JIRA
from datetime import datetime

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk import types


class JiraAgent:
    """
    A comprehensive Jira issue management agent using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self, server_url: str, username: str, api_token: str, project_key: str):
        """
        Initialize the Jira Agent.
        
        Args:
            server_url (str): The Jira server URL
            username (str): Jira username
            api_token (str): Jira API token
            project_key (str): The project key to create issues in
        """
        self.server_url = server_url
        self.username = username
        self.api_token = api_token
        self.project_key = project_key
        
        # Initialize Jira connection
        self.jira = self._connect_to_jira()
        
        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _connect_to_jira(self) -> JIRA:
        """
        Establish connection to Jira.
        
        Returns:
            JIRA: The Jira client instance
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            jira_client = JIRA(server=self.server_url, basic_auth=(self.username, self.api_token))
            
            # Test connection
            current_user = jira_client.myself()
            print(f"✅ Successfully connected to Jira at {self.server_url}")
            print(f"   Logged in as: {current_user['displayName']} ({current_user['emailAddress']})")
            
            return jira_client
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Jira: {e}")

    def validate_issue_type(self, issue_type: str) -> bool:
        """
        Validate if the issue type exists in the project.
        
        Args:
            issue_type (str): The issue type to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            issue_types = self.jira.issue_types()
            available_types = [it.name for it in issue_types]
            return issue_type in available_types
        except Exception:
            # If we can't validate, assume it's valid
            return True

    def create_issue(self, issue_type: str, summary: str, description: str = None, 
                    priority: str = None, labels: List[str] = None) -> Dict[str, Any]:
        """
        Create a Jira issue.
        
        Args:
            issue_type (str): The type of issue (Story, Task, Bug, Epic)
            summary (str): Issue summary/title
            description (str, optional): Issue description
            priority (str, optional): Issue priority
            labels (List[str], optional): Issue labels
            
        Returns:
            Dict containing issue creation result
        """
        try:
            # Validate issue type
            if not self.validate_issue_type(issue_type):
                return {
                    "success": False,
                    "error": f"Issue type '{issue_type}' is not valid for project {self.project_key}"
                }
            
            # Build issue dictionary
            issue_dict = {
                'project': {'key': self.project_key},
                'summary': summary,
                'issuetype': {'name': issue_type},
            }
            
            # Add optional fields
            if description:
                issue_dict['description'] = description
            
            if priority:
                issue_dict['priority'] = {'name': priority}
            
            if labels:
                issue_dict['labels'] = labels
            
            # Create the issue
            new_issue = self.jira.create_issue(fields=issue_dict)
            
            return {
                "success": True,
                "issue_key": new_issue.key,
                "issue_type": issue_type,
                "summary": summary,
                "description": description,
                "url": f"{self.server_url}/browse/{new_issue.key}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating issue: {e}"
            }

    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        """
        Retrieve information about a specific issue.
        
        Args:
            issue_key (str): The Jira issue key
            
        Returns:
            Dict containing issue information
        """
        try:
            issue = self.jira.issue(issue_key)
            
            return {
                "success": True,
                "key": issue.key,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name,
                "issue_type": issue.fields.issuetype.name,
                "priority": issue.fields.priority.name if issue.fields.priority else "None",
                "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                "reporter": issue.fields.reporter.displayName if issue.fields.reporter else "Unknown",
                "created": issue.fields.created,
                "updated": issue.fields.updated,
                "description": issue.fields.description or "No description",
                "url": f"{self.server_url}/browse/{issue.key}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving issue {issue_key}: {e}"
            }

    def search_issues(self, jql_query: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for issues using JQL.
        
        Args:
            jql_query (str, optional): JQL query string
            max_results (int): Maximum number of issues to return
            
        Returns:
            List of issue dictionaries
        """
        try:
            if not jql_query:
                jql_query = f"project = {self.project_key} ORDER BY created DESC"
            
            issues = self.jira.search_issues(jql_query, maxResults=max_results)
            
            result = []
            for issue in issues:
                result.append({
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "issue_type": issue.fields.issuetype.name,
                    "priority": issue.fields.priority.name if issue.fields.priority else "None",
                    "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                    "created": issue.fields.created,
                    "url": f"{self.server_url}/browse/{issue.key}"
                })
            
            return result
            
        except Exception as e:
            return [{"error": f"Error searching issues: {e}"}]

    def update_issue(self, issue_key: str, **fields) -> Dict[str, Any]:
        """
        Update a Jira issue.
        
        Args:
            issue_key (str): The issue key to update
            **fields: Fields to update
            
        Returns:
            Dict containing update result
        """
        try:
            issue = self.jira.issue(issue_key)
            
            # Update fields
            update_dict = {}
            if 'summary' in fields:
                update_dict['summary'] = fields['summary']
            if 'description' in fields:
                update_dict['description'] = fields['description']
            if 'priority' in fields:
                update_dict['priority'] = {'name': fields['priority']}
            if 'assignee' in fields:
                update_dict['assignee'] = {'name': fields['assignee']}
            
            issue.update(fields=update_dict)
            
            return {
                "success": True,
                "issue_key": issue_key,
                "updated_fields": list(fields.keys()),
                "url": f"{self.server_url}/browse/{issue_key}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error updating issue {issue_key}: {e}"
            }

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """
        Add a comment to a Jira issue.
        
        Args:
            issue_key (str): The issue key
            comment (str): Comment text
            
        Returns:
            Dict containing comment result
        """
        try:
            self.jira.add_comment(issue_key, comment)
            
            return {
                "success": True,
                "issue_key": issue_key,
                "comment": comment,
                "url": f"{self.server_url}/browse/{issue_key}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error adding comment to {issue_key}: {e}"
            }

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with Jira tools."""
        
        def create_issue_tool(issue_type: str, summary: str, description: str = "", 
                             priority: str = "", labels: str = "") -> str:
            """
            Create a new Jira issue.
            
            Args:
                issue_type (str): Type of issue (Story, Task, Bug, Epic, etc.)
                summary (str): Issue title/summary
                description (str): Detailed description (optional)
                priority (str): Issue priority (optional)
                labels (str): Comma-separated labels (optional)
                
            Returns:
                str: Creation result message
            """
            # Parse labels if provided
            label_list = [label.strip() for label in labels.split(",")] if labels else None
            
            result = self.create_issue(
                issue_type=issue_type,
                summary=summary,
                description=description if description else None,
                priority=priority if priority else None,
                labels=label_list
            )
            
            if result["success"]:
                response = f"✅ Successfully created {result['issue_type']}: {result['issue_key']}\n"
                response += f"📝 Summary: {result['summary']}\n"
                if result['description']:
                    response += f"📄 Description: {result['description']}\n"
                response += f"🔗 URL: {result['url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def get_issue_tool(issue_key: str) -> str:
            """
            Get detailed information about a Jira issue.
            
            Args:
                issue_key (str): The Jira issue key (e.g., PROJ-123)
                
            Returns:
                str: Issue details or error message
            """
            result = self.get_issue(issue_key)
            
            if result["success"]:
                response = f"🎫 Issue Details: {result['key']}\n\n"
                response += f"📝 Summary: {result['summary']}\n"
                response += f"📊 Status: {result['status']}\n"
                response += f"🏷️  Type: {result['issue_type']}\n"
                response += f"⚡ Priority: {result['priority']}\n"
                response += f"👤 Assignee: {result['assignee']}\n"
                response += f"👤 Reporter: {result['reporter']}\n"
                response += f"📅 Created: {result['created']}\n"
                response += f"📅 Updated: {result['updated']}\n"
                response += f"📄 Description: {result['description']}\n"
                response += f"🔗 URL: {result['url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def search_issues_tool(jql_query: str = "", max_results: str = "10") -> str:
            """
            Search for Jira issues using JQL or get recent project issues.
            
            Args:
                jql_query (str): JQL query string (optional, defaults to recent project issues)
                max_results (str): Maximum number of results to return (default: 10)
                
            Returns:
                str: Search results or error message
            """
            try:
                max_results_int = int(max_results)
            except ValueError:
                max_results_int = 10
            
            query = jql_query if jql_query else f"project = {self.project_key} ORDER BY created DESC"
            results = self.search_issues(query, max_results_int)
            
            if results and not results[0].get("error"):
                response = f"🔍 Search Results ({len(results)} issues found):\n\n"
                for issue in results:
                    response += f"🎫 {issue['key']}: {issue['summary']}\n"
                    response += f"   📊 Status: {issue['status']} | 🏷️ Type: {issue['issue_type']}\n"
                    response += f"   ⚡ Priority: {issue['priority']} | 👤 Assignee: {issue['assignee']}\n"
                    response += f"   📅 Created: {issue['created']}\n"
                    response += f"   🔗 {issue['url']}\n\n"
                return response.strip()
            elif results and results[0].get("error"):
                return f"❌ {results[0]['error']}"
            else:
                return f"🔍 No issues found for query: {query}"

        def update_issue_tool(issue_key: str, summary: str = "", description: str = "", 
                             priority: str = "", assignee: str = "") -> str:
            """
            Update an existing Jira issue.
            
            Args:
                issue_key (str): The Jira issue key to update
                summary (str): New summary/title (optional)
                description (str): New description (optional)
                priority (str): New priority (optional)
                assignee (str): New assignee username (optional)
                
            Returns:
                str: Update result message
            """
            update_fields = {}
            if summary:
                update_fields['summary'] = summary
            if description:
                update_fields['description'] = description
            if priority:
                update_fields['priority'] = priority
            if assignee:
                update_fields['assignee'] = assignee
            
            if not update_fields:
                return "❌ No fields specified for update. Please provide at least one field to update."
            
            result = self.update_issue(issue_key, **update_fields)
            
            if result["success"]:
                response = f"✅ Successfully updated issue {result['issue_key']}\n"
                response += f"📝 Updated fields: {', '.join(result['updated_fields'])}\n"
                response += f"🔗 URL: {result['url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def add_comment_tool(issue_key: str, comment: str) -> str:
            """
            Add a comment to a Jira issue.
            
            Args:
                issue_key (str): The Jira issue key
                comment (str): Comment text to add
                
            Returns:
                str: Comment result message
            """
            result = self.add_comment(issue_key, comment)
            
            if result["success"]:
                response = f"✅ Successfully added comment to {result['issue_key']}\n"
                response += f"💬 Comment: {result['comment']}\n"
                response += f"🔗 URL: {result['url']}"
                return response
            else:
                return f"❌ {result['error']}"

        def get_project_info_tool() -> str:
            """
            Get information about the current Jira project.
            
            Returns:
                str: Project information
            """
            try:
                project = self.jira.project(self.project_key)
                
                response = f"📁 Project Information: {self.project_key}\n\n"
                response += f"📝 Name: {project.name}\n"
                response += f"🔑 Key: {project.key}\n"
                response += f"📄 Description: {project.description or 'No description'}\n"
                response += f"👤 Lead: {project.lead.displayName if project.lead else 'Unknown'}\n"
                response += f"🔗 URL: {project.self}\n\n"
                
                # Get issue types
                issue_types = self.jira.issue_types()
                response += f"🏷️  Available Issue Types: {', '.join([it.name for it in issue_types])}"
                
                return response
                
            except Exception as e:
                return f"❌ Error getting project info: {e}"

        # Create the agent with comprehensive Jira capabilities
        agent = Agent(
            name="jira_issue_agent",
            model="gemini-2.0-flash-exp",
            description="A comprehensive Jira issue management agent with full CRUD operations",
            instruction=f"""You are a Jira specialist agent connected to project '{self.project_key}' with comprehensive issue management capabilities.

🎯 **Your Core Capabilities:**

🆕 **Issue Creation:**
- Create Stories, Tasks, Bugs, Epics, and other issue types
- Set priorities, descriptions, and labels
- Support for detailed issue specifications

🔍 **Issue Management:**
- Search and retrieve issues using JQL queries
- Get detailed issue information including status, assignee, dates
- List recent project issues with full metadata

✏️ **Issue Updates:**
- Update issue summaries, descriptions, priorities
- Change assignees and other field values
- Add comments and track changes

📊 **Project Operations:**
- Get project information and available issue types
- Search across the entire project or use custom JQL
- Access issue URLs for direct navigation

🚀 **Advanced Features:**
- JQL query support for complex searches
- Rich formatting with emojis and structured responses
- Error handling with helpful troubleshooting messages
- Batch operations and bulk updates

💡 **Usage Examples:**
- "Create a new story for 'User registration feature'"
- "Get details for issue PROJ-123"
- "Search for bugs with high priority"
- "Update PROJ-456 summary to 'Updated login flow'"
- "Add comment to PROJ-789: 'Testing completed successfully'"
- "Show me recent issues in the project"

🔧 **Technical Notes:**
- Connected to: {self.server_url}
- Default project: {self.project_key}
- Supports all standard Jira issue types
- JQL queries are supported for advanced searching

How can I help you manage your Jira issues today?""",
            tools=[
                create_issue_tool,
                get_issue_tool,
                search_issues_tool,
                update_issue_tool,
                add_comment_tool,
                get_project_info_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "jira_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the Jira agent using ADK's session and runner system.
        
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
    """Example usage of the Jira Agent with Google ADK."""
    
    # Get configuration from environment variables
    server_url = os.environ.get("JIRA_SERVER_URL")
    username = os.environ.get("JIRA_USERNAME")
    api_token = os.environ.get("JIRA_API_TOKEN")
    project_key = os.environ.get("JIRA_PROJECT_KEY")
    
    if not all([server_url, username, api_token, project_key]):
        print("❌ Please set the required environment variables:")
        print("   - JIRA_SERVER_URL")
        print("   - JIRA_USERNAME")
        print("   - JIRA_API_TOKEN")
        print("   - JIRA_PROJECT_KEY")
        return

    print("🚀 Initializing Jira Agent with Google ADK...")
    
    try:
        # Create the agent
        agent = JiraAgent(server_url, username, api_token, project_key)
        
        # Test conversations
        test_messages = [
            "Show me information about the current project",
            "Create a new story for 'Implement user authentication system'",
            "Search for recent issues in the project",
            "Get details for the most recent issue",
            "What are your capabilities?",
            "Create a bug report for 'Login page not loading' with description 'Users cannot access the login page on mobile devices'"
        ]
        
        print("\n" + "="*80)
        print("🧪 TESTING JIRA AGENT WITH GOOGLE ADK")
        print("="*80)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n[Test {i}] 💬 User: {message}")
            print("-" * 60)
            
            response = agent.chat_sync(message)
            print(f"🤖 Agent: {response}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to initialize Jira agent: {e}")


if __name__ == "__main__":
    main()