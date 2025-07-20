import os
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Forbidden
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner


class BigQueryInventoryAgent:
    """
    A comprehensive agent for managing BigQuery inventories and SharePoint documents
    using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self, project_id: str, sharepoint_site_url: str = None, 
                 sharepoint_client_id: str = None, sharepoint_client_secret: str = None):
        """
        Initialize the BigQuery Inventory Agent.
        
        Args:
            project_id (str): Google Cloud Project ID
            sharepoint_site_url (str, optional): SharePoint site URL
            sharepoint_client_id (str, optional): SharePoint app client ID
            sharepoint_client_secret (str, optional): SharePoint app client secret
        """
        self.project_id = project_id
        
        # Initialize BigQuery client
        try:
            self.bigquery_client = bigquery.Client(project=project_id)
            # Test BigQuery connection
            list(self.bigquery_client.list_datasets(max_results=1))
            print(f"✅ Successfully connected to BigQuery project: {project_id}")
        except Exception as e:
            print(f"⚠️  BigQuery connection failed: {e}")
            self.bigquery_client = None

        # Initialize SharePoint if credentials provided
        self.sharepoint_ctx = None
        if all([sharepoint_site_url, sharepoint_client_id, sharepoint_client_secret]):
            self.sharepoint_ctx = self._connect_to_sharepoint(
                sharepoint_site_url, sharepoint_client_id, sharepoint_client_secret
            )

        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _connect_to_sharepoint(self, site_url: str, client_id: str, client_secret: str) -> Optional:
        """Connect to SharePoint."""
        try:
            ctx_auth = AuthenticationContext(site_url)
            token_acquired = ctx_auth.acquire_token_for_app(
                client_id=client_id, 
                client_secret=client_secret
            )
            
            if not token_acquired:
                raise ConnectionError("Failed to acquire SharePoint authentication token")
            
            ctx = ClientContext(site_url, ctx_auth)
            
            # Test connection
            web = ctx.web
            ctx.load(web)
            ctx.execute_query()
            
            print(f"✅ Successfully connected to SharePoint: {web.properties.get('Title', 'Unknown Site')}")
            return ctx
            
        except Exception as e:
            print(f"⚠️  SharePoint connection failed: {e}")
            return None

    def get_bigquery_inventory(self, include_schema: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive BigQuery inventory.
        
        Args:
            include_schema (bool): Whether to include table schema information
            
        Returns:
            Dict containing inventory data or error message
        """
        if not self.bigquery_client:
            return {"error": "BigQuery client not available"}
        
        try:
            inventory = {
                "project_id": self.project_id,
                "generated_at": datetime.now().isoformat(),
                "datasets": {},
                "summary": {
                    "total_datasets": 0,
                    "total_tables": 0,
                    "total_views": 0
                }
            }
            
            datasets = list(self.bigquery_client.list_datasets())
            inventory["summary"]["total_datasets"] = len(datasets)
            
            for dataset in datasets:
                dataset_id = dataset.dataset_id
                dataset_info = {
                    "tables": [],
                    "views": [],
                    "created": None,
                    "location": None,
                    "description": None
                }
                
                try:
                    # Get dataset details
                    dataset_ref = self.bigquery_client.get_dataset(dataset_id)
                    dataset_info["created"] = dataset_ref.created.isoformat() if dataset_ref.created else None
                    dataset_info["location"] = dataset_ref.location
                    dataset_info["description"] = dataset_ref.description
                    
                    # List tables and views
                    tables = list(self.bigquery_client.list_tables(dataset_id))
                    
                    for table in tables:
                        table_info = {
                            "table_id": table.table_id,
                            "table_type": table.table_type,
                            "created": None,
                            "num_rows": None,
                            "size_bytes": None
                        }
                        
                        try:
                            # Get table details
                            table_ref = self.bigquery_client.get_table(table.reference)
                            table_info["created"] = table_ref.created.isoformat() if table_ref.created else None
                            table_info["num_rows"] = table_ref.num_rows
                            table_info["size_bytes"] = table_ref.num_bytes
                            
                            if include_schema and table_ref.schema:
                                table_info["schema"] = [
                                    {
                                        "name": field.name,
                                        "field_type": field.field_type,
                                        "mode": field.mode
                                    }
                                    for field in table_ref.schema
                                ]
                        except Exception as e:
                            table_info["error"] = str(e)
                        
                        if table.table_type == "TABLE":
                            dataset_info["tables"].append(table_info)
                            inventory["summary"]["total_tables"] += 1
                        elif table.table_type == "VIEW":
                            dataset_info["views"].append(table_info)
                            inventory["summary"]["total_views"] += 1
                            
                except Exception as e:
                    dataset_info["error"] = str(e)
                
                inventory["datasets"][dataset_id] = dataset_info
            
            return inventory
            
        except Exception as e:
            return {"error": f"Failed to list BigQuery inventory: {e}"}

    def execute_bigquery_query(self, query: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute a BigQuery query.
        
        Args:
            query (str): SQL query to execute
            dry_run (bool): Whether to perform a dry run
            
        Returns:
            Dict containing query results or error
        """
        if not self.bigquery_client:
            return {"error": "BigQuery client not available"}
        
        try:
            job_config = bigquery.QueryJobConfig(dry_run=dry_run)
            query_job = self.bigquery_client.query(query, job_config=job_config)
            
            if dry_run:
                return {
                    "dry_run": True,
                    "bytes_processed": query_job.total_bytes_processed,
                    "valid": True,
                    "query": query
                }
            
            results = query_job.result()
            rows = [dict(row) for row in results]
            
            return {
                "rows": rows,
                "total_rows": len(rows),
                "bytes_processed": query_job.total_bytes_processed,
                "job_id": query_job.job_id,
                "query": query
            }
            
        except Exception as e:
            return {"error": f"Query execution failed: {e}"}

    def get_sharepoint_document(self, library_name: str, file_name: str) -> str:
        """
        Retrieve a document from SharePoint.
        
        Args:
            library_name (str): Name of the SharePoint library
            file_name (str): Name of the file to retrieve
            
        Returns:
            str: File content or error message
        """
        if not self.sharepoint_ctx:
            return "SharePoint connection not available."
        
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
                        return file_content.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            return file_content.decode('utf-8-sig')  # Handle BOM
                        except UnicodeDecodeError:
                            return f"[Binary file - {len(file_content)} bytes. Cannot display as text.]"
            
            return f"File '{file_name}' not found in library '{library_name}'."
            
        except Exception as e:
            return f"Error reading file from SharePoint: {e}"

    def list_sharepoint_files(self, library_name: str) -> str:
        """
        List files in a SharePoint library.
        
        Args:
            library_name (str): Name of the SharePoint library
            
        Returns:
            str: Formatted list of files or error message
        """
        if not self.sharepoint_ctx:
            return "SharePoint connection not available."
        
        try:
            library = self.sharepoint_ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.sharepoint_ctx.load(files)
            self.sharepoint_ctx.execute_query()
            
            if not files:
                return f"No files found in library '{library_name}'"
            
            result = f"Files in SharePoint library '{library_name}':\n\n"
            for file in files:
                name = file.properties.get("Name", "Unknown")
                size = file.properties.get("Length", 0)
                modified = file.properties.get("TimeLastModified", "Unknown")
                size_mb = size / (1024 * 1024) if size > 0 else 0
                result += f"• {name} ({size_mb:.2f} MB) - Modified: {modified}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"Error listing SharePoint files: {e}"

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with all tools."""
        
        # Define tool functions for the agent
        def get_inventory_tool(include_schema: str = "false") -> str:
            """
            Get BigQuery inventory for the project.
            
            Args:
                include_schema (str): "true" to include schema details, "false" otherwise
                
            Returns:
                str: Formatted inventory report
            """
            include_schema_bool = include_schema.lower() == "true"
            inventory = self.get_bigquery_inventory(include_schema=include_schema_bool)
            
            if 'error' in inventory:
                return f"❌ Error: {inventory['error']}"
            
            # Format inventory for display
            result = f"📊 BigQuery Inventory Report\n"
            result += f"Project: {inventory['project_id']}\n"
            result += f"Generated: {inventory['generated_at']}\n\n"
            result += f"📈 Summary:\n"
            result += f"• Datasets: {inventory['summary']['total_datasets']}\n"
            result += f"• Tables: {inventory['summary']['total_tables']}\n"
            result += f"• Views: {inventory['summary']['total_views']}\n\n"
            
            result += f"📁 Dataset Details:\n"
            for dataset_id, dataset_info in inventory['datasets'].items():
                result += f"\n🔹 {dataset_id}\n"
                if 'error' in dataset_info:
                    result += f"   ❌ Error: {dataset_info['error']}\n"
                else:
                    result += f"   • Tables: {len(dataset_info['tables'])}\n"
                    result += f"   • Views: {len(dataset_info['views'])}\n"
                    if dataset_info['location']:
                        result += f"   • Location: {dataset_info['location']}\n"
                    if dataset_info['description']:
                        result += f"   • Description: {dataset_info['description']}\n"
                    
                    # List table names if requested
                    if dataset_info['tables']:
                        result += f"   • Table names: {', '.join([t['table_id'] for t in dataset_info['tables']])}\n"
            
            return result

        def execute_query_tool(query: str, dry_run: str = "false") -> str:
            """
            Execute a BigQuery SQL query.
            
            Args:
                query (str): The SQL query to execute
                dry_run (str): "true" for dry run validation, "false" to execute
                
            Returns:
                str: Query results or error message
            """
            dry_run_bool = dry_run.lower() == "true"
            result = self.execute_bigquery_query(query, dry_run=dry_run_bool)
            
            if 'error' in result:
                return f"❌ Query Error: {result['error']}"
            
            if dry_run_bool:
                return f"✅ Query Validation Successful\n" \
                       f"Query: {result['query']}\n" \
                       f"Bytes to process: {result['bytes_processed']:,}\n" \
                       f"Status: Valid syntax"
            
            response = f"✅ Query Executed Successfully\n"
            response += f"Query: {result['query']}\n"
            response += f"Rows returned: {result['total_rows']:,}\n"
            response += f"Bytes processed: {result.get('bytes_processed', 'Unknown')}\n"
            response += f"Job ID: {result.get('job_id', 'Unknown')}\n\n"
            
            if result['rows']:
                response += f"📋 Results (showing first 10 rows):\n"
                for i, row in enumerate(result['rows'][:10]):
                    response += f"{i+1}. {row}\n"
                
                if len(result['rows']) > 10:
                    response += f"... and {len(result['rows']) - 10} more rows\n"
            
            return response

        def get_sharepoint_file_tool(library_name: str, file_name: str) -> str:
            """
            Retrieve a document from SharePoint.
            
            Args:
                library_name (str): Name of the SharePoint library
                file_name (str): Name of the file to retrieve
                
            Returns:
                str: File content preview or error message
            """
            content = self.get_sharepoint_document(library_name, file_name)
            
            if content.startswith("Error") or content.startswith("File") or content.startswith("SharePoint"):
                return f"❌ {content}"
            
            if content.startswith("[Binary file"):
                return f"📁 {content}"
            
            # Provide content preview
            preview_length = min(1000, len(content))
            preview = content[:preview_length]
            if len(content) > preview_length:
                preview += "..."
            
            return f"📄 SharePoint Document: '{file_name}'\n" \
                   f"Library: '{library_name}'\n" \
                   f"Content length: {len(content):,} characters\n\n" \
                   f"Content preview:\n{preview}"

        def list_sharepoint_files_tool(library_name: str) -> str:
            """
            List files in a SharePoint library.
            
            Args:
                library_name (str): Name of the SharePoint library
                
            Returns:
                str: List of files or error message
            """
            result = self.list_sharepoint_files(library_name)
            
            if result.startswith("Error") or result.startswith("SharePoint"):
                return f"❌ {result}"
            
            return f"📁 {result}"

        # Create the agent
        agent = Agent(
            name="bigquery_sharepoint_agent",
            model="gemini-2.0-flash-exp",  # Use the latest Gemini model
            description="A comprehensive agent for BigQuery inventory management and SharePoint document operations",
            instruction="""You are a BigQuery and SharePoint specialist agent. You can help users with:

🔍 BigQuery Operations:
- Get comprehensive inventory reports of datasets, tables, and views
- Execute SQL queries with validation and results
- Analyze data structure and metadata

📁 SharePoint Operations:
- Retrieve and preview documents from SharePoint libraries
- List files in SharePoint document libraries
- Access file metadata and content

💡 Usage Tips:
- For BigQuery inventory, I can include schema details if needed
- For queries, I can do dry runs for validation before execution
- For SharePoint, I can handle both text and binary files
- Always specify library names and file names accurately for SharePoint operations

How can I help you today?""",
            tools=[
                get_inventory_tool,
                execute_query_tool, 
                get_sharepoint_file_tool,
                list_sharepoint_files_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "bigquery_agent", 
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
            return f"Error during chat: {e}"

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
            return f"Error: {e}"


def main():
    """Example usage of the BigQuery Inventory Agent."""
    
    # Get configuration from environment variables
    project_id = os.environ.get("GCP_PROJECT_ID")
    sharepoint_site_url = os.environ.get("SHAREPOINT_SITE_URL")
    sharepoint_client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
    sharepoint_client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")

    if not project_id:
        print("❌ Please set the GCP_PROJECT_ID environment variable.")
        return

    print("🚀 Initializing BigQuery Inventory Agent with Google ADK...")
    
    # Create the agent
    agent = BigQueryInventoryAgent(
        project_id=project_id,
        sharepoint_site_url=sharepoint_site_url,
        sharepoint_client_id=sharepoint_client_id,
        sharepoint_client_secret=sharepoint_client_secret
    )
    
    # Test conversations
    test_messages = [
        "Show me the BigQuery inventory for this project",
        "Get the BigQuery inventory with schema details",
        "Execute this query: SELECT COUNT(*) as total_datasets FROM `INFORMATION_SCHEMA.SCHEMATA`",
        "List files in the 'Shared Documents' SharePoint library",
        "What can you help me with?"
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING BIGQUERY INVENTORY AGENT")
    print("="*80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}] 💬 User: {message}")
        print("-" * 60)
        
        response = agent.chat_sync(message)
        print(f"🤖 Agent: {response}")
        print()


if __name__ == "__main__":
    main()