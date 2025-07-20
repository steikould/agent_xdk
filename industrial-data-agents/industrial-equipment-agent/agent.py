"""
Industrial Equipment Agent using Google's Agent Development Kit (ADK)
Location: industrial-data-agents/industrial-equipment-agent/agent.py
"""
import os
from typing import Optional, Dict, List, Any
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

# Google ADK imports
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner


class IndustrialEquipmentAgent:
    """
    An agent for accessing industrial equipment documentation from SharePoint
    using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self, site_url: str, client_id: str, client_secret: str):
        """
        Initialize the Industrial Equipment Agent.
        
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

    def _connect_to_sharepoint(self) -> Optional[ClientContext]:
        """
        Establish connection to SharePoint.
        
        Returns:
            ClientContext: The SharePoint client context or None if failed
        """
        try:
            ctx_auth = AuthenticationContext(self.site_url)
            token_acquired = ctx_auth.acquire_token_for_app(
                client_id=self.client_id, 
                client_secret=self.client_secret
            )
            
            if not token_acquired:
                print("❌ Failed to acquire SharePoint authentication token")
                return None
            
            ctx = ClientContext(self.site_url, ctx_auth)
            
            # Test connection
            web = ctx.web
            ctx.load(web)
            ctx.execute_query()
            
            print(f"✅ Successfully connected to SharePoint site: {web.properties.get('Title', 'Unknown Site')}")
            return ctx
            
        except Exception as e:
            print(f"❌ Failed to connect to SharePoint: {e}")
            return None

    def get_document(self, library_name: str, file_name: str) -> str:
        """
        Retrieve a document from SharePoint.
        
        Args:
            library_name (str): Name of the SharePoint library
            file_name (str): Name of the file to retrieve
            
        Returns:
            str: File content or error message
        """
        if not self.ctx:
            return "SharePoint connection not available."
        
        try:
            library = self.ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.ctx.load(files)
            self.ctx.execute_query()

            for file in files:
                if file.properties["Name"].lower() == file_name.lower():
                    file_content = file.download(self.ctx).content
                    
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

    def list_files(self, library_name: str) -> List[Dict[str, Any]]:
        """
        List files in a SharePoint library.
        
        Args:
            library_name (str): Name of the SharePoint library
            
        Returns:
            List of file information dictionaries
        """
        if not self.ctx:
            return [{"error": "SharePoint connection not available"}]
        
        try:
            library = self.ctx.web.lists.get_by_title(library_name)
            files = library.root_folder.files
            self.ctx.load(files)
            self.ctx.execute_query()
            
            file_list = []
            for file in files:
                file_info = {
                    'name': file.properties.get("Name", "Unknown"),
                    'size': file.properties.get("Length", 0),
                    'modified': file.properties.get("TimeLastModified", "Unknown"),
                    'created': file.properties.get("TimeCreated", "Unknown")
                }
                file_list.append(file_info)
            
            return file_list
            
        except Exception as e:
            return [{"error": f"Error listing files: {e}"}]

    def search_equipment_docs(self, equipment_type: str, library_name: str = "Equipment Documentation") -> List[Dict[str, Any]]:
        """
        Search for equipment-related documents.
        
        Args:
            equipment_type (str): Type of equipment to search for
            library_name (str): Name of the SharePoint library
            
        Returns:
            List of matching documents
        """
        all_files = self.list_files(library_name)
        if all_files and "error" in all_files[0]:
            return all_files
        
        matching_files = []
        for file_info in all_files:
            if equipment_type.lower() in file_info['name'].lower():
                matching_files.append(file_info)
        
        return matching_files

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with industrial equipment tools."""
        
        def get_equipment_document_tool(library_name: str, file_name: str) -> str:
            """
            Retrieve equipment documentation from SharePoint.
            
            Args:
                library_name (str): Name of the SharePoint library
                file_name (str): Name of the equipment document
                
            Returns:
                str: Document content or error message
            """
            content = self.get_document(library_name, file_name)
            
            if content.startswith("Error") or content.startswith("File") or content.startswith("SharePoint"):
                return f"❌ {content}"
            
            if content.startswith("[Binary file"):
                return f"📁 {content}"
            
            # Provide content preview
            preview_length = min(1500, len(content))
            preview = content[:preview_length]
            if len(content) > preview_length:
                preview += "..."
            
            return f"📄 Equipment Document: '{file_name}'\n" \
                   f"Library: '{library_name}'\n" \
                   f"Content length: {len(content):,} characters\n\n" \
                   f"Content:\n{preview}"

        def list_equipment_files_tool(library_name: str = "Equipment Documentation") -> str:
            """
            List all equipment documentation files.
            
            Args:
                library_name (str): Name of the SharePoint library
                
            Returns:
                str: Formatted list of equipment files
            """
            files = self.list_files(library_name)
            
            if files and "error" in files[0]:
                return f"❌ {files[0]['error']}"
            
            if not files:
                return f"📁 No files found in library '{library_name}'"
            
            result = f"📁 Equipment Documentation in '{library_name}':\n\n"
            
            # Group files by equipment type (heuristic based on naming)
            equipment_groups = {}
            for file_info in files:
                name = file_info['name']
                
                # Try to identify equipment type from filename
                equipment_type = "Other"
                equipment_keywords = {
                    "pump": "Pumps",
                    "compressor": "Compressors", 
                    "valve": "Valves",
                    "motor": "Motors",
                    "turbine": "Turbines",
                    "heat exchanger": "Heat Exchangers",
                    "pipe": "Piping",
                    "tank": "Tanks",
                    "sensor": "Sensors",
                    "plc": "Control Systems"
                }
                
                for keyword, category in equipment_keywords.items():
                    if keyword in name.lower():
                        equipment_type = category
                        break
                
                if equipment_type not in equipment_groups:
                    equipment_groups[equipment_type] = []
                equipment_groups[equipment_type].append(file_info)
            
            # Format grouped output
            for equipment_type, files in sorted(equipment_groups.items()):
                result += f"\n🔧 {equipment_type}:\n"
                for file_info in files:
                    size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
                    result += f"  • {file_info['name']} ({size_mb:.2f} MB)\n"
                    result += f"    Modified: {file_info['modified']}\n"
            
            return result.strip()

        def search_equipment_tool(equipment_type: str, library_name: str = "Equipment Documentation") -> str:
            """
            Search for specific equipment documentation.
            
            Args:
                equipment_type (str): Type of equipment (pump, valve, compressor, etc.)
                library_name (str): Name of the SharePoint library
                
            Returns:
                str: Search results
            """
            matching_files = self.search_equipment_docs(equipment_type, library_name)
            
            if matching_files and "error" in matching_files[0]:
                return f"❌ {matching_files[0]['error']}"
            
            if not matching_files:
                return f"🔍 No documentation found for equipment type '{equipment_type}' in '{library_name}'"
            
            result = f"🔍 Equipment Documentation for '{equipment_type}':\n\n"
            for file_info in matching_files:
                size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] > 0 else 0
                result += f"📄 {file_info['name']}\n"
                result += f"   • Size: {size_mb:.2f} MB\n"
                result += f"   • Modified: {file_info['modified']}\n"
                result += f"   • Created: {file_info['created']}\n\n"
            
            return result.strip()

        def get_equipment_categories_tool() -> str:
            """
            Get information about available equipment categories and documentation types.
            
            Returns:
                str: Equipment categories and documentation types
            """
            return """🏭 Industrial Equipment Categories and Documentation:

**1. Rotating Equipment:**
• Pumps (centrifugal, positive displacement, submersible)
• Compressors (reciprocating, screw, centrifugal)
• Turbines (gas, steam, hydraulic)
• Motors (electric, hydraulic, pneumatic)
• Fans and blowers

**2. Static Equipment:**
• Pressure vessels and tanks
• Heat exchangers (shell & tube, plate, air-cooled)
• Columns and towers
• Reactors
• Piping and pipelines

**3. Control and Instrumentation:**
• Control valves
• Safety valves and relief devices
• Flow meters and sensors
• PLCs and control systems
• SCADA components

**4. Auxiliary Equipment:**
• Filters and strainers
• Separators
• Mixers and agitators
• Conveyors and material handling

**Documentation Types Available:**
📋 Technical Specifications
📖 Operation & Maintenance Manuals
🔧 Installation Guides
⚠️ Safety Data Sheets
🔄 Spare Parts Catalogs
📊 Performance Curves
🛠️ Troubleshooting Guides
📝 Inspection Checklists
🎓 Training Materials
📐 Engineering Drawings

Use the search function to find specific equipment documentation!"""

        def get_maintenance_info_tool(equipment_name: str) -> str:
            """
            Get general maintenance information for equipment types.
            
            Args:
                equipment_name (str): Name or type of equipment
                
            Returns:
                str: Maintenance best practices and recommendations
            """
            equipment_lower = equipment_name.lower()
            
            maintenance_info = {
                "pump": """🔧 Pump Maintenance Best Practices:

**Daily Checks:**
• Monitor vibration levels
• Check for unusual noises
• Verify seal integrity (no leaks)
• Monitor discharge pressure and flow

**Weekly Maintenance:**
• Check coupling alignment
• Inspect bearings for temperature
• Verify lubrication levels
• Clean strainers if applicable

**Monthly Tasks:**
• Perform vibration analysis
• Check impeller clearances
• Test backup systems
• Review performance trends

**Annual Maintenance:**
• Complete overhaul inspection
• Replace wear rings if needed
• Balance impeller
• Update maintenance records""",
                
                "compressor": """🔧 Compressor Maintenance Guidelines:

**Daily Monitoring:**
• Check oil levels and pressure
• Monitor discharge temperature
• Verify cooling water flow
• Listen for abnormal sounds

**Weekly Tasks:**
• Drain moisture from air receiver
• Check belt tension (if applicable)
• Clean intake filters
• Inspect safety valves

**Monthly Maintenance:**
• Change oil filters
• Test unloader operation
• Check control system calibration
• Analyze oil samples

**Quarterly/Annual:**
• Replace air filters
• Overhaul valves
• Inspect internal components
• Perform efficiency tests""",
                
                "valve": """🔧 Valve Maintenance Procedures:

**Regular Inspection:**
• Check for external leaks
• Verify smooth operation
• Monitor actuator performance
• Inspect packing condition

**Preventive Maintenance:**
• Lubricate stems and bearings
• Adjust packing gland
• Calibrate positioners
• Test fail-safe operation

**Periodic Overhaul:**
• Replace seat and seals
• Lap seating surfaces
• Check body for erosion
• Update maintenance tags"""
            }
            
            # Find matching maintenance info
            for equipment_type, info in maintenance_info.items():
                if equipment_type in equipment_lower:
                    return info
            
            # Default response for unknown equipment
            return f"""🔧 General Maintenance Principles for {equipment_name}:

**Preventive Maintenance:**
• Follow manufacturer's recommendations
• Establish regular inspection schedules
• Monitor key performance indicators
• Keep detailed maintenance records

**Predictive Maintenance:**
• Implement vibration monitoring
• Use thermal imaging for hot spots
• Analyze oil and wear particles
• Track performance degradation

**Best Practices:**
• Train maintenance personnel
• Maintain spare parts inventory
• Use proper tools and procedures
• Document all maintenance activities

Search for specific '{equipment_name}' documentation for detailed procedures."""

        # Create the agent
        agent = Agent(
            name="industrial_equipment_expert",
            model="gemini-2.0-flash-exp",
            description="An expert agent for industrial equipment documentation and specifications",
            instruction="""You are an industrial equipment expert with access to technical documentation.

🏭 **Your Capabilities:**

📚 **Documentation Access:**
- Retrieve equipment manuals and specifications
- Access maintenance procedures
- Find installation guides
- Locate troubleshooting documentation

🔍 **Search Functions:**
- Search by equipment type (pumps, valves, compressors, etc.)
- Find documentation by specific model numbers
- Locate safety and compliance documents
- Access training materials

🔧 **Equipment Expertise:**
- Rotating equipment (pumps, compressors, turbines)
- Static equipment (vessels, heat exchangers, tanks)
- Control systems and instrumentation
- Auxiliary and support equipment

📋 **Information Services:**
- Provide maintenance best practices
- Share equipment categories and types
- Offer troubleshooting guidance
- Explain technical specifications

💡 **How I Help:**
- Find specific equipment documentation
- Provide maintenance recommendations
- Explain equipment operation principles
- Guide troubleshooting efforts
- Share safety considerations

I have access to SharePoint libraries containing industrial equipment documentation. 
Ask me to find specific manuals, search for equipment types, or get maintenance information!""",
            tools=[
                get_equipment_document_tool,
                list_equipment_files_tool,
                search_equipment_tool,
                get_equipment_categories_tool,
                get_maintenance_info_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "equipment_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the equipment agent using ADK's session and runner system.
        
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
    """Example usage of the Industrial Equipment Agent."""
    
    # Get configuration from environment variables
    site_url = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_URL")
    client_id = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_ID")
    client_secret = os.environ.get("INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_SECRET")

    if not all([site_url, client_id, client_secret]):
        print("❌ Please set the required environment variables:")
        print("   - INDUSTRIAL_EQUIPMENT_SHAREPOINT_URL")
        print("   - INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_ID")
        print("   - INDUSTRIAL_EQUIPMENT_SHAREPOINT_CLIENT_SECRET")
        return

    print("🚀 Initializing Industrial Equipment Agent with Google ADK...")
    
    # Create the agent
    agent = IndustrialEquipmentAgent(site_url, client_id, client_secret)
    
    # Test conversations
    test_messages = [
        "List all equipment documentation",
        "Search for pump documentation",
        "Get the file 'Centrifugal-Pump-Manual.pdf' from 'Equipment Documentation'",
        "What equipment categories do you have?",
        "Give me maintenance information for compressors"
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING INDUSTRIAL EQUIPMENT AGENT")
    print("="*80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}] 💬 User: {message}")
        print("-" * 60)
        
        response = agent.chat_sync(message)
        print(f"🤖 Agent: {response}")
        print()


if __name__ == "__main__":
    main()