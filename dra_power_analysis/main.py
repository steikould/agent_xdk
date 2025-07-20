"""
Main entry point for DRA Power Analysis System
Location: dra_power_analysis/main.py
"""
import os
import asyncio
import logging
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from dra_power_analysis import root_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DRAPowerAnalysisSystem:
    """Main class for running the DRA Power Analysis System."""
    
    def __init__(self):
        """Initialize the DRA Power Analysis System."""
        self.session_service = InMemorySessionService()
        self.app_name = "dra_power_analysis"
        self.runner = None
        
    async def initialize(self):
        """Initialize the system components."""
        try:
            # Create runner with root agent
            self.runner = Runner(
                agent=root_agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            logger.info("DRA Power Analysis System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def analyze(
        self, 
        location_id: str,
        pipeline_id: str,
        start_date: str,
        end_date: str,
        user_id: str = "analyst",
        session_id: Optional[str] = None
    ) -> str:
        """
        Run power analysis for specified parameters.
        
        Args:
            location_id: Location identifier (e.g., STN_A001)
            pipeline_id: Pipeline identifier (e.g., PL123)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            user_id: User identifier
            session_id: Session identifier (auto-generated if not provided)
            
        Returns:
            Analysis results as string
        """
        if not self.runner:
            return "❌ System not initialized. Call initialize() first."
        
        if not session_id:
            session_id = f"analysis_{location_id}_{pipeline_id}_{start_date}"
        
        try:
            # Create message
            message = f"Analyze power consumption for location {location_id} pipeline {pipeline_id} from {start_date} to {end_date}"
            
            content = {
                'role': 'user',
                'parts': [{'text': message}]
            }
            
            # Run analysis
            events = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )
            
            # Collect results
            responses = []
            async for event in events:
                if event.is_final_response():
                    for part in event.content.parts:
                        if hasattr(part, 'text'):
                            responses.append(part.text)
            
            return '\n'.join(responses) if responses else "No response generated."
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return f"❌ Analysis failed: {str(e)}"
    
    async def chat(self, message: str, user_id: str = "analyst", session_id: str = "chat_session") -> str:
        """
        Interactive chat with the system.
        
        Args:
            message: User message
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            System response
        """
        if not self.runner:
            return "❌ System not initialized. Call initialize() first."
        
        try:
            content = {
                'role': 'user',
                'parts': [{'text': message}]
            }
            
            events = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )
            
            responses = []
            async for event in events:
                if event.is_final_response():
                    for part in event.content.parts:
                        if hasattr(part, 'text'):
                            responses.append(part.text)
            
            return '\n'.join(responses) if responses else "No response generated."
            
        except Exception as e:
            logger.error(f"Chat failed: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"


async def main():
    """Main function to run the DRA Power Analysis System."""
    print("🚀 Starting DRA Power Analysis System...")
    
    # Initialize system
    system = DRAPowerAnalysisSystem()
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully")
    print("\n" + "="*60)
    
    # Example: Run an analysis
    print("📊 Running example analysis...")
    result = await system.analyze(
        location_id="STN_A001",
        pipeline_id="PL123",
        start_date="2024-01-01",
        end_date="2024-01-07"
    )
    print(result)
    
    print("\n" + "="*60)
    
    # Interactive mode
    print("💬 Entering interactive mode (type 'exit' to quit)...")
    while True:
        try:
            user_input = input("\n🎯 You: ").strip()
            if user_input.lower() == 'exit':
                break
            
            response = await system.chat(user_input)
            print(f"\n🤖 System: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


def run_analysis_sync(location_id: str, pipeline_id: str, start_date: str, end_date: str) -> str:
    """
    Synchronous wrapper for running analysis.
    
    Useful for integration with other systems.
    """
    async def _run():
        system = DRAPowerAnalysisSystem()
        if await system.initialize():
            return await system.analyze(location_id, pipeline_id, start_date, end_date)
        return "❌ Failed to initialize system"
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_run())


if __name__ == "__main__":
    # Set environment variables if needed
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        print("   Set it to your service account key file path")
        print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
    
    # Run the main function
    asyncio.run(main())