"""
Digital Twinning Agent using Google's Agent Development Kit (ADK)
Location: industrial-data-agents/digital-twinning-agent/agent.py
"""
import os
from typing import Optional
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner


class DigitalTwinningAgent:
    """
    A digital twinning knowledge agent using Google's Agent Development Kit (ADK).
    """
    
    def __init__(self):
        """Initialize the Digital Twinning Agent."""
        # Create the ADK agent with tools
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the main ADK agent with digital twinning knowledge tools."""
        
        def get_data_requirements_tool() -> str:
            """
            Get information about data requirements for digital twins.
            
            Returns:
                str: Detailed information about data requirements
            """
            return """📊 Digital Twin Data Requirements:

1. **Real-time Sensor Data:**
   • Temperature, pressure, vibration, flow rates
   • Equipment performance metrics
   • Environmental conditions
   • Update frequency: milliseconds to minutes

2. **Historical Operational Data:**
   • Past performance records
   • Maintenance history
   • Failure events and root causes
   • Operational parameters over time

3. **3D Models and Geometries:**
   • CAD models of physical assets
   • Material properties
   • Assembly information
   • Spatial relationships

4. **Maintenance Records:**
   • Scheduled maintenance activities
   • Repair history
   • Component replacements
   • Service bulletins

5. **Process Data:**
   • Production schedules
   • Input/output parameters
   • Quality metrics
   • Energy consumption

6. **External Data:**
   • Weather conditions
   • Market demands
   • Supply chain information
   • Regulatory requirements"""

        def get_simulation_info_tool() -> str:
            """
            Get information about simulation aspects of digital twins.
            
            Returns:
                str: Information about digital twin simulations
            """
            return """🔬 Digital Twin Simulation Components:

1. **Physics-Based Models:**
   • Finite Element Analysis (FEA) for structural behavior
   • Computational Fluid Dynamics (CFD) for flow analysis
   • Thermodynamic models for heat transfer
   • Multi-physics coupling for complex interactions

2. **Machine Learning Models:**
   • Predictive maintenance algorithms
   • Anomaly detection systems
   • Performance optimization models
   • Pattern recognition for failure modes

3. **Real-time Simulation Requirements:**
   • High-performance computing infrastructure
   • Low-latency data streaming
   • Model reduction techniques
   • Parallel processing capabilities

4. **Validation and Calibration:**
   • Model verification against physical data
   • Parameter tuning and optimization
   • Uncertainty quantification
   • Continuous model improvement

5. **Scenario Testing:**
   • What-if analysis capabilities
   • Stress testing under extreme conditions
   • Optimization scenarios
   • Risk assessment simulations"""

        def get_benefits_tool() -> str:
            """
            Get information about benefits of digital twinning.
            
            Returns:
                str: Benefits of implementing digital twins
            """
            return """💡 Digital Twin Benefits:

1. **Predictive Maintenance:**
   • Forecast equipment failures before they occur
   • Optimize maintenance schedules
   • Reduce unplanned downtime by 30-50%
   • Extend asset lifespan

2. **Operational Efficiency:**
   • Real-time performance optimization
   • Energy consumption reduction
   • Process parameter optimization
   • Increased throughput

3. **Risk Mitigation:**
   • Test scenarios virtually without physical risk
   • Identify potential failure modes
   • Validate changes before implementation
   • Improve safety protocols

4. **Cost Savings:**
   • Reduced maintenance costs
   • Lower operational expenses
   • Minimized production losses
   • Optimized resource allocation

5. **Decision Support:**
   • Data-driven insights
   • Real-time visibility
   • Performance benchmarking
   • Strategic planning support

6. **Innovation Enablement:**
   • Rapid prototyping
   • Design optimization
   • New service development
   • Continuous improvement"""

        def get_implementation_guide_tool() -> str:
            """
            Get a guide for implementing digital twins.
            
            Returns:
                str: Step-by-step implementation guide
            """
            return """🚀 Digital Twin Implementation Guide:

**Phase 1: Assessment and Planning (Months 1-2)**
• Identify critical assets for digital twinning
• Define business objectives and KPIs
• Assess data availability and quality
• Develop implementation roadmap

**Phase 2: Data Infrastructure (Months 2-4)**
• Install necessary sensors and IoT devices
• Establish data collection systems
• Implement data storage and management
• Ensure cybersecurity measures

**Phase 3: Model Development (Months 4-8)**
• Create 3D CAD models or import existing ones
• Develop physics-based simulation models
• Build machine learning algorithms
• Integrate real-time data feeds

**Phase 4: Platform Integration (Months 8-10)**
• Select and deploy digital twin platform
• Connect data sources and models
• Develop user interfaces and dashboards
• Implement analytics and visualization

**Phase 5: Testing and Validation (Months 10-11)**
• Validate model accuracy against real data
• Conduct user acceptance testing
• Refine models based on feedback
• Document processes and procedures

**Phase 6: Deployment and Scaling (Month 12+)**
• Deploy to production environment
• Train operators and maintenance staff
• Monitor performance and iterate
• Scale to additional assets

**Key Success Factors:**
✓ Strong executive sponsorship
✓ Cross-functional collaboration
✓ Data quality management
✓ Change management strategy
✓ Continuous improvement mindset"""

        def get_technology_stack_tool() -> str:
            """
            Get information about technology stack for digital twins.
            
            Returns:
                str: Technology components and platforms
            """
            return """🛠️ Digital Twin Technology Stack:

**1. IoT and Data Collection:**
• Industrial IoT sensors (temperature, pressure, vibration)
• Edge computing devices
• SCADA systems integration
• OPC UA protocols
• MQTT brokers

**2. Data Management:**
• Time-series databases (InfluxDB, TimescaleDB)
• Data lakes (Azure Data Lake, AWS S3)
• Stream processing (Apache Kafka, Azure Stream Analytics)
• Data quality tools
• ETL/ELT pipelines

**3. Modeling and Simulation:**
• CAD software (SolidWorks, AutoCAD, CATIA)
• Simulation tools (ANSYS, COMSOL, Simulink)
• Game engines for visualization (Unity, Unreal)
• Custom physics engines
• ML frameworks (TensorFlow, PyTorch)

**4. Platform Solutions:**
• Azure Digital Twins
• AWS IoT TwinMaker
• GE Predix
• PTC ThingWorx
• Siemens MindSphere

**5. Analytics and AI:**
• Predictive analytics platforms
• Machine learning services
• Computer vision systems
• Natural language processing
• Optimization algorithms

**6. Visualization and UI:**
• 3D visualization frameworks
• AR/VR capabilities
• Web-based dashboards
• Mobile applications
• Real-time monitoring interfaces"""

        # Create the agent
        agent = Agent(
            name="digital_twinning_expert",
            model="gemini-2.0-flash-exp",
            description="An expert agent on digital twinning for industrial assets",
            instruction="""You are a digital twinning expert with deep knowledge of industrial digital twins.

🎯 **Your Expertise Includes:**

📊 **Data Requirements:**
- Sensor data needs and specifications
- Historical data requirements
- 3D modeling and CAD integration
- Real-time data streaming architectures

🔬 **Simulation and Modeling:**
- Physics-based simulation techniques
- Machine learning integration
- Model validation and calibration
- Real-time performance optimization

💡 **Business Value:**
- ROI calculation and benefits realization
- Use case identification
- Success metrics and KPIs
- Change management strategies

🛠️ **Technical Implementation:**
- Technology stack selection
- Platform evaluation
- Integration strategies
- Best practices and pitfalls

🚀 **Project Management:**
- Implementation roadmaps
- Resource planning
- Risk mitigation
- Scaling strategies

💬 **How I Can Help:**
- Explain digital twin concepts and benefits
- Guide technology selection
- Provide implementation roadmaps
- Share best practices and case studies
- Answer specific technical questions

Ask me anything about digital twinning for industrial assets!""",
            tools=[
                get_data_requirements_tool,
                get_simulation_info_tool,
                get_benefits_tool,
                get_implementation_guide_tool,
                get_technology_stack_tool
            ]
        )
        
        return agent

    async def chat(self, message: str, app_name: str = "digital_twin_agent", 
                   user_id: str = "default_user", session_id: str = "default_session") -> str:
        """
        Chat with the digital twinning agent using ADK's session and runner system.
        
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
    """Example usage of the Digital Twinning Agent."""
    print("🚀 Initializing Digital Twinning Agent with Google ADK...")
    
    # Create the agent
    agent = DigitalTwinningAgent()
    
    # Test conversations
    test_messages = [
        "What are the data requirements for a digital twin?",
        "Tell me about simulation aspects",
        "What are the benefits of digital twinning?",
        "How do I implement a digital twin project?",
        "What technology stack do I need?"
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING DIGITAL TWINNING AGENT")
    print("="*80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}] 💬 User: {message}")
        print("-" * 60)
        
        response = agent.chat_sync(message)
        print(f"🤖 Agent: {response}")
        print()


if __name__ == "__main__":
    main()