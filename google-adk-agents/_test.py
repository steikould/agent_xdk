"""
Test template for Google ADK agents
Use this template for creating consistent test files
"""
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from your_agent import YourSystemAgent  # Update import


class TestYourSystemAgent(unittest.TestCase):
    """Test cases for YourSystemAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_param1 = "test_value_1"
        self.test_param2 = "test_value_2"
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    @patch('your_agent.YourSystemClient')  # Update with actual client
    def test_agent_initialization_success(self, mock_client):
        """Test successful agent initialization."""
        # Mock the client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Create agent
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Assertions
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.agent)
        self.assertEqual(agent.connection_param1, self.test_param1)
        mock_client.assert_called_once()

    @patch('your_agent.YourSystemClient')
    def test_agent_initialization_failure(self, mock_client):
        """Test agent initialization with connection failure."""
        # Mock connection failure
        mock_client.side_effect = Exception("Connection failed")
        
        # Test that ConnectionError is raised
        with self.assertRaises(ConnectionError):
            agent = YourSystemAgent(self.test_param1, self.test_param2)

    @patch('your_agent.YourSystemClient')
    def test_system_operation_1_success(self, mock_client):
        """Test successful system operation."""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Execute
        result = agent.system_operation_1("test_param", "optional_param")
        
        # Assert
        self.assertTrue(result["success"])
        self.assertIn("data", result)

    @patch('your_agent.YourSystemClient')
    def test_system_operation_2_search(self, mock_client):
        """Test search operation."""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Execute
        results = agent.system_operation_2("search query")
        
        # Assert
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    @patch('your_agent.YourSystemClient')
    def test_chat_sync_method(self, mock_client):
        """Test synchronous chat method."""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Mock the async chat method
        with patch.object(agent, 'chat', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Test response"
            
            # Execute
            response = agent.chat_sync("Test message")
            
            # Assert
            self.assertEqual(response, "Test response")

    @patch('your_agent.YourSystemClient')
    async def test_chat_async_method(self, mock_client):
        """Test asynchronous chat method."""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Mock ADK components
        with patch('your_agent.InMemorySessionService') as mock_session_service:
            with patch('your_agent.Runner') as mock_runner:
                # Setup mocks
                mock_session = MagicMock()
                mock_session_service.return_value.create_session = AsyncMock(return_value=mock_session)
                
                # Mock event with response
                mock_event = MagicMock()
                mock_event.is_final_response.return_value = True
                mock_event.content.parts = [MagicMock(text="Test response")]
                
                # Make run_async return an async generator
                async def mock_events():
                    yield mock_event
                
                mock_runner.return_value.run_async = MagicMock(return_value=mock_events())
                
                # Execute
                response = await agent.chat("Test message")
                
                # Assert
                self.assertEqual(response, "Test response")

    @patch('your_agent.YourSystemClient')
    def test_tool_execution(self, mock_client):
        """Test tool execution through agent."""
        # Setup
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        agent = YourSystemAgent(self.test_param1, self.test_param2)
        
        # Get the tool functions
        tools = agent.agent.tools
        self.assertGreater(len(tools), 0)
        
        # Test first tool
        tool_1 = tools[0]
        result = tool_1("test_param")
        self.assertIsInstance(result, str)
        self.assertIn("✅", result)  # Check for success emoji

    def test_error_handling(self):
        """Test error handling in agent operations."""
        with patch('your_agent.YourSystemClient') as mock_client:
            # Setup agent
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            agent = YourSystemAgent(self.test_param1, self.test_param2)
            
            # Mock operation to raise exception
            with patch.object(agent, 'system_operation_1', side_effect=Exception("Test error")):
                result = agent.system_operation_1("param")
                
                # Assert error handling
                self.assertFalse(result["success"])
                self.assertIn("error", result)
                self.assertIn("Test error", result["error"])


class TestIntegration(unittest.TestCase):
    """Integration tests for the agent."""
    
    @unittest.skipIf(not os.environ.get("RUN_INTEGRATION_TESTS"), 
                     "Skipping integration tests (set RUN_INTEGRATION_TESTS=1)")
    def test_real_system_connection(self):
        """Test actual connection to the system."""
        # This test requires real credentials
        param1 = os.environ.get("YOUR_SYSTEM_PARAM1")
        param2 = os.environ.get("YOUR_SYSTEM_PARAM2")
        
        if not all([param1, param2]):
            self.skipTest("Missing required environment variables")
        
        # Test real connection
        agent = YourSystemAgent(param1, param2)
        self.assertIsNotNone(agent.client)
        
        # Test a real operation
        result = agent.system_operation_1("test")
        self.assertTrue(result["success"])


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()