import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# Configure basic logging for all agents
# In a larger application, this might be configured externally
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Agent(ABC):
    """
    Abstract Base Class for all agents in the system.
    It defines a common interface and provides basic logging capabilities.
    """

    def __init__(self, agent_name: str):
        """
        Initializes the agent with a specific name for logging purposes.

        Args:
            agent_name (str): The name of the agent, used in logging.
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(self.agent_name)
        self.logger.info(f"Agent '{self.agent_name}' initialized.")

    @abstractmethod
    def execute(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The main method for an agent to perform its primary task.
        It takes input data, processes it, and returns output data.

        Args:
            data (Dict[str, Any], optional): Input data for the agent.
                                            Defaults to None if the agent is a starting point.

        Returns:
            Dict[str, Any]: Output data from the agent's processing.
                            Can be None if the agent doesn't produce direct output
                            or is an endpoint (e.g., UI display).
        """
        pass

    def _validate_input(
        self, data: Dict[str, Any], required_keys: list[str] = None
    ) -> bool:
        """
        Protected method for basic input validation.
        Checks if the provided data is a dictionary and if required keys are present.
        Specific agents should override or extend this for more detailed validation.

        Args:
            data (Dict[str, Any]): The input data to validate.
            required_keys (list[str], optional): A list of keys that must be present in the data.
                                                 Defaults to None.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        if not isinstance(data, dict):
            self.logger.error("Input validation failed: Data is not a dictionary.")
            return False

        if required_keys:
            for key in required_keys:
                if key not in data:
                    self.logger.error(
                        f"Input validation failed: Missing required key '{key}'."
                    )
                    return False

        self.logger.debug("Input validation successful.")
        return True

    def _handle_error(
        self, error_message: str, exception: Exception = None
    ) -> Dict[str, Any]:
        """
        Protected method for consistent error handling and logging.
        Logs the error and returns a structured error dictionary.

        Args:
            error_message (str): A descriptive message for the error.
            exception (Exception, optional): The exception object, if an exception occurred.
                                           Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing error details.
        """
        if exception:
            self.logger.error(f"{error_message}: {exception}", exc_info=True)
        else:
            self.logger.error(error_message)

        return {
            "status": "error",
            "agent_name": self.agent_name,
            "error_message": error_message,
            "exception_type": str(type(exception).__name__) if exception else None,
        }

    def _log_message(self, message: str, level: int = logging.INFO):
        """
        Protected method for logging messages at different levels.

        Args:
            message (str): The message to log.
            level (int, optional): The logging level (e.g., logging.INFO, logging.WARNING).
                                   Defaults to logging.INFO.
        """
        self.logger.log(level, message)

    def get_agent_name(self) -> str:
        """
        Returns the name of the agent.

        Returns:
            str: The agent's name.
        """
        return self.agent_name


if __name__ == "__main__":
    # Example of how a concrete agent might inherit from Agent
    class MyImprovedTestAgent(Agent):
        def __init__(self):
            super().__init__("MyImprovedTestAgent")

        def execute(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
            self.logger.info(f"{self.agent_name} executing.")

            # Ensure data is a dictionary before detailed validation
            if (
                data is None
            ):  # Allow None if the agent is a starting point or handles it
                # In this example, we require data for this test agent
                return self._handle_error(
                    "Input data is None, but was expected for MyImprovedTestAgent."
                )

            if not isinstance(data, dict):
                return self._handle_error("Input data must be a dictionary.")

            if not self._validate_input(data, required_keys=["test_key"]):
                # _validate_input already logged the specific key error
                return self._handle_error(
                    f"Input validation failed for {self.agent_name}."
                )

            self._log_message(f"Received valid data: {data}")

            # Simulate processing
            processed_value = data.get("test_key", 0) * 2  # Ensure default for safety
            output_data = {
                "status": "success",
                "result": f"processed value is {processed_value}",
            }
            self.logger.info(f"{self.agent_name} execution completed.")
            return output_data

    print("\nTesting Improved Agent:")
    improved_agent = MyImprovedTestAgent()

    result_valid_imp = improved_agent.execute({"test_key": 25})
    print(f"Improved - Valid data: {result_valid_imp}")

    result_invalid_key_imp = improved_agent.execute({"wrong_key": 50})
    print(f"Improved - Invalid key: {result_invalid_key_imp}")

    result_non_dict_imp = improved_agent.execute("this is not a dict")
    print(f"Improved - Non-dict data: {result_non_dict_imp}")

    result_empty_dict_imp = improved_agent.execute({})  # Missing 'test_key'
    print(f"Improved - Empty dict (missing key): {result_empty_dict_imp}")

    result_none_data_imp = improved_agent.execute(None)  # Data is None
    print(f"Improved - None data: {result_none_data_imp}")

    # Test logging
    improved_agent._log_message("This is a custom debug message.", logging.DEBUG)
    improved_agent._log_message("This is a custom warning message.", logging.WARNING)

    print(f"Agent name: {improved_agent.get_agent_name()}")
