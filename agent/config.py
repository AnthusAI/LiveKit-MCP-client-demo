import logging
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """
    Load MCP configuration from local YAML config file.
    Returns a dictionary with 'mcpServers' key.
    """
    logger.info("Looking for MCP configuration...")

    # Look for mcp_config.yaml
    local_path = os.path.abspath("mcp_config.yaml")
    if os.path.exists(local_path):
        logger.info(f"Found local config at: {local_path}")
        try:
            with open(local_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'mcpServers' in config:
                    return config
        except Exception as e:
            logger.error(f"Error reading local config: {e}")

    # No config found
    logger.warning("No MCP configuration found.")
    logger.info("Please create a configuration file at mcp_config.yaml")
    logger.info("Example format:")
    logger.info("""
mcpServers:
  example-server:
    command: python
    args:
      - -m
      - example_module
    env:
      SOME_VAR: value
""")

    return {"mcpServers": {}}
