import os
import sys
import logging
from dotenv import load_dotenv
from agent.config import load_config
from agent.mcp_client import MCPClient
from agent.realtime_agent import RealtimeAgent
from agent.pipeline_agent import PipelineAgent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure turn detector logs are visible
turn_detector_logger = logging.getLogger("livekit.plugins.turn_detector")
turn_detector_logger.setLevel(logging.DEBUG)
turn_detector_logger.propagate = True

def main():
    load_dotenv()

    # Check for LiveKit CLI commands that should pass through directly
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        # For download-files, create a pipeline agent without MCP setup
        # This allows the LiveKit CLI to handle downloading model files
        logger.info("=" * 60)
        logger.info("📥 Downloading model files...")
        logger.info("=" * 60)

        url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
        api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")

        agent = PipelineAgent(url, api_key, api_secret)
        agent.mcp_clients = []  # No MCP clients for download
        agent.start()
        return

    # Check command line arguments for agent type
    agent_type = "realtime"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ["console", "dev"]:
            # These are LiveKit CLI modes, check next arg
            if len(sys.argv) > 2:
                agent_type = sys.argv[2]
                # Remove the agent type from argv so LiveKit CLI doesn't see it
                sys.argv.pop(2)
        else:
            agent_type = sys.argv[1]

    # Validate agent type
    if agent_type not in ["realtime", "pipeline"]:
        logger.error(f"❌ Invalid agent type: {agent_type}")
        logger.info("Usage: python -m agent.main [console|dev] [realtime|pipeline]")
        logger.info("       python -m agent.main download-files")
        logger.info("  realtime: Uses OpenAI Realtime API (audio streaming)")
        logger.info("  pipeline: Uses standard STT → LLM → TTS pipeline")
        logger.info("  download-files: Download model files for pipeline agent")
        sys.exit(1)

    # 1. Load Config
    logger.info("=" * 60)
    logger.info(f"🚀 LiveKit MCP Agent Starting ({agent_type.upper()} mode)")
    logger.info("=" * 60)

    config = load_config()

    # 2. Initialize MCP Clients
    logger.info("")
    logger.info("=" * 60)
    logger.info("🔌 Initializing MCP Clients")
    logger.info("=" * 60)

    mcp_clients = []
    server_configs = config.get('mcpServers', {})

    if not server_configs:
        logger.warning("⚠️  No MCP servers configured in mcp_config.yaml")
    else:
        logger.info(f"Found {len(server_configs)} MCP server(s) in config")

    for name, server_config in server_configs.items():
        try:
            logger.info(f"Connecting to MCP server: {name}")
            client = MCPClient(name, server_config)
            client.connect()
            mcp_clients.append(client)
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server '{name}': {e}")
            import traceback
            traceback.print_exc()

    if mcp_clients:
        logger.info("=" * 60)
        logger.info(f"✅ Successfully connected to {len(mcp_clients)} MCP server(s)")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️  No MCP servers connected")
        logger.warning("=" * 60)
            
    # 3. Start LiveKit Agent
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not url or not api_key or not api_secret:
        logger.warning("LiveKit credentials missing in .env, using defaults for local dev")
        if not url: url = "ws://localhost:7880"
        if not api_key: api_key = "devkey"
        if not api_secret: api_secret = "secret"

    # Create the appropriate agent based on type
    if agent_type == "realtime":
        agent = RealtimeAgent(url, api_key, api_secret)
        logger.info("Using OpenAI Realtime API (audio streaming)")
    else:  # pipeline
        agent = PipelineAgent(url, api_key, api_secret)
        logger.info("Using standard pipeline (STT → LLM → TTS)")

    agent.mcp_clients = mcp_clients

    logger.info("")
    logger.info("=" * 60)
    logger.info("🎙️  Starting LiveKit Agent Worker")
    logger.info("=" * 60)
    logger.info(f"Agent Type: {agent_type.upper()}")
    logger.info(f"LiveKit URL: {url}")
    logger.info(f"MCP Clients Attached: {len(mcp_clients)}")
    logger.info("=" * 60)

    try:
        agent.start()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("🛑 Stopping agent...")
        logger.info("👋 Goodbye!")

if __name__ == "__main__":
    main()


