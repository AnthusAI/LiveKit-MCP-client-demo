import subprocess
import json
import logging
import os

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.process = None
        self.request_id = 0

    def connect(self):
        cmd = [self.config['command']] + self.config.get('args', [])
        env = os.environ.copy()
        if 'env' in self.config:
            env.update(self.config['env'])

        try:
            logger.info(f"🚀 Starting MCP server '{self.name}'")
            logger.debug(f"   Command: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1  # Line buffered
            )
            logger.info(f"✅ MCP server '{self.name}' process started (PID: {self.process.pid})")

            # Check for any stderr output (errors/warnings during startup)
            import select
            import time
            time.sleep(0.1)  # Give process time to output startup errors
            if self.process.stderr and select.select([self.process.stderr], [], [], 0)[0]:
                stderr_line = self.process.stderr.readline()
                if stderr_line:
                    logger.warning(f"   Server stderr: {stderr_line.strip()}")

            # Send initialize request
            logger.debug(f"   Sending initialize request to '{self.name}'")
            init_result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "livekit-mcp-client",
                    "version": "1.0.0"
                }
            })
            logger.info(f"✅ MCP server '{self.name}' initialized successfully")
            logger.debug(f"   Server capabilities: {init_result.get('capabilities', {})}")
            logger.debug(f"   Server info: {init_result.get('serverInfo', {})}")

            # Send initialized notification (no response expected)
            logger.debug(f"   Sending initialized notification to '{self.name}'")
            self._send_notification("notifications/initialized")

        except Exception as e:
            logger.error(f"❌ Failed to start MCP server '{self.name}': {e}")
            # Check stderr for error details
            if self.process and self.process.stderr:
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    logger.error(f"   Server stderr: {stderr_output}")
            raise

    def _send_request(self, method, params=None):
        if not self.process:
            raise Exception("Not connected")

        self.request_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params is not None:
            req["params"] = params

        input_str = json.dumps(req) + "\n"
        logger.debug(f"   Sending to '{self.name}': {input_str.strip()}")
        self.process.stdin.write(input_str)
        self.process.stdin.flush()

        # Read response (blocking for simplicity)
        response_line = self.process.stdout.readline()
        if not response_line:
            # Check stderr for any error messages
            import select
            if self.process.stderr and select.select([self.process.stderr], [], [], 0.1)[0]:
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    logger.error(f"   Server stderr: {stderr_output}")
            raise Exception("Server closed connection")

        logger.debug(f"   Received from '{self.name}': {response_line.strip()}")
        response = json.loads(response_line)
        if "error" in response:
            # Log stderr if there's an error
            import select
            if self.process.stderr and select.select([self.process.stderr], [], [], 0.1)[0]:
                stderr_line = self.process.stderr.readline()
                if stderr_line:
                    logger.debug(f"   Server stderr: {stderr_line.strip()}")
            raise Exception(f"MCP Error: {response['error']}")

        return response.get("result")

    def _send_notification(self, method, params=None):
        """Send a JSON-RPC notification (no response expected)"""
        if not self.process:
            raise Exception("Not connected")

        req = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            req["params"] = params

        input_str = json.dumps(req) + "\n"
        logger.debug(f"   Sending notification to '{self.name}': {input_str.strip()}")
        self.process.stdin.write(input_str)
        self.process.stdin.flush()

    def list_tools(self):
        try:
            logger.debug(f"   Requesting tool list from '{self.name}'")
            # Don't pass params - Plexus expects no params field at all
            result = self._send_request("tools/list")
            tools = result.get("tools", [])
            logger.info(f"📋 Found {len(tools)} tools from '{self.name}'")
            for tool in tools:
                logger.debug(f"   - {tool.get('name')}: {tool.get('description', 'No description')}")
            return tools
        except Exception as e:
            logger.error(f"❌ Error listing tools from '{self.name}': {e}")
            return []
            
    def call_tool(self, name, arguments):
        try:
            logger.info(f"🔧 Calling tool '{name}' on '{self.name}'")
            logger.debug(f"   Arguments: {arguments}")
            result = self._send_request("tools/call", {"name": name, "arguments": arguments})
            logger.info(f"✅ Tool '{name}' completed successfully")
            logger.debug(f"   Result: {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Error calling tool '{name}' on '{self.name}': {e}")
            raise


