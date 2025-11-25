import logging
import asyncio
import os
from livekit.agents import cli, WorkerOptions, JobContext, voice, llm
from livekit import rtc
from livekit.plugins import openai
from livekit.plugins.openai import realtime
from typing import List, Any

logger = logging.getLogger(__name__)

class RealtimeAgent:
    """
    Realtime agent using OpenAI's Realtime API for end-to-end audio streaming.
    This uses server-side turn detection and keeps everything in audio modality.

    Pipeline: Audio → OpenAI Realtime API → Audio (with function calling)
    """
    def __init__(self, url=None, api_key=None, api_secret=None):
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.mcp_clients = []
        self.room = None

    def _convert_mcp_tool_to_function_tool(self, client, mcp_tool):
        """Convert MCP tool to LiveKit FunctionTool"""
        tool_name = mcp_tool["name"]

        async def _tool_callback(raw_arguments: dict[str, Any]) -> Any:
            logger.info(f"🔧 OpenAI is calling MCP tool '{tool_name}'")
            logger.debug(f"   Arguments: {raw_arguments}")
            try:
                # Call the MCP tool synchronously (we're wrapping it in async)
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.call_tool(tool_name, raw_arguments)
                )
                logger.info(f"✅ MCP tool '{tool_name}' returned result")
                return result
            except Exception as e:
                logger.error(f"❌ MCP tool '{tool_name}' failed: {e}")
                raise

        raw_schema = {
            "name": mcp_tool["name"],
            "description": mcp_tool.get("description", ""),
            "parameters": mcp_tool.get("inputSchema", {}),
        }

        return llm.function_tool(_tool_callback, raw_schema=raw_schema)

    async def entrypoint(self, ctx: JobContext):
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect()
        self.room = ctx.room

        logger.info(f"Connected! Room has {len(ctx.room.remote_participants)} participants")

        # Collect MCP tools
        logger.info("=" * 60)
        logger.info("🔧 Loading MCP Tools")
        logger.info("=" * 60)

        all_functions = []

        if not self.mcp_clients:
            logger.warning("⚠️  No MCP clients configured")

        for client in self.mcp_clients:
            try:
                logger.info(f"📋 Listing tools from '{client.name}'...")
                mcp_tools = client.list_tools()

                for mcp_tool in mcp_tools:
                    tool_name = mcp_tool["name"]
                    logger.info(f"   ➕ Registering tool: {tool_name}")

                    # Convert MCP tool to LiveKit FunctionTool
                    func_tool = self._convert_mcp_tool_to_function_tool(client, mcp_tool)

                    # Add to tools list
                    all_functions.append(func_tool)
                    logger.debug(f"      Description: {mcp_tool.get('description', 'No description')}")

            except Exception as e:
                logger.error(f"❌ Failed to load tools from '{client.name}': {e}")

        logger.info("=" * 60)
        logger.info(f"✅ Registered {len(all_functions)} total tools")
        logger.info("=" * 60)

        # Create the OpenAI Realtime Model
        logger.info("Creating OpenAI Realtime Model...")
        model = realtime.RealtimeModel(
            voice="marin",
            model="gpt-4o-realtime-preview-2024-12-17",
        )

        # Create an Agent with the model and tools
        logger.info("Creating agent with tools...")

        if all_functions:
            agent = voice.Agent(
                instructions="You are a helpful assistant with access to tools. Use the available tools when appropriate to help the user. Be friendly and concise in your responses.",
                llm=model,
                tools=all_functions,
            )
            logger.info(f"🎯 Agent created with {len(all_functions)} tools available")
        else:
            agent = voice.Agent(
                instructions="You are a helpful assistant. Be friendly and concise in your responses.",
                llm=model,
            )
            logger.info("🎯 Agent created without tools")

        logger.info("Creating agent session...")
        # Create the agent session
        session = voice.AgentSession()

        # Start the session with the agent and room
        logger.info("Starting agent session...")
        await session.start(agent, room=ctx.room)

        logger.info("🤖 Agent is live and ready to chat!")

        # Keep the agent alive - session runs until the room is disconnected
        # The session handles everything internally
        while ctx.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await asyncio.sleep(1)
        
    async def send_chat(self, text):
        if self.room:
            await self.room.local_participant.publish_data(text.encode('utf-8'))

    def start(self):
        cli.run_app(WorkerOptions(
            entrypoint_fnc=self.entrypoint,
            ws_url=self.url,
            api_key=self.api_key,
            api_secret=self.api_secret
        ))
