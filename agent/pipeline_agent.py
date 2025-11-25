import logging
import asyncio
import os
from livekit.agents import cli, WorkerOptions, JobContext, voice, llm
from livekit import rtc
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import List, Any
from agent.vosk_stt import VoskSTT

logger = logging.getLogger(__name__)

# Increase logging level for turn detector to show predictions
turn_detector_logger = logging.getLogger("livekit.plugins.turn_detector")
turn_detector_logger.setLevel(logging.INFO)

class PipelineAgent:
    """
    Standard pipeline agent using separate STT, LLM, and TTS components.
    This provides more control over turn detection and the processing pipeline.

    Pipeline: Audio → STT (Whisper) → Text → LLM (GPT-4) → Text → TTS → Audio
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
            logger.info(f"🔧 LLM is calling MCP tool '{tool_name}'")
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

        # Create the standard pipeline components
        logger.info("Creating standard pipeline components...")
        logger.info("  🎙️  VAD: Silero (signals silence)")
        logger.info("  🎤 STT: Vosk (LOCAL, ultra-low latency streaming)")
        logger.info("  🧠 LLM: OpenAI GPT-4o")
        logger.info("  🔊 TTS: OpenAI TTS")
        logger.info("  🧩 Turn Detection: MultilingualModel (text-based semantic)")

        # Use Silero VAD with DEFAULT settings (matches LiveKit examples)
        # Default min_silence_duration = 0.55s triggers turn detection checks
        # Turn detector then decides based on TEXT if user is actually done
        vad = silero.VAD.load()  # Use defaults!

        # Create STT using Vosk for ULTRA-LOW LATENCY streaming (~100ms updates)
        # This should finally give responsive turn detection!
        model_path = "/Users/ryan.porter/Projects/LiveKit-MCP-client-demo/models/vosk-model-small-en-us-0.15"
        stt = VoskSTT(
            model_path=model_path,
            language="en",
        )

        # Create LLM (Language Model) using OpenAI GPT-4
        llm_model = openai.LLM(model="gpt-4o")

        # Create TTS (Text-to-Speech) using OpenAI TTS
        tts = openai.TTS(voice="alloy")

        # Create semantic turn detector - this is a TEXT-BASED ML model
        # It analyzes the transcribed text to determine when user is done speaking
        # unlikely_threshold: probability below this = "unlikely done" = keep waiting
        # Setting to 0.15 means agent responds when probability >= 0.15 (15% confidence)
        turn_detector = MultilingualModel(unlikely_threshold=0.15)

        # Create an Agent with the pipeline components and tools
        logger.info("Creating agent with standard pipeline...")

        if all_functions:
            agent = voice.Agent(
                instructions="You are a helpful assistant with access to tools. Use the available tools when appropriate to help the user. Be friendly and concise in your responses.",
                tools=all_functions,
            )
            logger.info(f"🎯 Agent created with {len(all_functions)} tools available")
        else:
            agent = voice.Agent(
                instructions="You are a helpful assistant. Be friendly and concise in your responses.",
            )
            logger.info("🎯 Agent created without tools")

        logger.info("Creating agent session...")
        # Create the agent session with semantic turn detection
        # KEY: Pass turn_detector directly (not a string mode)
        # This makes LiveKit use the text-based MultilingualModel for turn detection
        # VAD is only used to detect silence (when you stop speaking), NOT for turn decisions
        session = voice.AgentSession(
            turn_detection=turn_detector,  # MultilingualModel instance for text analysis
            stt=stt,
            vad=vad,  # VAD signals silence to STT, but doesn't control turn decisions
            llm=llm_model,
            tts=tts,
            # Delays for turn detection
            min_endpointing_delay=0.8,  # Respond 0.8s after high probability
            max_endpointing_delay=30.0,  # Wait up to 30s when probability is low
        )

        # Add event handlers for turn detection logging
        @session.on("user_started_speaking")
        def on_user_started_speaking():
            logger.info("🎙️  User started speaking")

        @session.on("user_stopped_speaking")
        def on_user_stopped_speaking():
            logger.info("🔇 User stopped speaking (VAD detected silence)")

        @session.on("user_speech_committed")
        def on_user_speech_committed(msg):
            logger.info("=" * 80)
            logger.info(f"📝 FINAL TRANSCRIPT: '{msg.text}'")
            logger.info("   ⏳ Turn detector is now analyzing if user is done speaking...")
            logger.info("=" * 80)

        # NEW: Log interim transcripts to show streaming is working
        @session.on("user_speech_transcribed")
        def on_user_speech_transcribed(msg):
            logger.info(f"💬 INTERIM: '{msg.text}'")

        @session.on("agent_started_speaking")
        def on_agent_started_speaking():
            logger.info("🤖 Agent started speaking")

        @session.on("agent_stopped_speaking")
        def on_agent_stopped_speaking():
            logger.info("✅ Agent finished speaking")

        @session.on("function_calls_collected")
        def on_function_calls_collected(msg):
            logger.info("=" * 80)
            logger.info("🎯 TURN DETECTOR TRIGGERED - Agent will now respond!")
            logger.info("=" * 80)
            if msg.function_calls:
                logger.info(f"   Function calls requested: {[fc.name for fc in msg.function_calls]}")

        # Start the session with the agent and room
        logger.info("Starting agent session...")
        await session.start(agent, room=ctx.room)

        logger.info("🤖 Agent is live and ready to chat!")
        logger.info("   Pipeline: Audio → Vosk (local streaming) → GPT-4o → TTS → Audio")
        logger.info("   STT: Vosk running locally with ~100ms latency")
        logger.info("   Turn detection: MultilingualModel (semantic, text-based)")
        logger.info("")
        logger.info("📊 Turn Detector Configuration:")
        logger.info("   - Mode: TEXT-BASED semantic analysis")
        logger.info("   - Threshold: 0.15 (agent responds when probability >= 15%)")
        logger.info("   - Response delay: 0.8s after high probability")
        logger.info("   - STT: TRUE STREAMING with ~100ms partial results (Vosk)")
        logger.info("   - VAD: 0.55s silence triggers turn detection check")
        logger.info("   - Turn detector analyzes TEXT to decide if you're done")
        logger.info("   - Watch for 'eou prediction' logs showing probabilities")
        logger.info("")

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
