"""
WebSocket Bridge: Exotel ↔ VAPI
Forwards audio bidirectionally between Exotel's Voicebot and VAPI AI
"""
import asyncio
import json
import logging
import websockets
import aiohttp
from typing import Optional
from config import VAPI_API_KEY, VAPI_ASSISTANT_ID, VAPI_API_URL, N8N_WEBHOOK_URL

logger = logging.getLogger(__name__)


class ExotelVAPIBridge:
    def __init__(self):
        self.vapi_api_key = VAPI_API_KEY
        self.vapi_assistant_id = VAPI_ASSISTANT_ID
        self.active_calls = {}  # Track active bridge connections

    async def create_vapi_call(self, call_metadata: dict) -> str:
        """
        Create a VAPI web call and return the WebSocket listen URL

        Args:
            call_metadata: Metadata about the call (caller number, etc.)

        Returns:
            WebSocket URL to connect to VAPI
        """
        url = f"{VAPI_API_URL}/call/web"
        headers = {
            "Authorization": f"Bearer {self.vapi_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "assistantId": self.vapi_assistant_id,
            "assistantOverrides": {
                "variableValues": {
                    "lead_name": call_metadata.get("lead_name", "Customer"),
                    "property_type": call_metadata.get("property_type", "Property"),
                    "location": "Kharadi, Pune",
                    "budget": "Not specified"
                }
            },
            "metadata": {
                "source": "exotel",
                "caller": call_metadata.get("caller", "unknown")
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    logger.error(f"Failed to create VAPI call: {response.status} - {error_text}")
                    raise Exception(f"VAPI API error: {response.status}")

                data = await response.json()
                listen_url = data.get("monitor", {}).get("listenUrl")

                if not listen_url:
                    logger.error(f"No listenUrl in VAPI response: {data}")
                    raise Exception("VAPI did not return a listen URL")

                logger.info(f"Created VAPI call: {data.get('id')}")
                logger.info(f"WebSocket URL: {listen_url}")

                return listen_url, data.get("id")

    async def forward_audio(self, source_ws, dest_ws, direction: str):
        """
        Forward audio/messages from source WebSocket to destination WebSocket

        Args:
            source_ws: Source WebSocket
            dest_ws: Destination WebSocket
            direction: "exotel_to_vapi" or "vapi_to_exotel"
        """
        try:
            async for message in source_ws:
                if dest_ws.open:
                    # Check if message is text (control message) or binary (audio)
                    if isinstance(message, bytes):
                        # Binary audio data - forward as-is
                        await dest_ws.send(message)
                        logger.debug(f"Forwarded {len(message)} bytes {direction}")
                    else:
                        # Text message (JSON control messages)
                        try:
                            msg_data = json.loads(message)
                            logger.info(f"{direction} control message: {msg_data}")

                            # Forward control messages
                            await dest_ws.send(message)
                        except json.JSONDecodeError:
                            # Not JSON, forward as-is
                            await dest_ws.send(message)
                else:
                    logger.warning(f"Destination WebSocket closed, stopping {direction} forward")
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {direction}")
        except Exception as e:
            logger.error(f"Error forwarding {direction}: {e}")

    async def send_call_results(self, call_id: str, results: dict):
        """
        Send call results to n8n webhook (optional)
        """
        if not N8N_WEBHOOK_URL:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(N8N_WEBHOOK_URL, json=results) as response:
                    if response.status == 200:
                        logger.info(f"Sent call results to n8n for call {call_id}")
                    else:
                        logger.warning(f"Failed to send results to n8n: {response.status}")
        except Exception as e:
            logger.error(f"Error sending results to n8n: {e}")

    async def handle_call(self, exotel_ws, path):
        """
        Main handler for each incoming Exotel call

        Args:
            exotel_ws: WebSocket connection from Exotel
            path: WebSocket path (can contain query parameters)
        """
        call_id = None
        vapi_ws = None

        try:
            # Extract call metadata from path/query params if available
            # Format: /bridge?CallFrom=9116430391&CallTo=01414939962
            call_metadata = {"caller": "unknown", "lead_name": "Customer"}

            logger.info(f"New Exotel connection from {exotel_ws.remote_address}")
            logger.info(f"Path: {path}")

            # Parse query params if present
            if "?" in path:
                query_string = path.split("?")[1]
                params = dict(param.split("=") for param in query_string.split("&") if "=" in param)
                call_metadata["caller"] = params.get("CallFrom", "unknown")
                logger.info(f"Call from: {call_metadata['caller']}")

            # Step 1: Create VAPI call
            logger.info("Creating VAPI web call...")
            vapi_ws_url, call_id = await self.create_vapi_call(call_metadata)

            # Step 2: Connect to VAPI WebSocket
            logger.info(f"Connecting to VAPI WebSocket: {vapi_ws_url}")
            vapi_ws = await websockets.connect(vapi_ws_url)
            logger.info("Connected to VAPI WebSocket")

            # Track active call
            self.active_calls[call_id] = {
                "exotel_ws": exotel_ws,
                "vapi_ws": vapi_ws,
                "start_time": asyncio.get_event_loop().time()
            }

            # Step 3: Forward audio bidirectionally
            logger.info("Starting bidirectional audio forwarding...")
            await asyncio.gather(
                self.forward_audio(exotel_ws, vapi_ws, "exotel_to_vapi"),
                self.forward_audio(vapi_ws, exotel_ws, "vapi_to_exotel")
            )

        except Exception as e:
            logger.error(f"Error in bridge handler: {e}", exc_info=True)
        finally:
            # Cleanup
            if call_id:
                end_time = asyncio.get_event_loop().time()
                duration = end_time - self.active_calls.get(call_id, {}).get("start_time", end_time)
                logger.info(f"Call {call_id} ended. Duration: {duration:.2f}s")

                # Send results to n8n
                await self.send_call_results(call_id, {
                    "call_id": call_id,
                    "duration": duration,
                    "status": "completed"
                })

                # Remove from active calls
                if call_id in self.active_calls:
                    del self.active_calls[call_id]

            # Close connections
            if vapi_ws and vapi_ws.open:
                await vapi_ws.close()
                logger.info("Closed VAPI WebSocket")

            logger.info("Bridge handler finished")
