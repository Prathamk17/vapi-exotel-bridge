"""
WebSocket Bridge: Exotel ↔ VAPI
Forwards audio bidirectionally between Exotel's Voicebot and VAPI AI
Handles protocol translation between Exotel's JSON/base64 format and VAPI's raw binary format
"""
import asyncio
import json
import logging
import websockets
import aiohttp
import base64
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

    async def forward_exotel_to_vapi(self, exotel_ws, vapi_ws):
        """
        Forward audio from Exotel to VAPI with protocol translation
        Exotel sends: JSON messages with base64-encoded audio
        VAPI expects: Raw binary PCM audio
        """
        media_count = 0
        try:
            async for message in exotel_ws:
                if not vapi_ws.open:
                    logger.warning("VAPI WebSocket closed, stopping exotel_to_vapi forward")
                    break

                # Exotel sends text messages with JSON
                if isinstance(message, str):
                    try:
                        msg_data = json.loads(message)
                        event_type = msg_data.get("event")

                        # Enhanced logging: Log ALL message structures
                        logger.info(f"📥 Exotel message: event={event_type}, keys={list(msg_data.keys())}")

                        if event_type == "media":
                            # Log full media structure for debugging
                            logger.info(f"🎵 Media event structure: {json.dumps(msg_data, indent=2)[:500]}")

                            # Extract and decode audio payload
                            media_payload = msg_data.get("media", {}).get("payload")
                            if media_payload:
                                # Decode base64 audio
                                audio_data = base64.b64decode(media_payload)
                                # Send raw binary to VAPI
                                await vapi_ws.send(audio_data)
                                media_count += 1
                                logger.info(f"✅ Forwarded {len(audio_data)} bytes audio to VAPI (chunk #{media_count})")
                            else:
                                logger.warning(f"⚠️  Media event has no payload! Full message: {msg_data}")
                        elif event_type == "connected":
                            logger.info(f"Exotel control: connected - {msg_data}")
                        elif event_type == "start":
                            logger.info(f"Exotel control: start - {msg_data}")
                            # Send acknowledgment to Exotel to start media streaming
                            ack_message = {
                                "event": "start",
                                "stream_sid": msg_data.get("stream_sid", "default"),
                                "media_format": msg_data.get("media_format", {})
                            }
                            await exotel_ws.send(json.dumps(ack_message))
                            logger.info(f"📤 Sent start acknowledgment to Exotel")
                        elif event_type == "stop":
                            logger.info(f"Exotel stream stopped. Total media chunks received: {media_count}")
                            break
                        else:
                            logger.info(f"Exotel event: {event_type} - {msg_data}")
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message from Exotel: {message[:100]}")
                    except Exception as e:
                        logger.error(f"Error processing Exotel message: {e}", exc_info=True)
                else:
                    logger.warning(f"Unexpected binary message from Exotel (length: {len(message)})")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Exotel WebSocket connection closed. Total media chunks: {media_count}")
        except Exception as e:
            logger.error(f"Error in exotel_to_vapi forward: {e}", exc_info=True)

    async def forward_vapi_to_exotel(self, vapi_ws, exotel_ws):
        """
        Forward audio from VAPI to Exotel with protocol translation
        VAPI sends: Raw binary PCM audio
        Exotel expects: JSON messages with base64-encoded audio
        """
        sequence_number = 1
        audio_count = 0
        try:
            async for message in vapi_ws:
                if not exotel_ws.open:
                    logger.warning("Exotel WebSocket closed, stopping vapi_to_exotel forward")
                    break

                # VAPI sends binary audio data
                if isinstance(message, bytes):
                    # Encode to base64 and wrap in Exotel format
                    audio_base64 = base64.b64encode(message).decode('utf-8')
                    exotel_message = {
                        "event": "media",
                        "stream_sid": "vapi_stream",
                        "sequence_number": str(sequence_number),
                        "media": {
                            "chunk": str(sequence_number),
                            "timestamp": str(sequence_number * 20),  # Approximate timestamp
                            "payload": audio_base64
                        }
                    }
                    await exotel_ws.send(json.dumps(exotel_message))
                    audio_count += 1
                    logger.info(f"🔊 Forwarded {len(message)} bytes audio to Exotel (chunk #{audio_count})")
                    sequence_number += 1

                # VAPI also sends text control messages
                elif isinstance(message, str):
                    try:
                        msg_data = json.loads(message)
                        logger.info(f"📨 VAPI control message: {msg_data}")
                        # Could translate VAPI control messages to Exotel format if needed
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON text from VAPI: {message[:100]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"VAPI WebSocket connection closed. Total audio chunks sent: {audio_count}")
        except Exception as e:
            logger.error(f"Error in vapi_to_exotel forward: {e}", exc_info=True)

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

            # Step 3: Forward audio bidirectionally with protocol translation
            logger.info("Starting bidirectional audio forwarding with protocol translation...")
            await asyncio.gather(
                self.forward_exotel_to_vapi(exotel_ws, vapi_ws),
                self.forward_vapi_to_exotel(vapi_ws, exotel_ws)
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
