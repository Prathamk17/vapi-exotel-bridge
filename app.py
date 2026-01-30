"""
VAPI-Exotel Bridge Server
WebSocket server that bridges Exotel calls to VAPI AI assistant
"""
import asyncio
import logging
import websockets
from bridge import ExotelVAPIBridge
from config import BRIDGE_HOST, BRIDGE_PORT, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Start the WebSocket bridge server
    """
    bridge = ExotelVAPIBridge()

    logger.info("=" * 60)
    logger.info("VAPI-Exotel Bridge Server")
    logger.info("=" * 60)
    logger.info(f"Starting WebSocket server on {BRIDGE_HOST}:{BRIDGE_PORT}")
    logger.info(f"Endpoint: ws://{BRIDGE_HOST}:{BRIDGE_PORT}/bridge")
    logger.info("=" * 60)

    # Start WebSocket server
    async with websockets.serve(
        bridge.handle_call,
        BRIDGE_HOST,
        BRIDGE_PORT,
        ping_interval=20,
        ping_timeout=10
    ):
        logger.info("✅ Bridge server is running")
        logger.info("Waiting for Exotel connections...")

        # Run forever
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
