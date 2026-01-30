# VAPI-Exotel Bridge

WebSocket bridge server that connects Exotel's bidirectional audio streaming to VAPI AI assistant.

## Architecture

```
Exotel Call → Exotel Voicebot → [This Bridge] → VAPI AI → Response → Bridge → Exotel → Caller
```

## Features

- ✅ Bidirectional audio forwarding (Exotel ↔ VAPI)
- ✅ Real-time WebSocket communication
- ✅ VAPI web call creation on-demand
- ✅ Call metadata tracking
- ✅ n8n webhook integration (optional)
- ✅ Production-ready logging

## Setup

### 1. Clone/Download

```bash
cd ~/Desktop/vapi-exotel-bridge
```

### 2. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```
VAPI_API_KEY=your_private_key_here
VAPI_ASSISTANT_ID=your_assistant_id_here
```

### 4. Run Locally

```bash
python app.py
```

Server will start on `ws://localhost:8080/bridge`

## Deployment

### Option A: Railway (Recommended)

1. Create account at railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Connect this repository
4. Add environment variables in Railway dashboard
5. Deploy!

Railway will give you: `wss://your-app.railway.app/bridge`

### Option B: Render

1. Create account at render.com
2. New → Web Service
3. Connect repository
4. Environment: Python 3
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `python app.py`
7. Add environment variables
8. Deploy!

### Option C: Your Own Server

```bash
# On your server
git clone <your-repo>
cd vapi-exotel-bridge
pip install -r requirements.txt
python app.py

# Or use systemd/supervisor for auto-restart
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VAPI_API_KEY` | Yes | Your VAPI private API key |
| `VAPI_ASSISTANT_ID` | Yes | VAPI assistant ID |
| `BRIDGE_HOST` | No | Server host (default: 0.0.0.0) |
| `BRIDGE_PORT` | No | Server port (default: 8080) |
| `N8N_WEBHOOK_URL` | No | n8n webhook for call results |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

## Exotel Configuration

In your Exotel Voicebot applet, set the WebSocket URL to:

```
wss://your-deployed-bridge.railway.app/bridge?CallFrom={{CallFrom}}
```

**Or** if deploying locally for testing:
```
ws://your-server-ip:8080/bridge?CallFrom={{CallFrom}}
```

## n8n Integration

### Simplified Workflow

Delete Workflow 2 (VAPI Webhook Provider) - not needed!

**Workflow 1 remains:**
```
[Gmail Trigger]
     ↓
[Code - Parse Email]
     ↓
[Google Sheets - Add Lead]
     ↓
[Exotel - Make Call] (Voicebot points to this bridge)
```

### Optional: Receive Call Results

Create a new webhook in n8n to receive call results from the bridge:

1. Add Webhook node in n8n
2. Path: `/call-results`
3. Copy webhook URL
4. Add to `.env` as `N8N_WEBHOOK_URL`

The bridge will POST call results when calls end.

## Testing

### Local Test

```bash
# Terminal 1: Start bridge
python app.py

# Terminal 2: Test with a WebSocket client
# Use: https://www.piesocket.com/websocket-tester
# Connect to: ws://localhost:8080/bridge?CallFrom=9116430391
```

### Production Test

1. Deploy to Railway/Render
2. Update Exotel Voicebot URL
3. Call your Exotel number
4. Check logs for connection status

## Logs

Logs show:
- ✅ Incoming Exotel connections
- ✅ VAPI call creation
- ✅ WebSocket connection status
- ✅ Audio forwarding activity
- ✅ Errors and exceptions

## Troubleshooting

### Bridge won't start
- Check `.env` file exists and has correct values
- Verify VAPI_API_KEY and VAPI_ASSISTANT_ID are set

### Exotel not connecting
- Verify Exotel Voicebot URL is correct (wss:// protocol)
- Check bridge server is running and accessible
- Review Exotel logs for connection errors

### No audio flowing
- Check VAPI logs for "assistant did not receive customer audio"
- Verify WebSocket connections are established (check logs)
- Ensure firewall allows WebSocket connections

### VAPI call creation fails
- Verify VAPI_API_KEY is correct (private key, not public)
- Check VAPI_ASSISTANT_ID exists in your VAPI dashboard
- Review bridge logs for VAPI API error messages

## Support

For issues:
1. Check logs: `python app.py` shows detailed logging
2. Verify environment variables
3. Test VAPI assistant separately
4. Check Exotel configuration

## License

MIT
