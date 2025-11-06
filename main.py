import cv2
import numpy as np
import uvicorn
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import your custom modules
import PoseModule as pm
from process_pushup import process_frame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# App Initialization
app = FastAPI(title="AI Pushup Trainer API")

# CORS Middleware - Allow your Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector (shared across connections)
detector = pm.poseDetector()
logger.info("✓ Pose detector initialized")

@app.get("/")
def root():
    return {
        "message": "Welcome to the AI Pushup Trainer API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/live_pushup",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "detector": "ready"}

@app.websocket("/ws/live_pushup")
async def websocket_live_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live pushup analysis.
    Receives binary JPEG frames, processes them, and sends back:
    1. Processed frame with overlay
    2. JSON data with count and feedback
    """
    client_id = id(websocket)
    logger.info(f"📥 Client {client_id} attempting to connect")
    
    await websocket.accept()
    logger.info(f"✅ Client {client_id} connected successfully")
    
    # State variables for this specific user
    count = 0
    direction = 0
    form = 0
    frame_count = 0
    error_count = 0
    max_errors = 10
    
    # Session tracking
    session_start_time = datetime.utcnow()
    correct_reps = 0
    incorrect_reps = 0
    
    try:
        while True:
            # 1. Receive Frame from Client
            try:
                data = await websocket.receive_bytes()
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"📸 Client {client_id}: Processed {frame_count} frames (count={count})")
                
            except Exception as recv_error:
                logger.error(f"❌ Client {client_id}: Error receiving data: {recv_error}")
                break
            
            # 2. Decode Frame
            try:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    logger.warning(f"⚠️ Client {client_id}: Failed to decode frame #{frame_count}")
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error(f"❌ Client {client_id}: Too many decode errors, closing connection")
                        break
                    continue
                
                error_count = 0
                
            except Exception as decode_error:
                logger.error(f"❌ Client {client_id}: Decode error: {decode_error}")
                error_count += 1
                if error_count >= max_errors:
                    break
                continue

            # 3. Process the Frame
            try:
                processed_img, new_count, new_dir, new_form, feedback = process_frame(
                    frame, detector, count, direction, form
                )
                
                # 4. Update the state
                if new_count != count:
                    logger.info(f"💪 Client {client_id}: Pushup count updated: {count} → {new_count}")
                    # Determine if it was correct or incorrect
                    if new_form == 1:
                        correct_reps += 1
                    else:
                        incorrect_reps += 1
                
                count, direction, form = new_count, new_dir, new_form

            except Exception as process_error:
                logger.error(f"❌ Client {client_id}: Processing error: {process_error}")
                processed_img = frame.copy()
                cv2.putText(processed_img, "Processing Error", (50, 50), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                feedback = "Error"

            # 5. Encode Frame Back to JPEG
            try:
                success, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if not success:
                    logger.error(f"❌ Client {client_id}: Failed to encode frame #{frame_count}")
                    continue
                
            except Exception as encode_error:
                logger.error(f"❌ Client {client_id}: Encode error: {encode_error}")
                continue
            
            # 6. Send Processed Frame Back to Client
            try:
                await websocket.send_bytes(buffer.tobytes())
                
                # 7. Also send JSON data with metrics (as text message)
                # The client will need to handle both binary and text messages
                json_data = {
                    "count": int(count),
                    "feedback": feedback,
                    "form": int(form),
                    "correct_reps": correct_reps,
                    "incorrect_reps": incorrect_reps,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Send JSON data every 10 frames to reduce overhead
                if frame_count % 10 == 0:
                    import json
                    await websocket.send_text(json.dumps(json_data))
                
            except Exception as send_error:
                logger.error(f"❌ Client {client_id}: Error sending frame: {send_error}")
                break

    except WebSocketDisconnect:
        logger.info(f"👋 Client {client_id} disconnected normally")
        # Calculate session duration
        session_duration = (datetime.utcnow() - session_start_time).total_seconds()
        logger.info(f"📊 Session Stats - Frames: {frame_count}, Reps: {count}, Duration: {session_duration}s")
    except Exception as e:
        logger.error(f"❌ Client {client_id}: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
    finally:
        logger.info(f"🔌 Client {client_id} session ended. Stats: {frame_count} frames, {count} reps")

# Run the API
if __name__ == "__main__":
    logger.info("🚀 Starting Pushup Analyzer API...")
    logger.info("📡 WebSocket endpoint: ws://0.0.0.0:8000/ws/live_pushup")
    logger.info("🌐 HTTP endpoint: http://0.0.0.0:8000")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
