import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_feedback_and_count(elbow, shoulder, hip, direction, count, form):
    """
    Determines the feedback message and updates the count based on the angles.
    Returns: feedback, count, direction, form
    """
    feedback = "Fix Form"
    
    # Check if proper starting position (arms extended, body straight)
    if elbow > 160 and shoulder > 40 and hip > 160:
        form = 1
    
    if form == 1:
        # Down position - elbow bent, hip still straight
        if elbow <= 90 and hip > 160:
            feedback = "Good - Push Up"
            if direction == 0:
                count += 0.5
                direction = 1
        # Up position - arms extended, body straight  
        elif elbow > 160 and shoulder > 40 and hip > 160:
            feedback = "Good - Go Down"
            if direction == 1:
                count += 0.5
                direction = 0
        else:
            feedback = "Fix Form"
    else:
        feedback = "Get into starting position"
    
    return feedback, count, direction, form

def draw_ui(img, per, bar, count, feedback, form, elbow_angle=0, shoulder_angle=0, hip_angle=0):
    """Draws the UI elements on the image."""
    try:
        h, w = img.shape[:2]
        
        # Progress bar (only when form is correct)
        if form == 1:
            bar_x = w - 40
            cv2.rectangle(img, (bar_x, 50), (bar_x + 20, 380), (0, 255, 0), 3)
            cv2.rectangle(img, (bar_x, int(bar)), (bar_x + 20, 380), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (bar_x - 15, 430), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Pushup counter (bottom left)
        cv2.rectangle(img, (10, h - 100), (150, h - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Reps: {int(count)}', (20, h - 50), 
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
        
        # Feedback text (top center)
        feedback_color = (0, 255, 0) if form == 1 else (0, 0, 255)
        text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_PLAIN, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        
        # Background for feedback text
        cv2.rectangle(img, (text_x - 10, 10), (text_x + text_size[0] + 10, 60), 
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(img, feedback, (text_x, 45), 
                    cv2.FONT_HERSHEY_PLAIN, 2, feedback_color, 3)
        
        # Show angles (top left) - smaller and less intrusive
        angle_text_y = h - 120
        cv2.putText(img, f'E:{int(elbow_angle)}', (10, angle_text_y), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f'S:{int(shoulder_angle)}', (10, angle_text_y + 25), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f'H:{int(hip_angle)}', (10, angle_text_y + 50), 
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
    except Exception as e:
        logger.error(f"Error drawing UI: {e}")

def process_frame(img, detector, count, direction, form):
    """
    Processes a single frame for pushup analysis.
    Returns: processed_image, count, direction, form, feedback
    """
    feedback = "Move into frame"
    per, bar = 0, 0
    elbow_angle = 0
    shoulder_angle = 0
    hip_angle = 0
    
    # Validate input image
    if img is None or img.size == 0:
        logger.error("⚠️ Invalid/empty image received")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Invalid Frame", (200, 240), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        return error_img, count, direction, form, feedback
    
    try:
        img_copy = img.copy()
        
        # 1. Find pose (populates detector.results)
        img_copy = detector.findPose(img_copy, draw=True)
        
        # 2. Find position (populates detector.lmList)
        lmList = detector.findPosition(img_copy, draw=False)

        if len(lmList) != 0:
            try:
                # 3. Find angles - Using RIGHT side landmarks for side view
                # Elbow angle: shoulder(12) - elbow(14) - wrist(16)
                elbow_angle = detector.findAngle(img_copy, 12, 14, 16, draw=True)
                # Shoulder angle: elbow(14) - shoulder(12) - hip(24)
                shoulder_angle = detector.findAngle(img_copy, 14, 12, 24, draw=True)
                # Hip angle: shoulder(12) - hip(24) - knee(26)
                hip_angle = detector.findAngle(img_copy, 12, 24, 26, draw=True)
                
                # 4. Calculate progress percentage
                per = np.interp(elbow_angle, (90, 160), (0, 100))
                bar = np.interp(elbow_angle, (90, 160), (380, 50))
                
                # 5. Run logic to update count and feedback
                feedback, count, direction, form = update_feedback_and_count(
                    elbow_angle, shoulder_angle, hip_angle, direction, count, form
                )
                
                logger.debug(f"✓ Processed: count={count}, feedback={feedback}")
            except Exception as angle_error:
                logger.error(f"Error calculating angles: {angle_error}")
                feedback = "Error detecting pose"
        else:
            feedback = "Move into frame"
            logger.debug("No pose detected in frame")
    
    except Exception as e:
        logger.error(f"❌ Critical error in process_frame: {e}")
        import traceback
        traceback.print_exc()
        feedback = "Processing error"
        try:
            cv2.putText(img, "Error", (50, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            return img, count, direction, form, feedback
        except:
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error", (200, 240), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            return error_img, count, direction, form, feedback
    
    # 6. Always draw the UI on the image
    draw_ui(img_copy, per, bar, count, feedback, form, elbow_angle, shoulder_angle, hip_angle)
    
    # 7. Return the modified image and the new state
    return img_copy, count, direction, form, feedback
