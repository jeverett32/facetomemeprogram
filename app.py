import cv2
import mediapipe as mp
import os
import random
import numpy as np
import pygame  # For sound effects

# --- Configuration ---
MEME_DB_PATH = "meme_database"
SOUNDS_PATH = "sounds"

# --- UI Settings ---
OVERLAY_WIDTH = 213
OVERLAY_HEIGHT = 160

# --- CUSTOM SENSITIVITY (Tuned for your face) ---
# Resting Eye was ~0.049 -> Goal is now 0.065 (Must open eyes wider)
EYE_OPEN_THRESHOLD = 0.055   

# Resting Brow was ~0.104 -> Goal is now 0.120 (Must raise brows higher)
BROW_RAISED_THRESHOLD = 0.120 

# The "Guard" for Looking Up
LOOKING_UP_GUARD_LIMIT = 0.42 

# Initialize Sound Engine
pygame.mixer.init()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Helper: Load Sounds ---
def load_sounds(path):
    sound_list = []
    if not os.path.exists(path):
        os.makedirs(path)
        return []
    
    print("\n--- LOADING SOUNDS ---")
    for f in os.listdir(path):
        if f.lower().endswith(('.mp3', '.wav')):
            full_path = os.path.join(path, f)
            try:
                s = pygame.mixer.Sound(full_path)
                sound_list.append(s)
                print(f"Loaded: {f}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
    print(f"--- READY: {len(sound_list)} sounds loaded ---\n")
    return sound_list

# --- Helper: Pre-Load Memes ---
def load_meme_database(path):
    print("\n--- LOADING MEME DATABASE ---")
    meme_dict = {}
    if not os.path.exists(path):
        os.makedirs(path)
        return {}
    
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            meme_dict[folder] = []
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, f)
                    img_data = cv2.imread(img_path)
                    if img_data is not None:
                        img_data = cv2.resize(img_data, (OVERLAY_WIDTH, OVERLAY_HEIGHT))
                        meme_dict[folder].append(img_data)
            print(f"Folder '{folder}': Loaded {len(meme_dict[folder])} images.")
    return meme_dict

# --- Logic: Detect Specific Poses ---
def detect_expression(face_lm, right_hand_lm, left_hand_lm, image_shape):
    h, w, _ = image_shape
    debug_values = {} 
    
    if not face_lm: return "neutral_face", debug_values

    # Landmarks
    nose_tip = face_lm.landmark[1]
    upper_lip = face_lm.landmark[13]
    lower_lip = face_lm.landmark[14]
    left_brow = face_lm.landmark[65]
    right_brow = face_lm.landmark[295]
    left_eye_top = face_lm.landmark[159]
    right_eye_top = face_lm.landmark[386]
    left_eye_btm = face_lm.landmark[145]
    chin = face_lm.landmark[152]
    head_top = face_lm.landmark[10]

    # --- Calculations ---
    face_height = abs(chin.y - head_top.y)
    
    mouth_dist = abs(lower_lip.y - upper_lip.y)
    mouth_open_ratio = mouth_dist / face_height
    
    left_brow_dist = abs(left_brow.y - left_eye_top.y)
    right_brow_dist = abs(right_brow.y - right_eye_top.y)
    avg_brow_ratio = ((left_brow_dist + right_brow_dist) / 2) / face_height

    eye_open_dist = abs(left_eye_top.y - left_eye_btm.y)
    eye_open_ratio = eye_open_dist / face_height

    nose_vertical_pos = (nose_tip.y - head_top.y) / face_height
    
    # Debug Data
    debug_values['eye_open'] = eye_open_ratio
    debug_values['brow_height'] = avg_brow_ratio
    debug_values['nose_pos'] = nose_vertical_pos

    # --- Hand Checks ---
    active_hand = right_hand_lm if right_hand_lm else left_hand_lm
    hand_near_mouth = False
    hand_over_eyes = False
    is_fist = False
    finger_is_up = False

    if active_hand:
        hand_y = active_hand.landmark[9].y
        hand_x = active_hand.landmark[9].x
        nose_x = nose_tip.x
        
        if abs(hand_y - upper_lip.y) < 0.15: hand_near_mouth = True
        
        vertically_aligned = (hand_y < nose_tip.y) and (hand_y > head_top.y - 0.1)
        horizontally_aligned = abs(hand_x - nose_x) < 0.2
        if vertically_aligned and horizontally_aligned:
            hand_over_eyes = True

        if active_hand.landmark[8].y > active_hand.landmark[5].y - 0.02: is_fist = True
        
        index_is_straight = active_hand.landmark[8].y < active_hand.landmark[6].y
        middle_is_curled = active_hand.landmark[12].y > active_hand.landmark[10].y
        ring_is_curled = active_hand.landmark[16].y > active_hand.landmark[14].y
        if index_is_straight and middle_is_curled and ring_is_curled:
             finger_is_up = True
             is_fist = False

    # --- DECISION TREE ---
    if hand_over_eyes: return "hand_over_eyes", debug_values
    if hand_near_mouth and is_fist: return "fist_over_mouth", debug_values
    if finger_is_up: return "finger_up", debug_values
    if mouth_open_ratio > 0.05 and active_hand: return "open_mouth_hand_up", debug_values

    if nose_vertical_pos < 0.40: 
        return "looking_up", debug_values
        
    if mouth_open_ratio > 0.1 and nose_vertical_pos > 0.55: 
        return "open_mouth_face_down", debug_values

    # --- WIDE EYES CHECK (UPDATED: REQUIRES BOTH) ---
    is_looking_up_guard = nose_vertical_pos < LOOKING_UP_GUARD_LIMIT

    if not is_looking_up_guard:
        # NEW LOGIC: You must have High Brows AND Open Eyes
        # This prevents naturally high eyebrows from triggering it alone.
        if avg_brow_ratio > BROW_RAISED_THRESHOLD and eye_open_ratio > EYE_OPEN_THRESHOLD: 
            return "wide_eyes", debug_values

    return "neutral_face", debug_values

# --- Main ---
def main():
    memes = load_meme_database(MEME_DB_PATH)
    sounds = load_sounds(SOUNDS_PATH)
    cap = cv2.VideoCapture(0)
    
    current_meme_img = None
    last_pose = "neutral_face"

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            detected_pose = "neutral_face"
            debug_info = {}
            if results.face_landmarks:
                detected_pose, debug_info = detect_expression(results.face_landmarks, results.right_hand_landmarks, results.left_hand_landmarks, frame.shape)

            if detected_pose != last_pose:
                if detected_pose != "neutral_face":
                    if len(sounds) > 0:
                        pygame.mixer.stop()
                        random.choice(sounds).play()
                    
                    if detected_pose in memes and len(memes[detected_pose]) > 0:
                        current_meme_img = random.choice(memes[detected_pose])
                    
                    print(f"New Pose Detected: {detected_pose}")
                last_pose = detected_pose

            # --- UI DRAWING ---
            if current_meme_img is not None:
                y_offset = 10
                x_offset = 640 - OVERLAY_WIDTH - 10
                frame[y_offset:y_offset+OVERLAY_HEIGHT, x_offset:x_offset+OVERLAY_WIDTH] = current_meme_img
                cv2.rectangle(frame, (x_offset, y_offset), 
                              (x_offset+OVERLAY_WIDTH, y_offset+OVERLAY_HEIGHT), (0, 255, 0), 2)

            display_text = "Ready..." if detected_pose == "neutral_face" else detected_pose.replace("_", " ").upper()
            color = (200, 200, 200) if detected_pose == "neutral_face" else (0, 255, 0)
            cv2.putText(frame, display_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # --- DEBUG PANEL ---
            if 'eye_open' in debug_info:
                eye_val = round(debug_info['eye_open'], 3)
                brow_val = round(debug_info['brow_height'], 3)
                
                # Colors based on thresholds
                eye_color = (0, 255, 0) if eye_val > EYE_OPEN_THRESHOLD else (0, 0, 255)
                brow_color = (0, 255, 0) if brow_val > BROW_RAISED_THRESHOLD else (0, 0, 255)

                cv2.putText(frame, f"Eye: {eye_val} (Goal: {EYE_OPEN_THRESHOLD})", (380, 430), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
                cv2.putText(frame, f"Brow: {brow_val} (Goal: {BROW_RAISED_THRESHOLD})", (380, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, brow_color, 1)

            cv2.imshow('Meme Mirror AI', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()