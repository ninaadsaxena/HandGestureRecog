import cv2
import time
import mediapipe as mp
import numpy as np

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def draw_landmarks(self, img, hand_landmarks):
        if hand_landmarks:
            for hand_lms in hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS
                )

    def recognize_gesture(self, hand_landmarks, handedness):
        # Get relevant points (tips and bases of fingers)
        thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
        thumb_base = np.array([hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y])

        tips = [hand_landmarks.landmark[8], hand_landmarks.landmark[12], hand_landmarks.landmark[16], hand_landmarks.landmark[20]]
        bases = [hand_landmarks.landmark[6], hand_landmarks.landmark[10], hand_landmarks.landmark[14], hand_landmarks.landmark[18]]

        # Thumb up detection (works for both hands)
        is_thumb_up = thumb_tip[1] < thumb_base[1] if handedness == "Left" else thumb_tip[1] > thumb_base[1]
        fingers_bent = all(tip.y > base.y for tip, base in zip(tips, bases))

        # Other gestures
        all_fingers_extended = all(tip.y < base.y for tip, base in zip(tips, bases))
        peace_sign = tips[0].y < bases[0].y and tips[1].y < bases[1].y and all(tip.y > base.y for tip, base in zip(tips[2:], bases[2:]))
        rock_on_sign = tips[0].y < bases[0].y and tips[3].y < bases[3].y and all(tip.y > base.y for tip, base in zip(tips[1:3], bases[1:3]))

        if is_thumb_up and fingers_bent:
            return "Thumbs Up"
        elif all_fingers_extended:
            return "Open Palm"
        elif peace_sign:
            return "Peace"
        elif rock_on_sign:
            return "Rock On"
        return "Unknown"

    def recognize_hands(self, result):
        gesture_info = []
        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = result.multi_handedness[idx].classification[0].label
                gesture_name = self.recognize_gesture(hand_landmarks, handedness)
                gesture_info.append({
                    'handedness': handedness,
                    'gesture': gesture_name
                })
        return gesture_info


def draw_stylish_text(img, text, position, font_scale=1.0, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    bg_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text
    padding = 10

    # Draw background rectangle
    cv2.rectangle(img, (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding),
                  bg_color, -1)

    # Draw text over the rectangle
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandGestureRecognition()
    pTime = 0

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to capture video frame.")
            break

        # Flip the frame horizontally to correct the handedness issue
        frame = cv2.flip(frame, 1)

        results = detector.process_frame(frame)
        detector.draw_landmarks(frame, results.multi_hand_landmarks)

        gesture_info = detector.recognize_hands(results)

        # Draw FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        draw_stylish_text(frame, f'FPS: {int(fps)}', (10, 50), 1, 2)

        # Display hand gesture info below FPS
        if gesture_info:
            y_offset = 100
            for hand in gesture_info:
                text = f'{hand["handedness"]} Hand: {hand["gesture"]}'
                draw_stylish_text(frame, text, (10, y_offset), 1, 2)
                y_offset += 50  # Move the next hand info below the current one

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
