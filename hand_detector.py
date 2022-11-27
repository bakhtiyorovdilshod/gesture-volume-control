import cv2, math
import mediapipe as mp


class HandDetector:
    """
        Hand Tracking Class
    """
    def __init__(self, mode=False, model_complexity=1, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.model_complex = model_complexity
 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complex,
                                        self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
 
    def find_hands(self, img, draw=True):
        """
            Finding hands from capture
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks
                    (
                        img, 
                        hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img
 
    def find_position(self, img, hand_number=0, draw=True):
        """
            finding ladkmarks point's positions in hand
        """
        self.landmarks = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
 
        return self.landmarks
    
    def find_distance(self, p1, p2, img, draw=True):
        """
            Finding two fingers's distance from each other
        """
        x1, y1 = self.landmarks[p1][1], self.landmarks[p1][2]
        x2, y2 = self.landmarks[p2][1], self.landmarks[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length
