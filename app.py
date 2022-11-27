import cv2, osascript
import numpy as np
import hand_detector as hd

hand_detector = hd.HandDetector(detection_con=0.7)
min_volume, max_volume = 10, 100
volume = 0
vol_bar, vol_per = 400, 0
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print('The project has been started')
while True:
    success, img = cap.read()
    img = hand_detector.find_hands(img)
    landmarks = hand_detector.find_position(img, draw=False)
    if len(landmarks) != 0:
        length = hand_detector.find_distance(4, 8, img)
        volume = np.interp(length, [50, 300], [min_volume, max_volume])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        vol_per = np.interp(length, [50, 300], [0, 100])
        print(f'The volume is {volume}')
        osascript.osascript(f'set volume output volume {volume}')
 
 
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
 
    cv2.imshow("Gesture Volume Control Project", img)
    cv2.waitKey(1)

print('The project has been finished')

