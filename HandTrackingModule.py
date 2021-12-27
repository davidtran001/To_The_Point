import cv2
import mediapipe as mp
import time
from random import randrange

def centerText(w, h, text, font, size, thickness):
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, size, thickness)[0]
    # get coords based on boundary
    textX = (w - textsize[0]) // 2
    textY = (h + textsize[1]) // 2
    return textX, textY


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changes frame from BGR to RGB

        self.results = self.hands.process(imgRGB) # detects hand in frame 

        # Draw points and connections in detected hand
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for i, lm in enumerate(handLms.landmark): # print the location of the landmarks in each frame

                    # # convert the landmark position to correspond to pixels on frame
                    # h, w, c = frame.shape
                    # cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(i, 'x: ', cx, 'y: ', cy)
                    
                    # if i == 0: # draw on a specific landmark on hand
                    #     cv2.circle(frame, (cx,cy), 25, (255,0,255), cv2.FILLED)
                    if draw:
                        self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNum=0, draw=True, drawPointNum=8):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for i, lm in enumerate(myHand.landmark): # print the location of the landmarks in each frame

                        # convert the landmark position to correspond to pixels on frame
                        h, w, c = frame.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(i, 'x: ', cx, 'y: ', cy)
                        lmList.append([i, cx, cy])
                        if draw and i == drawPointNum:
                            cv2.circle(frame, (cx,cy), 10, (255,0,255), cv2.FILLED)
                        
        return lmList

def main():
    pTime = 0

    # define code for each finger tip
    finger_thumb = 4
    finger_index = 8
    finger_middle = 12
    finger_ring = 16
    finger_pinky = 20
    finger_dict = {finger_thumb: 'Thumb', finger_index: 'Index', finger_middle: 'Middle', finger_ring: 'Ring', finger_pinky: 'Pinky'}

    # initialize window size
    screen_width = 800
    screen_height = 600

    cap = cv2.VideoCapture(-0, cv2.CAP_DSHOW)  # selects video capture source 

    detector = handDetector() # create handDetector object

    points = 0
    found_obj = True

    start_time = time.time()

    start_screen = True

    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame=frame, draw=True)
        lmList = detector.findPosition(frame=frame, handNum=0, draw=False, drawPointNum=8)

        # display framerate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (1,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 2)

        cv2.imshow('img1', frame)  # display the frame        

        # kill webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            cv2.imwrite('capture.png', frame)
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == '__main__':
    main()