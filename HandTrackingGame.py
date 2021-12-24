import cv2
import mediapipe as mp
import time
from random import randrange

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
                        if draw and i == 8:
                            cv2.circle(frame, (cx,cy), 25, (255,0,255), cv2.FILLED)
                        
        return lmList

def main():
    pTime = 0

    # initialize window size
    screen_width = 800
    screen_height = 600

    cap = cv2.VideoCapture(-0, cv2.CAP_DSHOW)  # Selects video capture source 
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    detector = handDetector()

    points = 0
    found_obj = True

    while True:
        success, frame = cap.read()
        
        # generate random objective coordinates
        if found_obj:
            x_obj = randrange(150, screen_width-150)
            y_obj = randrange(150, screen_height-150)
            found_obj = False

        frame = detector.findHands(frame=frame, draw=False)
        lmList = detector.findPosition(frame=frame, handNum=0, draw=True, drawPointNum=8)

        cv2.circle(frame, (x_obj, y_obj), 25, (0,0,255), cv2.FILLED)


        if len(lmList) > 0:
            print(lmList[8]) # print location of tip of index finger (8)
            # (0,0)
            #     ...
            #       (w,h)

            # check if index finger is pointing to objective
            if x_obj-20 <= lmList[8][1] <= x_obj+20 and y_obj-20 <= lmList[8][2] <= y_obj+20:
                found_obj = True
                points += 1

        # display framerate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        # cv2.putText(frame, str(int(fps)), (1,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)
        cv2.putText(frame, 'points: ' + str(points), (1,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)

        cv2.imshow('img1', frame)  # display the frame

        # kill webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            cv2.imwrite('capture.png', frame)
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == '__main__':
    main()