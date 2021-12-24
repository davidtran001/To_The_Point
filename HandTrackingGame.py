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
                        if draw and i == 8:
                            cv2.circle(frame, (cx,cy), 25, (255,0,255), cv2.FILLED)
                        
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
        screen_width = frame.shape[1]
        screen_height = frame.shape[0]

        # display framerate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        timer = int(20-(cTime-start_time))
        # cv2.putText(frame, str(int(fps)), (1,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3) 

        if start_screen: 
            start_text = 'To The Point'
            start_textX, start_textY = centerText(screen_width, screen_height, text=start_text, font=cv2.FONT_HERSHEY_PLAIN, size=5, thickness=2)
            cv2.putText(frame, start_text, (start_textX, start_textY), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2) 
            instructions = 'Show your hand to begin!'
            instructions_textX, instructions_textY = centerText(screen_width, screen_height, text=instructions, font=cv2.FONT_HERSHEY_PLAIN, size=3, thickness=2)
            cv2.putText(frame, instructions, (instructions_textX+100, instructions_textY+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) 

            if len(lmList) > 0:
                start_screen = False
                start_time = time.time()

        elif timer > 0:
            # generate random objective coordinates
            if found_obj:
                finger_obj = randrange(1,6)*4
                x_obj = randrange(150, screen_width-150)
                y_obj = randrange(150, screen_height-150)
                found_obj = False

            cv2.circle(frame, (x_obj, y_obj), 25, (0,0,255), cv2.FILLED) # display objective onto screen

            if len(lmList) > 0:
                # print(lmList[finger_obj]) # print location of tip of index finger (8)
                # (0,0)
                #     ...
                #       (w,h)

                # check if index finger is pointing to objective
                if x_obj-25 <= lmList[finger_obj][1] <= x_obj+25 and y_obj-25 <= lmList[finger_obj][2] <= y_obj+25:
                    found_obj = True
                    points += 1
                    
            cv2.putText(frame, str(timer), (1,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3) 
            cv2.putText(frame, 'points: ' + str(points), (screen_width-225,screen_height-5), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            cv2.putText(frame, finger_dict[finger_obj], (1,screen_height-8), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        else: # GAMEOVER screen
            gameover_text = 'GAMEOVER!'
            gameover_textX, gameover_textY = centerText(screen_width, screen_height, text=gameover_text, font=cv2.FONT_HERSHEY_PLAIN, size=5, thickness=2)
            total_points_text = 'points: ' + str(points)
            total_points_textX, total_points_textY = centerText(screen_width+20, screen_height+120, text=gameover_text, font=cv2.FONT_HERSHEY_PLAIN, size=5, thickness=2)
            cv2.putText(frame, gameover_text, (gameover_textX, gameover_textY), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2) # add text centered on image
            cv2.putText(frame, total_points_text, (total_points_textX, total_points_textY), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            # display restart button
            restart_x = screen_width-150
            restart_y = screen_height-100
            cv2.circle(frame, (restart_x, restart_y), 25, (0,255,0), cv2.FILLED) # display objective onto screen
            cv2.putText(frame, 'Index finger to restart -->', (screen_width-425, screen_height-100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if len(lmList) > 0 and restart_x-25 <= lmList[finger_index][1] <= restart_x+25 and restart_y-25 <= lmList[finger_index][2] <= restart_y+25:
                start_time = time.time()
                points = 0

        cv2.imshow('img1', frame)  # display the frame        

        # kill webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            cv2.imwrite('capture.png', frame)
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == '__main__':
    main()