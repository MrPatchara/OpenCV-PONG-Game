import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]
stage = 1

# Create a fullscreen window
cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def update_stage(score):
    global stage, speedX, speedY, ballPos
    total_score = score[0] + score[1]
    new_stage = total_score // 5 + 1
    if new_stage != stage:
        stage = new_stage
        speedX = 15 + (stage - 1) * 5
        speedY = 15 + (stage - 1) * 5
        ballPos = [640, 360]  # Reset ball position to center
        return True
    return False

def countdown():
    for i in range(3, 0, -1):
        img = imgBackground.copy()
        cv2.putText(img, f"Stage {stage}", (img.shape[1] // 2 - 150, img.shape[0] // 2 - 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(img, f"Starting in {i}", (img.shape[1] // 2 - 200, img.shape[0] // 2 + 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[0] + score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (0, 0, 255), 5)  # Change color to red (0, 0, 255)

    # If game not over move the ball
    else:
        if update_stage(score):
            countdown()

        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(img, f"Stage {stage}", (540, 700), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        stage = 1
        imgGameOver = cv2.imread("Resources/gameOver.png")
    elif key == 27:  # Esc key to quit
        break

cap.release()
cv2.destroyAllWindows()