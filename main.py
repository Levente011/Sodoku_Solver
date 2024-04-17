import cv2
import numpy as np

stream_url = 'https://192.168.1.65:8080/video'
cap = cv2.VideoCapture(stream_url)

cap.set(3, 640)
cap.set(4, 480)

def getContours(img, original_img):
    contours,hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 90000:
            cv2.drawContours(original_img, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width,height = 900,900

            pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_perspective = cv2.warpPerspective(original_img, matrix, (width, height))
            img_corners = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)

            for x in range(0, 900):
                for y in range(0, 900):
                    if img_corners[x][y] < 100:
                        img_corners[x][y] = 0
                    else:
                        img_corners[x][y] = 255

            # return img_contours
            cv2.imshow('Corners', img_contours)

while True:
    success, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 3)
    img_canny = cv2.Canny(img_blur, 50, 50)
    img_copy = img.copy()

    getContours(img_canny, img_copy)

    cv2.imshow('Sodoku Solver', img_copy)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break