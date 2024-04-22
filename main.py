import cv2 #import the library 
import numpy as np

import matplotlib.pyplot as plt


import tensorflow as tf
from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array

stream_url = 'https://192.168.1.65:8080/video'
cap = cv2.VideoCapture(stream_url)

cap.set(3, 640)
cap.set(4, 480)

def getContours(img,original_img):
    
    contours,hierachy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
       
        if area > 60000:
            cv2.drawContours(original_img,cnt,-1,(0,255,0),2)
            #print(area)
            
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)
            
            width,height= 900,900

            pts1 = np.float32([[bx,by],[ax,ay],[cx,cy],[dx,dy]])
            pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

            matrix = cv2.getPerspectiveTransform(pts1,pts2)
            img_perspective = cv2.warpPerspective(original_img,matrix,(width,height))
            img_corners = cv2.cvtColor(img_perspective,cv2.COLOR_BGR2GRAY)
            
            # Turn into a binary image
            for x in range(0, 900):
                for y in range(0, 900):
                    if img_corners[x][y]<100:
                        img_corners[x][y]=0
                    else:
                        img_corners[x][y]=255
                        
            cv2.imshow('Corners',img_corners)
            
            return img_corners
            
            #classify(img_corners)

def classify(Img):
    crop_val = 10
    digits_list = []

    for i in range(9):
        for j in range(9):
            J = j*100 + crop_val
            I = i*100 + crop_val
            cell = Img[I:I+100 - 2*crop_val, J:J+100 - 2*crop_val]

            img_canny = cv2.Canny(cell, 50, 150)
            plt.imshow(img_canny, cmap='gray')
            plt.show()

            contours, hierachy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            digit = 0
            prob = 1.0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > 7.5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    image_rect = cell[y:y+h, x:x+w]
                    image_rect = cv2.resize(image_rect, (100, 100))

                    image_num = img_to_array(image_rect)
                    image_num = np.array(image_num).reshape(-1, 100, 100, 1)
                    image_num = image_num.astype('float32')
                    image_num = image_num / 255.0

                    model = tf.keras.models.load_model('model.h5')
                    prediction = model.predict(image_num)
                    digit = int(np.argmax(prediction))
                    prob = np.amax(prediction)

            print("Detected: ", digit)
            print("Probability: ", prob)
            digits_list.append(digit)

    return digits_list


def is_valid(grid, num, coordinate):
    # Check row
    for i in range(len(grid[0])):
        if grid[coordinate[0]][i] == num and coordinate[1] != i:
            return False

    # Check column
    for i in range(len(grid)):
        if grid[i][coordinate[1]] == num and coordinate[0] != i:
            return False

    # Check box
    box_x = coordinate[1] // 3
    box_y = coordinate[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if grid[i][j] == num and (i,j) != coordinate:
                return False

    return True

def solve(grid):
    find = find_empty(grid)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if is_valid(grid, i, (row, col)):
            grid[row][col] = i

            if solve(grid):
                return True

            grid[row][col] = 0
        

    return False



def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return (i, j)  # row, col

    return None

def save_sudoku(sudoku2d, sudoku2d_unsolved):
    
    solved_cell = np.ones((900, 900, 3))
    # Draw the sublines
    for i in range(8):
        solved_cell = cv2.line(solved_cell, ((i+1)*100, 0), ((i+1)*100, 900), (255, 255, 255), 5)
        solved_cell = cv2.line(solved_cell, (0,(i+1)*100), (900, (i+1)*100), (255, 255, 255), 5)
    # Draw the lines
    for i in range(2):
        solved_cell = cv2.line(solved_cell, ((i+1)*300, 0), ((i+1)*300, 900), (255, 255, 255), 10)
        solved_cell = cv2.line(solved_cell, (0,(i+1)*300), (900, (i+1)*300), (255, 255, 255), 10)




    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    thickness = 4
    for (index_row, row) in enumerate(sudoku2d):
        for (index_num, num) in enumerate(row):
            pos = (index_num*100 + 25, index_row*100 + 70)
            color = (200, 200, 200)
            if sudoku2d_unsolved[index_row][index_num] == 0:
                color = (0, 200, 0)
            
            solved_cell = cv2.putText(solved_cell ,str(num), pos, font, 
                           fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite('solved.png', solved_cell)
    cv2.imshow('Solved',solved_cell)

grid_found = False
grid_img = None

while True:
    success, img = cap.read()
    img_copy = img.copy()
    cv2.imshow('Sudoku Solver', img_copy)
    if not grid_found:
        # Turning the original image to canny.
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(5,5),3)
        imgCanny = cv2.Canny(imgBlur,50,50)
        img_contours_bin = getContours(imgCanny, img_copy)
        if img_contours_bin is not None:
            grid_found = True
            grid_img = img_contours_bin
            sudoku = classify(img_contours_bin)
            sudoku2d = [sudoku[i*9:(i+1)*9] for i in range(9)]
            sudoku2d = np.array(sudoku2d)
            sudoku2d_unsolved = sudoku2d.copy()
            print(sudoku2d_unsolved)
            solve(sudoku2d)
            save_sudoku(sudoku2d, sudoku2d_unsolved)
            break
    
    else:
        if grid_img is not None:
            try:
                # Display the ongoing solving process
                cv2.imshow('Sudoku Solver', grid_img)
            except Exception as e:
                print(f"Error during grid processing: {e}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    
    
cv2.destroyAllWindows()
