import cv2
import numpy as np
from tensorflow.keras.models import load_model

from sudoku_extractor import preprocess_image, find_sudoku_grid, four_point_transform
from digit_recognizer import recognize_digit

# def getContours(img, original_img):
#     contours,hierachy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 100000:
#             print(area)
#             cv2.drawContours(original_img,cnt,-1,(0,255,0),3)

def getContours(img, original_img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 90000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:  # Check for the biggest rectangular area
                biggest = approx
                max_area = area
    if biggest is not None:
        cv2.drawContours(original_img, [biggest], -1, (0, 255, 0), 3)
        cv2.putText(original_img, 'Sudoku Grid', (biggest.ravel()[0], biggest.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw circles on each corner
        for point in biggest:
            x, y = point.ravel()
            cv2.circle(original_img, (x, y), 10, (255, 0, 0), -1)
    return biggest  # Returns the corners of the largest rectangle (if found)

def order_points(pts):
    # Initial sort based on the y-coordinates
    sorted_pts = pts[np.argsort(pts[:, 1])]

    # Grab the top and bottom points
    top = sorted_pts[:2]
    bottom = sorted_pts[2:]

    # Now sort the top and bottom points based on their x-coordinates
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    # Return the points in top-left, top-right, bottom-right, bottom-left order
    return np.array([top[0], top[1], bottom[1], bottom[0]])

def four_point_transform(image, pts):
    # Sort the points into the correct order
    rect = order_points(np.array(pts, dtype="float32"))

    # Calculate the width and height of the output grid
    width_A = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width_B = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height_B = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    # Define the destination points for the transformation
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Perform the perspective transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (max_width, max_height))
    return warp

def extract_cells(warped_grid):
    cells = []
    cell_size = warped_grid.shape[0] // 9  # Assume the grid is 300x300
    for y in range(9):
        row = []
        for x in range(9):
            start_x = x * cell_size
            start_y = y * cell_size
            cell = warped_grid[start_y:start_y + cell_size, start_x:start_x + cell_size]
            row.append(cell)
        cells.append(row)
    return cells

def recognize_sudoku_grid(cells, model):
    sudoku_grid = []
    for row in cells:
        sudoku_row = []
        for cell in row:
            # Preprocess the cell image as required by your model
            # Ensure the cell image is resized, normalized, and reshaped as expected by the model
            result, probability = prediction(cell, model)
            sudoku_row.append(result)
        sudoku_grid.append(sudoku_row)
    return sudoku_grid

def print_sudoku(matrix):
    for row in matrix:
        print(" ".join(str(num) if num != 0 else '.' for num in row))



# def extract_cells(warped_grid):
#     cells = []
#     grid_size = warped_grid.shape[0]  # Feltételezzük, hogy a rács négyzet alakú
#     cell_size = grid_size // 9

#     for y in range(9):
#         for x in range(9):
#             start_x = x * cell_size
#             start_y = y * cell_size
#             cell = warped_grid[start_y:start_y + cell_size, start_x:start_x + cell_size]
#             cells.append(cell)
#     return cells


# def print_sudoku(matrix):
#     for row in matrix:
#         print(" ".join(str(num) if num != 0 else '.' for num in row))


# def detect_grid(image):
#     ret, thresh = cv2.threshold(image, 50, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print("Number of contours detected:", len(contours))

#     max_area = 0
#     largest_rectangle = None
#     text_position = (0, 0)

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         area = w * h
#         approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
#         if len(approx) == 4 and area > max_area:
#             max_area = area
#             largest_rectangle = cnt
#             text_position = (x, y)

#     if largest_rectangle is not None:
#         x, y, w, h = cv2.boundingRect(largest_rectangle)
#         ratio = float(w) / h
#         if ratio >= 0.9 and ratio <= 1.1:
#             image = cv2.drawContours(image, [largest_rectangle], -1, (255, 0, 0), 3)
#             cv2.putText(image, 'Square', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#         else:
#             image = cv2.drawContours(image, [largest_rectangle], -1, (255, 0, 0), 3)
#             cv2.putText(image, 'Rectangle', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Contours", image)
#     # return image  # Optionally return the modified image if needed elsewhere

model = load_model('model/digits.h5')

def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    prob = np.amax(prediction)
    result = np.argmax(prediction)
    cv2.imshow("Aktuális", image)
    print("prob:", prob)
    print("res:", result)
    cv2.waitKey(3000)
    if prob < 0.75:
        result = 0
    return result, prob


def main():
    stream_url = 'https://192.168.1.65:8080/video'
    cap = cv2.VideoCapture(stream_url)

    # WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    
    if not cap.isOpened():
        print("Error: Unable to open camera stream.")
        return

    while True:
    
        success, frame = cap.read()
        frame_copy = frame.copy()

        # box_size = (320, 320)
        # box = [(int(WIDTH // 2 - box_size[0] // 2), int(HEIGHT // 2 - box_size[1] // 2)), 
        #               (int(WIDTH // 2 + box_size[0] // 2), int(HEIGHT // 2 + box_size[1] // 2))]
        
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5,5), 3)
        img_canny = cv2.Canny(img_blur,50,50)

        corners = getContours(img_canny, frame_copy)

        if corners is not None:
            corners_of_grid = [tuple(point.ravel()) for point in corners]
            corners_of_grid = order_points(np.array(corners_of_grid))
            print("Corners of the grid:", corners_of_grid)
            warped_grid = four_point_transform(img_gray, corners_of_grid)
            cells = extract_cells(warped_grid)

            sudoku_matrix = recognize_sudoku_grid(cells, model)
            print_sudoku(sudoku_matrix)
            break


        # corners_of_grid = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # Example: Show the top-left cell

        # img_cropped = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        # img_gray = cv2.resize(img_gray, (400, 400))
        # cv2.imshow("Cropped", img_gray)
        # detect_grid(img_gray)

        # result, probability = prediction(img_gray, model)

        # cv2.putText(frame_copy, f"Prediction: {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #             (255, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame_copy, "Probability: " + "{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #              (255, 0, 255), 2, cv2.LINE_AA)
        
        # cv2.rectangle(frame_copy, box[0], box[1], (0, 255, 0), 3)

        cv2.imshow("Sudoku Solver", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


        # processed_frame = preprocess_image(frame)
        # grid_contour = find_sudoku_grid(processed_frame)
        # if grid_contour is not None:
        #     warped_grid = four_point_transform(frame, grid_contour)
        #     cells = extract_cells(warped_grid)
        #     sudoku_matrix = []
        #     for i in range(0, len(cells), 9):  # Minden sorra iterál
        #         row = [recognize_digit(cells[j]) for j in range(i, i + 9)]  # Egy sor celláinak feldolgozása
        #         sudoku_matrix.append(row)
        #     print_sudoku(sudoku_matrix)
        #     break



if __name__ == '__main__':
    main()
