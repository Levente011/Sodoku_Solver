import cv2
import numpy as np
from tensorflow.keras.models import load_model

from sudoku_extractor import preprocess_image, find_sudoku_grid, four_point_transform
from digit_recognizer import recognize_digit

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


def main():
    stream_url = 'https://192.168.1.65:8080/video'
    cap = cv2.VideoCapture(stream_url)

    WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    model = load_model('model/digits.h5')

    def prediction(image, model):
        img = cv2.resize(image, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        prob = np.amax(prediction)
        result = np.argmax(prediction)
        if prob < 0.75:
            result = 0
        return result, prob
    
    if not cap.isOpened():
        print("Error: Unable to open camera stream.")
        return

    while True:
    
        _, frame = cap.read()
        frame_copy = frame.copy()

        box_size = (320, 320)
        box = [(int(WIDTH // 2 - box_size[0] // 2), int(HEIGHT // 2 - box_size[1] // 2)), 
                      (int(WIDTH // 2 + box_size[0] // 2), int(HEIGHT // 2 + box_size[1] // 2))]
        
        img_cropped = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (400, 400))
        cv2.imshow("Cropped", img_gray)

        result, probability = prediction(img_gray, model)

        cv2.putText(frame_copy, f"Prediction: {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, "Probability: " + "{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                     (255, 0, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(frame_copy, box[0], box[1], (0, 255, 0), 3)

        cv2.imshow("Sudoku Solver", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
