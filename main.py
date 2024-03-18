import cv2
from sudoku_extractor import preprocess_image, find_sudoku_grid, four_point_transform, order_points
from digit_recognizer import recognize_digit
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame = preprocess_image(frame)
        grid_contour = find_sudoku_grid(processed_frame)
        if grid_contour is not None and len(grid_contour) == 4:
                reshaped_contour = grid_contour.reshape(4, 2)
                warped_grid = four_point_transform(frame, reshaped_contour)
                cv2.imshow("Warped Sudoku Grid", warped_grid)

        cv2.imshow("Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
