import cv2
from sudoku_extractor import preprocess_image, find_sudoku_grid, four_point_transform
from digit_recognizer import recognize_digit
import numpy as np

def extract_cells(warped_grid):
    cells = []
    grid_size = warped_grid.shape[0]  # Feltételezzük, hogy a rács négyzet alakú
    cell_size = grid_size // 9

    for y in range(9):
        for x in range(9):
            start_x = x * cell_size
            start_y = y * cell_size
            cell = warped_grid[start_y:start_y + cell_size, start_x:start_x + cell_size]
            cells.append(cell)
    return cells


def print_sudoku(matrix):
    for row in matrix:
        print(" ".join(str(num) if num != 0 else '.' for num in row))

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame = preprocess_image(frame)
        grid_contour = find_sudoku_grid(processed_frame)
        if grid_contour is not None:
            warped_grid = four_point_transform(frame, grid_contour)
            cells = extract_cells(warped_grid)
            sudoku_matrix = []
            for i in range(0, len(cells), 9):  # Minden sorra iterál
                row = [recognize_digit(cells[j]) for j in range(i, i + 9)]  # Egy sor celláinak feldolgozása
                sudoku_matrix.append(row)
            print_sudoku(sudoku_matrix)
            break

        cv2.imshow("Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
