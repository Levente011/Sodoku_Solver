import pytesseract
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_digit(cell_image):
    pil_img = Image.fromarray(cell_image)
    text = pytesseract.image_to_string(pil_img, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    return ''.join(filter(str.isdigit, text))
