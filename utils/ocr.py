from paddleocr import PaddleOCR

def ocr():
    ocr = PaddleOCR(lang='en')
    return ocr