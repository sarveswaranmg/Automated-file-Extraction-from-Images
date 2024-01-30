import easyocr
import cv2
import os
import subprocess, sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab
import re, sys
import regex as re
import camelot
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): 
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
     
        reader = easyocr.Reader(['en'], gpu=False)
        image = cv2.imread('C:/Users/chenc/OneDrive/Desktop/MLPROJ/saved_img.jpg')
        # result = reader.readtext('C:/Users/chenc/OneDrive/Desktop/MLPROJ/saved_img.jpg')
        # Total = []
        # for (bbox, text, prob) in result:
        #     Total.append(text)
        #     (tl, tr, br, bl) = bbox
        #     tl = (int(tl[0]), int(tl[1]))
        #     tr = (int(tr[0]), int(tr[1]))
        #     br = (int(br[0]), int(br[1]))
        #     bl = (int(bl[0]), int(bl[1]))
        #     cv2.rectangle(image, tl, br, (0, 300, 0), 1)
        #     cv2.putText(image, text, (tl[0], tl[1] - 2),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (300, 0, 0),1)
        # plt.rcParams['figure.figsize'] = (20,20)
        # plt.show()
        # reader = easyocr.Reader(['en'], gpu=False)
        image = cv2.imread('C:/Users/chenc/OneDrive/Desktop/MLPROJ/saved_img.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        filtered = cv2.adaptiveThreshold(thresh.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        or_image = cv2.bitwise_or(thresh, closing)
        easy_ocr_result=reader.readtext(thresh,detail=0)
        string= easy_ocr_result
        for item in string:
            stringocr = ''.join([str(elem) for elem in item])
        # regex = '(\d\d\w\w\w\d\d\d\d)'
        patterns = [r"[0-9][0-9][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]", r"[0-9][0-9][0-9][0-9][0-9][0-9]",r"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]",r"[0-9][A-Z][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]"]
        pattern = "|".join(patterns)
        matches=re.findall(pattern, stringocr)
        # matches = re.findall("[0-9][0-9][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9][0-9][0-9]", stringocr)
        print1=''.join([str(elem) for elem in matches])
        print("Output from EasyOCR:",string)
        print(matches)
        print("The required characters:",print1)
        rootDir ="C:\\Users"
        fileToSearch=print1 + '.pdf'
        for relPath, dirs, files in os.walk(rootDir):
            if(fileToSearch in files):
                fullPath = os.path.join(rootDir,'/',relPath, fileToSearch)
                print("File found in:",fullPath)
                subprocess.Popen([fullPath], shell=True)
        break
    elif key == ord('q'):
        cap.release()