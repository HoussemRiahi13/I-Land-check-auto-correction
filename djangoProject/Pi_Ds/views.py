import json

import torch
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image, ExifTags
from django.http import JsonResponse
import tempfile
import numpy as np
from scipy.ndimage import interpolation as inter
from matplotlib import pyplot as plt
import argparse
import cv2
import imutils
from PIL import Image
import requests
from ArabicOcr import arabicocr
import requests
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import tempfile
import numpy as np
from num2words import num2words
import PIL
from PIL import Image
from textblob_ar import TextBlob
from textblob_ar import TextBlob
from textblob_ar.correction import TextCorrection
import numpy as nps




def say_hello(request):
    dict={"IsSelected": False, "IsValid": False,"IsCorrected":False,"IsContour":False,"Handwritting":"","Chiffre":"","Message":""}
    dict=model(request.GET['nom'],request.GET['type'])
    print(dict)

    return JsonResponse(dict)



def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src


import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

model_path = r'Check.tflite'

# Load the labels into a list
"""
classes = ['???'] * model.model_spec.config.num_classes
label_map = model.model_spec.config.label_map
for label_id, label_name in label_map.as_dict().items():
  classes[label_id-1] = label_name
  print(label_id)
  print(label_name)
"""

classes = ['check']
# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Retur the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all outputs from the model
    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    count = int(get_output_tensor(interpreter, 2))
    classes = get_output_tensor(interpreter, 3)

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    print(results)
    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    ymincrop, xmincrop, ymaxcrop, xmaxcrop = 0, 0, 0, 0
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])
        print(f'{ymin} ,{xmin}, {ymax}, {xmax}')
        # Find the class index of the current object
        class_id = int(obj['class_id'])

        if (class_id == 0):
            ymincrop, xmincrop, ymaxcrop, xmaxcrop = ymin, xmin, ymax, xmax
            if (ymincrop < 0):
                ymincrop = 0
            if (xmincrop < 0):
                xmincrop = 0
            if (ymaxcrop < 0):
                ymaxcrop = 0
            if (xmaxcrop < 0):
                xmaxcrop = 0
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    crop_img = original_uint8[int(ymincrop - (ymincrop * 0.4)):int((ymincrop + (ymaxcrop - ymincrop)) * 1.4),
               int(xmincrop - (xmincrop * 0.4)):int((xmincrop + (xmaxcrop - xmincrop)) * 1.4)]
    return original_uint8, crop_img, results, ymincrop, xmincrop, ymaxcrop, xmaxcrop

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import nltk
from nltk.tokenize import word_tokenize
import string
from spellchecker import SpellChecker
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import tempfile
import cv2
import numpy as np

processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-large-handwritten')  # able tranforming the image to tensor
model = torch.load('trocr-large-handwritten.pt')  # able to decode the tensor to a text
def model(name,type):
    ymin, xmin, ymax, xmax = 0, 0, 0, 0

    INPUT_IMAGE_URL = rf"C:\xampp\htdocs\uploadimage\upload\{name}.jpg"
    DETECTION_THRESHOLD = 0.5

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image, img, results, ymin, xmin, ymax, xmax = run_odt_and_draw_results(
        INPUT_IMAGE_URL,
        interpreter,
        threshold=DETECTION_THRESHOLD
    )
    if (results[0]['score']<0.94):
        dic={"IsSelected":False,"IsValid":False,"IsCorrected":False,"IsContour":False,"Handwritting":"","Chiffre":"","Message":"No Check detected Please Try Agian"}
        return dic
    else:
        if (len(results) != 1 or xmax - xmin < ymax - ymin):
            im = PIL.Image.open(INPUT_IMAGE_URL)
            img41 = im.rotate(90, expand=True)
            img41.save(f"{name}-1.jpg")
            path = f"{name}-1.jpg"
            detection_result_image, img, results, ymin, xmin, ymax, xmax = run_odt_and_draw_results(
                path,
                interpreter,
                threshold=DETECTION_THRESHOLD)
            if (len(results) != 1 or xmax - xmin < ymax - ymin):
                im = PIL.Image.open(INPUT_IMAGE_URL)
                img42 = im.rotate(180, expand=True)
                img42.save(f"{name}-2.jpg")
                path1 = f"{name}-2.jpg"
                detection_result_image, img, results, ymin, xmin, ymax, xmax = run_odt_and_draw_results(
                    path1,
                    interpreter,
                    threshold=DETECTION_THRESHOLD)
                if (len(results) != 1 or xmax - xmin < ymax - ymin):
                    im = PIL.Image.open(INPUT_IMAGE_URL)
                    img43 = im.rotate(270, expand=True)
                    img43.save(f"{name}-3.jpg")
                    path2 = f"{name}-3.jpg"
                    detection_result_image, img, results, ymin, xmin, ymax, xmax = run_odt_and_draw_results(
                        path2,
                        interpreter,
                        threshold=DETECTION_THRESHOLD)
                    if (len(results) != 1):
                        dic = {"IsSelected": False, "IsValid": False, "IsCorrected": False,"IsContour":False, "Handwritting": "",
                               "Chiffre": "","Message":"No Check detected Please Try Agian"}
                        return dic
        widthImg, heightImg = 0, 0
        # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
        # imgBlur = cv2.GaussianBlur(img, (5, 5), 1) # ADD GAUSSIAN BLUR
        # GET TRACK BAR VALUES FOR THRESHOLDS
        imgThreshold = cv2.Canny(img, 60, 250)  # APPLY CANNY BLUR
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

        ## FIND ALL COUNTOURS
        imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
        if biggest.size==0:
            dic = {"IsSelected": True, "IsValid": False, "IsCorrected": False,"IsContour":False, "Handwritting": "",
                   "Chiffre": "", "Message": "Please Verify That your background - or check position"}
            return dic
        if biggest.size != 0:
            print(biggest)
            A, B, C, D = biggest[0][0], biggest[1][0], biggest[2][0], biggest[3][0]
            List = []
            List.append(int(np.linalg.norm(A - B)))
            List.append(int(np.linalg.norm(A - C)))
            List.append(int(np.linalg.norm(A - D)))
            List.sort()
            heightImg = List[0]
            widthImg = List[1]
            biggest = reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            imgBigContour = drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            rgb = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2RGB)
            Hei,Wid=rgb.shape[0],rgb.shape[1]
            rgbcrop = rgb[int(Hei * 0.03):int(Hei * 0.32), int(Wid * 0.09):int(Wid * 0.34)]
            cv2.imwrite('1pop.jpg',rgbcrop)
        image_path = '1pop.jpg'
        out_image = 'out5.jpg'
        results = arabicocr.arabic_ocr(image_path, out_image)
        rotated = imgWarpColored
        print(results)
        if(len(results)==0):
           rotated = imutils.rotate_bound(imgWarpColored, angle=180)

        """results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        # display the orientation information
        print("[INFO] detected orientation: {}".format(
            results["orientation"]))
        print("[INFO] rotate by {} degrees to correct".format(
            results["rotate"]))
        print("[INFO] detected script: {}".format(results["script"]))"""
        # rotate the image to correct the orientation
        # show the original image and output image after orientation
        # correction
        Height = rotated.shape[0]
        Width = rotated.shape[1]
        top, bottom, left, right = int(Height * 0.066), int(Height * 0.33), int(Width * 0.79), int(Width * 1)
        im1 = rotated[top:bottom, left:right]
        top, bottom, left, right = int(Height * 0.33), int(Height * 0.54), int(Width * 0), int(Width * 1)
        im2 = rotated[top:bottom, left:right]
        h, w = im2.shape[0], im2.shape[1]
        # Create a Rectangle patch
        cv2.rectangle(im2, pt1=(0, 0), pt2=(int(0.32 * w), int(0.39 * h)), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(im2, pt1=(int(0.71 * w), 0), pt2=(w, int(0.39 * h)), color=(255, 255, 255), thickness=-1)
        cv2.imwrite("1.jpg", im1)
        cv2.imwrite("2.jpg", im2)

        if type == "fr" :
            image_path15 = '1.jpg'

            num15 = ICR(image_path15)
            numt=num2words(num15,lang="fr")
            dic = {"IsSelected": True, "IsValid": True, "IsCorrected": False, "Handwritting": numt,
                               "Chiffre": num15, "Message": ""}
            return dic
        if type == "eng" :
            image_path1 = '1.jpg'

            num13 = ICR(image_path1)

            image_path2 = '2.jpg'
            gen_text = ICR(image_path2)
            dic = {"IsSelected": True, "IsValid": True, "IsCorrected": False, "Handwritting": gen_text,
                               "Chiffre": num13, "Message": ""}
            return dic
        else:

            im1 = cv2.imread('1.jpg')

            chiff=40000
            image_path = '2.jpg'
            im = cv2.imread(image_path)
            im1 = thick_font(im)
            filename = '6thick.jpg'

            # Using cv2.imwrite() method
            # Saving the image
            cv2.imwrite(filename, im1)
            image_path = '6thick.jpg'
            out_image = 'out.jpg'
            results12 = arabicocr.arabic_ocr(image_path, out_image)
            words = []
            for i in range(len(results12)):
                word = results12[i][1]
                words.append(word)
            text = TriList(results12)
            blobs = []
            for i in range(len(text)):
                blob = TextBlob(text[i])
                for j in range(len(blob.tokens)):
                    blobs.append(blob.tokens[j])
            f_nbr = []
            for i in range(len(blobs)):
                if (len(blobs[i]) > 2):
                    print(TextCorrection().correction(blobs[i], top=False))
                    f_nbr.append(TextCorrection().correction(blobs[i], top=True))
            sent = ""
            for i in range(len(f_nbr)):
                sent = sent + f_nbr[i] + " "
            sent = sent.rstrip(sent[-1])
            if (sent == chiff):
                dic = {"IsSelected": True, "IsValid": False, "IsCorrected": False, "Handwritting": sent,
                       "Chiffre": chiff, "Message": ""}
                return dic
            else:
                dic = {"IsSelected": True, "IsValid": True, "IsCorrected": True, "Handwritting": sent,
                       "Chiffre": chiff, "Message": ""}
                return dic

def set_image_dpi(image):
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = image.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def ICR(image_path):
  image = Image.open(image_path).convert("RGB")
  pixel_values = processor(images=image, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return generated_text

def load_image(image_path):
  return Image.open(image_path).convert("RGB")



def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((3,3),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_fontLite(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def ChiffreContruct(tab):
    sentence=[]
    sentence.append(tab[0]+" "+tab[2])
    sentence.append(tab[1]+" "+tab[2])
    return sentence

def TriList(tab):
  var=[]
  for i in range(1,len(tab)):
    k = tab[i]
    j = i-1
    while j >= 0 and k[0][0][0]>tab[j][0][0][0] :
      tab[j + 1] = tab[j]
      j -= 1
      tab[j + 1] = k
  for i in range(len(tab)):
    var.append(tab[i][1])
  return var

