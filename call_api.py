import os
import requests
import cv2
import numpy as np

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# API_URL = "http://localhost:8000/ocr/"
API_URL = "http://localhost:8000/compute_quality/"

# IMAGE_PATH = "image_test/photo_6071287700461305339_y.jpg"
# IMAGE_PATH = "20241017_201143/input.png"

IMAGE_DIR = "/media/shawnle/e513b78c-bb84-4ba3-ac87-68d8a51869f1/Documents/2024/Apr/ipsa-models-whiteboard-extraction-dvc-ws/"
# IMAGE_PATH = os.path.join(IMAGE_DIR, "2024-11-28", "2024-11-27 00:54:52.541794506 +0000 UTC m=+2102080.942312283.jpg")
IMAGE_PATH = os.path.join(IMAGE_DIR, "2024-11-28", "2024-11-27 00:55:02.705649523 +0000 UTC m=+2102091.106167291.jpg")

image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Cannot open: {IMAGE_PATH}")
    exit()


with open(IMAGE_PATH, "rb") as image_file:
    response = requests.post(API_URL, files={"file": image_file}, data={"langs": "en", "gpu": 0})

if response.status_code == 200:
    pass_rate = response.json().get("pass_rate", 0)
    print(f"Pass rate: {pass_rate}")

    ocr_results = response.json().get("results", [])
    for result in ocr_results:
        bbox = result["bbox"]
        # text = result["text"]
        text = ''
        contr = result["contrast"]
        prob = result["confidence"]

        print("[INFO] bb {:.4f}: c {}".format(prob, contr))

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        text = cleanup_text(text)

        if 0 <= tl[0] < image.shape[1] and 0 <= tl[1] < image.shape[0]:
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
            cv2.putText(
                image, text, (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        else:
            print(f"Error box: {bbox}")

    cv2.imshow("OCR Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"API call failed with status code {response.status_code}")
    print("Response:", response.json())
