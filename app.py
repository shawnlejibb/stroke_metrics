from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from easyocr import Reader

import cv2
import numpy as np

from main import get_contrast_metric

app = FastAPI()

@app.post("/compute_quality")
async def compute_quality(
    file: UploadFile = File(...),
    langs: str = "en",
    contrast_thresh: int = 10,
    gpu: int = -1,
):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return JSONResponse({"error": f"Cannot process image: {str(e)}"}, status_code=400)

    langs_list = langs.split(",")
    print(f"[INFO] OCR'ing with the following languages: {langs_list}")
    try:
        reader = Reader(langs_list, gpu=gpu > 0)
        results = reader.readtext(image)
    except Exception as e:
        return JSONResponse({"error": f"OCR failed: {str(e)}"}, status_code=500)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Laplacian(blur, cv2.CV_64F)    

    ocr_results = []
    cnt_pass = 0
    for cnt, (bbox, text, prob) in enumerate(results):
        res = {
        #     "text": text,
            "bbox": [[int(coord[0]), int(coord[1])] for coord in bbox],
            "confidence": float(prob)
        }

        bbox = [[int(coord[0]), int(coord[1])] for coord in bbox],
        
        print('bbox:', bbox)
        (tl, tr, br, bl) = bbox[0]
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))

        c = get_contrast_metric(
            edge, 
            tl,
            br,
            save_graph=True,
            save_graph_fn=f'contrast_{cnt}.png',
            args=None,
            rgb=None
        )

        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, 
                    "{:0.1f}".format(float(c[0])),
                    tl,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)
        cv2.imwrite(f'output_{cnt}.png', image)
        
        is_good = 1
        if c > contrast_thresh:
            cnt_pass += 1
        else:
            is_good = 0
        
        res["contrast"] = float(c[0])
        res["pass"] = is_good
        ocr_results.append(res)
    pc_pass = cnt_pass / len(results) * 100

    print('ocr_results:', ocr_results)
    print('pass_rate:', pc_pass)

    return {
        "results": ocr_results,
        "pass_rate": pc_pass
    }
