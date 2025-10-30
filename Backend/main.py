from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
import shutil
import tempfile
import os
import logging

# ✅ Import your segmenter and OCR modules
from segmenter import SindhiTextSegmenter
from test import preprocess_image, load_model, decode_prediction

# -------------------------------------------------------------------

app = FastAPI(title="Sindhi OCR API", description="Segments and recognizes Sindhi text from images")

# Load the model once when the API starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/Sindhi_ocr_sahash_13_sept__epoch.pth"
print("[INFO] Loading OCR model...")
model, idx_to_char = load_model(MODEL_PATH, device)
print("[INFO] Model loaded successfully ✅")

# -------------------------------------------------------------------

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Upload an image → Segment → OCR → Return recognized text
    """
    try:
        # 1️⃣ Save the uploaded image temporarily
        temp_dir = Path(tempfile.mkdtemp())
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2️⃣ Run the segmenter
        output_dir = temp_dir / "segmented_output"
        segmenter = SindhiTextSegmenter(
            image_path=str(input_path),
            output_dir=str(output_dir),
            line_padding=5,
            word_padding=5,
            letter_padding=7,
            output_height=64,
            denoise_strength=12,
            auto_deskew=True,
            skew_threshold=0.5,
            min_text_pixels=50,
            text_threshold=0.05
        )

        print("\n[INFO] Running Sindhi text segmentation...")
        segmented_images = segmenter.process(segment_level="words")

        if not segmented_images:
            return JSONResponse({"error": "No segmented images found."}, status_code=400)

        # 3️⃣ Run OCR on each segmented word
        recognized_texts = []
        for img_path in segmented_images:
            try:
                img_tensor = preprocess_image(img_path, imgH=48, maxW=512, rtl=True)
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    output = output.log_softmax(2)

                predicted_text = decode_prediction(output, idx_to_char)
                recognized_texts.append(predicted_text)

            except Exception as e:
                logging.error(f"[ERROR] OCR failed for {img_path}: {e}")

        # 4️⃣ Combine text and return as JSON
        full_text = " ".join(recognized_texts)
        return {"recognized_text": full_text.strip()}

    except Exception as e:
        logging.error(str(e))
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
