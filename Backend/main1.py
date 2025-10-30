# from segmenter import SindhiTextSegmenter

# segmenter = SindhiTextSegmenter(
#     image_path="input_images/image1.png",
#     output_dir="output_sindhi",
#     line_padding=5,
#     word_padding=5,
#     letter_padding=7,
#     output_height=64,
#     denoise_strength=12,
#     auto_deskew=True,
#     skew_threshold=0.5,
#     min_text_pixels=50,      # For lines and words
#     text_threshold=0.05      # For lines and words
# )

# # Now this will generate ligatures!
# segmenter.process(segment_level="ligatures")


# main.py
import os
import torch
from segmenter import SindhiTextSegmenter
from test import preprocess_image, load_model, decode_prediction

def main():
    # 1️⃣ Segment the image
    segmenter = SindhiTextSegmenter(
        image_path="input_images/image1.png",
        output_dir="segmented_output",
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
        print("[ERROR] No segmented images found.")
        return

    # 2️⃣ Load the OCR model
    model_path = "model/Sindhi_ocr_sahash_20_epoch.pth"  # <-- your .pt file here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model, idx_to_char = load_model(model_path, device)

    # 3️⃣ Run OCR on each segmented image
    print("\n[INFO] Running OCR on segmented images...\n")

    for img_path in segmented_images:
        try:
            img_tensor = preprocess_image(img_path, imgH=48, maxW=512, rtl=True)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                output = model(img_tensor)
                output = output.log_softmax(2)

            predicted_text = decode_prediction(output, idx_to_char)

            print(f"🖼️ {os.path.basename(img_path)} → {predicted_text}")

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

if __name__ == "__main__":
    main()
