"""
HDFC Bank Form OCR Extractor
Requirements: pip install pytesseract Pillow numpy opencv-python
Also install Tesseract OCR engine: https://github.com/UB-Mannheim/tesseract/wiki
"""

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import json
import re

# -------------------------------------------------------
# STEP 1: LOAD IMAGE
# -------------------------------------------------------
def load_image(image_path):
    img = Image.open(image_path)
    print(f"[INFO] Image loaded: {image_path} | Size: {img.size} | Mode: {img.mode}")
    return img


# -------------------------------------------------------
# STEP 2: PREPROCESS IMAGE FOR BETTER OCR
# -------------------------------------------------------
def preprocess_image(img):
    # Upscale 4x for low-res scanned forms
    width, height = img.size
    img = img.resize((width * 4, height * 4), Image.LANCZOS)

    # Convert to grayscale
    img = img.convert("L")

    # Sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))

    # Boost contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)

    # Convert to numpy for OpenCV processing
    arr = np.array(img)

    # Denoise
    arr = cv2.fastNlMeansDenoising(arr, h=10)

    # Adaptive thresholding (better than simple binary for forms)
    arr = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Deskew
    coords = np.column_stack(np.where(arr < 128))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 10:  # Only correct small skews
            (h, w) = arr.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    processed = Image.fromarray(arr)
    print("[INFO] Preprocessing complete.")
    return processed


# -------------------------------------------------------
# STEP 3: RUN OCR
# -------------------------------------------------------
def run_ocr(img, lang="eng", psm=6):
    """
    PSM modes:
      3  = Fully automatic page segmentation (default)
      4  = Single column of text
      6  = Assume a single uniform block of text (good for forms)
      11 = Sparse text - find as much text as possible
    """
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    print("[INFO] OCR complete.")
    return text


# -------------------------------------------------------
# STEP 4: EXTRACT STRUCTURED DATA (key-value pairs)
# -------------------------------------------------------
def extract_fields(raw_text):
    """
    Attempts to extract common HDFC form fields from raw OCR text.
    """
    fields = {}

    patterns = {
        "Branch Code":        r"Branch\s*Code[:\s]+([A-Z0-9]+)",
        "Form No":            r"Form\s*No[:\s.]+([A-Z0-9\-]+)",
        "First Name":         r"First\s*Name[:\s]+([A-Za-z]+)",
        "Middle Name":        r"Middle\s*Name[:\s]+([A-Za-z]*)",
        "Last Name":          r"Last\s*Name[:\s]+([A-Za-z]+)",
        "Date of Birth":      r"Date\s*of\s*Birth[:\s]+([\d]{1,2}[\/\-][\d]{1,2}[\/\-][\d]{2,4})",
        "PAN":                r"PAN[:\s]+([A-Z]{5}[0-9]{4}[A-Z])",
        "Mobile No":          r"Mobile\s*No[:\s]+([\d\s\-]{10,})",
        "Email":              r"Email[:\s]+([\w\.\-]+@[\w\.\-]+\.\w+)",
        "PIN Code":           r"PIN[:\s]+([\d]{6})",
        "Account Type":       r"Account\s*Type[:\s]+([A-Za-z/ ]+)",
        "Nominee Name":       r"Nominee\s*Name[:\s]+([A-Za-z ]+)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        fields[field] = match.group(1).strip() if match else "Not Found"

    return fields


# -------------------------------------------------------
# STEP 5: SAVE RESULTS
# -------------------------------------------------------
def save_results(raw_text, structured_fields, output_txt="ocr_output.txt", output_json="ocr_fields.json"):
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HDFC BANK FORM — RAW OCR TEXT\n")
        f.write("=" * 60 + "\n\n")
        f.write(raw_text)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_fields, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Raw text saved to: {output_txt}")
    print(f"[INFO] Structured fields saved to: {output_json}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    IMAGE_PATH = "bank1.png"   # <-- Change to your image path

    # Load
    img = load_image(IMAGE_PATH)

    # Preprocess
    img_processed = preprocess_image(img)

    # Save preprocessed image (optional debug)
    img_processed.save("bank1.jpg")
    print("[INFO] Preprocessed image saved:bank1.png")

    # OCR
    raw_text = run_ocr(img_processed, lang="eng", psm=6)

    # Print raw OCR
    print("\n" + "=" * 60)
    print("RAW OCR OUTPUT:")
    print("=" * 60)
    print(raw_text)

    # Extract structured fields
    structured = extract_fields(raw_text)
    print("\n" + "=" * 60)
    print("EXTRACTED FIELDS (Structured):")
    print("=" * 60)
    for k, v in structured.items():
        print(f"  {k:<20}: {v}")

    # Save
    save_results(raw_text, structured)


if __name__ == "__main__":
    main()