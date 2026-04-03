from ultralytics import YOLO
import easyocr
import cv2
import re

# Load models
yolo_model = YOLO("../yolov8n.pt")
ocr_reader = easyocr.Reader(['en'])

files = ["cocacola.webp", "mirinda.webp", "red_bull.webp", "sprite.webp"]

detections = []

# 🔥 Better product identification
def identify_product(label, text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())

    if any(word in text for word in ["coca", "cola", "taste"]):
        return "Coca Cola"
    
    elif "sprite" in text:
        return "Sprite"
    
    elif "mirinda" in text:
        return "Mirinda"
    
    elif "red bull" in text or "bull" in text:
        return "Red Bull"
    
    elif label == "bottle":
        return "Generic Bottle Product"
    
    elif label == "can":
        return "Generic Can Product"
    
    return "Unknown"


# 🔍 PROCESS EACH IMAGE PROPERLY
for file in files:
    image = cv2.imread("image/" + file)
    results = yolo_model(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            cropped = image[y1:y2, x1:x2]

            # OCR
            ocr_result = ocr_reader.readtext(cropped)
            text_detected = " ".join([res[1] for res in ocr_result])

            detections.append({
                "file": file,
                "label": label,
                "confidence": confidence,
                "text": text_detected
            })


# 📦 BUILD INVENTORY
inventory = {}

for d in detections:
    product = identify_product(d["label"], d["text"])
    inventory[product] = inventory.get(product, 0) + 1

    print("-----------")
    print(f"File: {d['file']}")
    print(f"Label: {d['label']}")
    print(f"OCR Text: {d['text']}")
    print(f"Mapped Product: {product}")

print("\nFINAL INVENTORY:")
print(inventory)