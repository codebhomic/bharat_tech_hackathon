from ultralytics import YOLO
import easyocr
import cv2

yolo_model = YOLO("../yolov8n.pt")
ocr_reader = easyocr.Reader(['en'])

image = cv2.imread("image/cocacola.webp")

results = yolo_model(image)

detections = []

def identify_product(label, text):
    text = text.lower()

    if "dove" in text:
        return "Dove Shampoo"
    
    elif "lux" in text:
        return "Lux Soap"
    
    elif "coca cola" in text:
        return "Coca Cola"
    
    elif "original taste" in text:
        return "Coca Cola"
    
    elif label == "bottle":
        return "Generic Shampoo"
    
    elif label == "box":
        return "Generic Soap"
    
    return "Unknown"

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = yolo_model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        # Crop detected object
        cropped = image[y1:y2, x1:x2]

        detections.append({
            "label": label,
            "confidence": confidence,
            "image": cropped
        })  

for d in detections:
    ocr_result = ocr_reader.readtext(d["image"])
    
    text_detected = " ".join([res[1] for res in ocr_result])
    
    d["text"] = text_detected

inventory = {}

for d in detections:
    product = identify_product(d["label"], d["text"])
    
    inventory[product] = inventory.get(product, 0) + 1

print(f"Detected text: {d["text"]}\n detected label: {d["label"]}\nInventory identified: {inventory}")