
import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.ops import nms
from PIL import Image, ImageDraw

@st.cache_resource  # Cache the model
def load_pretrained_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_cats_and_dogs(image, model, confidence_threshold=0.91, iou_threshold=0.5):

    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    label_map = {17: "cat", 18: "dog"}

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    for i, score in enumerate(scores):
        if score >= confidence_threshold and labels[i].item() in label_map:
            filtered_boxes.append(boxes[i])
            filtered_labels.append(label_map[labels[i].item()])
            filtered_scores.append(score)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    if filtered_boxes:
        filtered_boxes = torch.stack(filtered_boxes)
        filtered_scores = torch.tensor(filtered_scores)
        keep = nms(filtered_boxes, filtered_scores, iou_threshold)

        # Keep only non-overlapping boxes
        filtered_boxes = filtered_boxes[keep].tolist()
        filtered_labels = [filtered_labels[i] for i in keep]

    return filtered_boxes, filtered_labels

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        color = "red" if label == "cat" else "blue" 
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1), label, fill=color)
    return image

def main():
    st.title("Cat and Dog Detection")
    st.write("Upload an image, and the app will detect cats and dogs, drawing bounding boxes around them.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
     
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        model = load_pretrained_model()

        boxes, labels = detect_cats_and_dogs(image, model)

        if boxes:

            result_image = draw_boxes(image.copy(), boxes, labels)
            st.image(result_image, caption="Detection Result", use_container_width=True)
        else:
            st.write("No cats or dogs detected in the image.")

if __name__ == "__main__":
    main()
