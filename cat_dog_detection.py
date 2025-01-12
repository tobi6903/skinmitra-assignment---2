import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw


@st.cache_resource  # Cache the model
def load_pretrained_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True) 
    model.eval() 
    return model


def detect_cats_and_dogs(image, model, confidence_threshold=0.91):

    image_tensor = F.to_tensor(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    label_map = {17: "cat", 18: "dog"}

    filtered_boxes = []
    filtered_labels = []
    for i, score in enumerate(scores):
        if score >= confidence_threshold and labels[i].item() in label_map:
            filtered_boxes.append(boxes[i].cpu().numpy())
            filtered_labels.append(label_map[labels[i].item()]) 

    return filtered_boxes, filtered_labels

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red")
    return image

def main():
    st.title("Cat and Dog Detection")
    st.write("Upload an image, and the app will detect cats and dogs if present, drawing bounding boxes around them.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
    
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        model = load_pretrained_model()

        boxes, labels = detect_cats_and_dogs(image, model)

        if boxes:

            result_image = draw_boxes(image.copy(), boxes, labels)
            st.image(result_image, caption="Detection Result", use_column_width=True)
        else:
            st.write("No cats or dogs detected in the image.")

if __name__ == "__main__":
    main()
