# Object Detection with Bounding Boxes for Cats and Dogs

## Project Description
This project uses a pre-trained Faster R-CNN model to detect objects in an image and specifically filters for cats and dogs. 

## Features
1. Detect objects in images using a pre-trained Faster R-CNN model.
2. Filter detections specifically for cats and dogs.
3. Display images with bounding boxes drawn around the detected cats and dogs.
4. Built using Streamlit for an interactive web application experience.

## Requirements
To run this project, you will need the following libraries installed:

- Python 3.8+
- torch
- torchvision
- streamlit
- pillow

Install the required packages using the following command:
```bash
pip install torch torchvision streamlit pillow
```

## Dataset
This project does not require a specific dataset for training, as it utilizes a pre-trained Faster R-CNN model on the COCO dataset. Simply provide any image as input for inference.

## How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tobi6903/skinmitra-assignment---2.git
   cd https://github.com/tobi6903/skinmitra-assignment---2.git
   ```

2. **Run the Streamlit Application**
   ```bash
   streamlit run cat_dog_detection.py
   ```

3. **Upload an Image**
   - Open the web application in your browser (default: `http://localhost:8501`).
   - Use the file uploader to select an image.

4. **View the Results**
   - If cats or dogs are detected in the image, bounding boxes will be drawn around them.
   - The labeled and processed image will be displayed in the app.

## Code Explanation

### Key Functions

1. **`load_pretrained_model()`**
   - Loads a pre-trained Faster R-CNN model and sets it to evaluation mode.

2. **`detect_cats_and_dogs(image_path, model, confidence_threshold)`**
   - Performs object detection on the input image.
   - Filters results for cats and dogs based on a confidence threshold.

3. **`draw_boxes(image, boxes, labels)`**
   - Draws bounding boxes and labels on the detected objects in the image.

4. **Streamlit Integration**
   - A file uploader allows users to upload an image.
   - Results are displayed interactively with bounding boxes drawn on detected cats and dogs.


## Sample Output
When an image containing cats or dogs is uploaded, the application will:
1. Display the original image.
2. Show the processed image with bounding boxes around detected cats and dogs.

