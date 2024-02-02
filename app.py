from flask import Flask, render_template, request
from collections import Counter
from ultralytics import YOLO
import openai
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "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"

# Initialize chatbot messages with default chef bot
messages = [
    {"role": "system", "content": "Hi, I am Your Chef."},
    {"role": "user", "content": "Suggest a recipe based on the detected ingredients."}
]

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def boundingboxPredicted(results, model, image_path):
    output_folder = 'predictions'  # Define the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)

    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        # Save the original image with bounding boxes
        output_image_path = os.path.join(output_folder, 'predictions.jpg')
        cv2.imwrite(output_image_path, img)
        print(f"Predictions saved in {output_folder}")

def run_object_detection(image_path):
    model_directory = r"D:\project-finesh\project"
    model_filename = "new_model.pt"
    model_path = os.path.join(model_directory, model_filename)

    infer = YOLO(model_path)
    result = infer.predict(image_path)
    item_counts = Counter(infer.names[int(c)] for r in result for c in r.boxes.cls)
    object_list = list(item_counts.keys())
    boundingboxPredicted(result, infer, image_path)
    return object_list


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return render_template('result.html', error="No file part")

    image_file = request.files['image']

    # If the user does not select a file, browser also
    # submit an empty part without filename
    if image_file.filename == '':
        return render_template('result.html', error="No selected file")

    # Check if the file is allowed (optional)
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('result.html', error="Invalid file type")

    # Save the image without using secure_filename
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the image using the correct relative path
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    
    # Print the image path for debugging
    print(f"Image saved at: {image_path}")
    # Run object detection and get the recipe prompt
    object_list = run_object_detection(image_path)

    # Ask for recipe suggestions based on object detection results
    recipe_prompt = f"I detected {', '.join(object_list)} in the image. Suggest a recipe for these ingredients."
    messages.append({"role": "user", "content": recipe_prompt})

    # Get chatbot response using OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Display chatbot response
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})

    # Pass variables to the template
    return render_template('result.html', reply=reply, image_filename=image_file.filename, object_list=object_list)


if __name__ == '__main__':
    app.run(debug=True)
