import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2  # OpenCV for video processing
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import time

Image.MAX_IMAGE_PIXELS = None
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, 'forest_fire_classification_model.h5')
IMAGE_UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
VIDEO_TEMP_FOLDER = os.path.join(APP_ROOT, 'temp_video_uploads')

ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}

IMG_WIDTH, IMG_HEIGHT = 224, 224
CATEGORIES = ['fire', 'nofire', 'smoke']
CONFIDENCE_THRESHOLD = 0.60

app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER
app.config['VIDEO_TEMP_FOLDER'] = VIDEO_TEMP_FOLDER

if not os.path.exists(IMAGE_UPLOAD_FOLDER):
    os.makedirs(IMAGE_UPLOAD_FOLDER)
if not os.path.exists(VIDEO_TEMP_FOLDER):
    os.makedirs(VIDEO_TEMP_FOLDER)

model = None
print(f"Attempting to load model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {gpus}")
            try:
                # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
                # tf.config.set_logical_device_configuration(
                #     gpus[0],
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) # Example: Limit memory
                # Allow memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for GPUs.")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
        else:
            print("No GPUs available, using CPU.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
else:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image_for_model(image_pil):
    img = image_pil.convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_on_image(image_path):
    if not model:
        return "Model not loaded", None, "The prediction model could not be loaded."
    try:
        pil_img = Image.open(image_path)
        processed_img_array = preprocess_image_for_model(pil_img)
        prediction = model.predict(processed_img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_label_raw = CATEGORIES[predicted_class_idx]
        max_probability = float(np.max(prediction))
        probabilities = {CATEGORIES[i]: float(prediction[0][i]) for i in range(len(CATEGORIES))}
        custom_message = None
        final_predicted_label = predicted_label_raw
        if max_probability < CONFIDENCE_THRESHOLD:
            custom_message = (f"The model's confidence is low ({max_probability*100:.2f}% for '{predicted_label_raw}'). "
                              f"The image might not clearly fall into 'fire', 'nofire', or 'smoke' categories, or it could be ambiguous.")
        return final_predicted_label, probabilities, custom_message
    except FileNotFoundError:
        return f"Error: Uploaded image file not found on server.", None, None
    except Exception as e:
        return f"Error processing image: {str(e)}", None, None

def predict_on_video(video_path, frame_skip=5): # Default frame_skip to 5
    if not model:
        return "Model not loaded", None, 0, "The prediction model could not be loaded."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video.", None, 0, None

    prediction_counts = {category: 0 for category in CATEGORIES}
    total_frames_read = 0
    total_frames_processed = 0
    current_frame_index = 0 # More robust way to track frames for skipping

    video_processing_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        total_frames_read += 1
        
        # Frame skipping logic
        if current_frame_index % frame_skip != 0:
            current_frame_index += 1
            continue
        
        current_frame_index += 1 # Increment even for processed frames
        total_frames_processed += 1

        if total_frames_processed % 20 == 0: # Print progress for every 20 processed frames
            elapsed_time = time.time() - video_processing_start_time
            print(f"Video processing: Read {total_frames_read} frames, Processed {total_frames_processed} frames. Time: {elapsed_time:.2f}s")

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_frame_array = preprocess_image_for_model(pil_img)
        
        # model.predict can be slow if called per frame.
        # For very high performance, batching predictions or using a TFLite model would be better.
        prediction_probs_frame = model.predict(processed_frame_array)[0] 
        predicted_class_idx = np.argmax(prediction_probs_frame)
        label = CATEGORIES[predicted_class_idx]
        prediction_counts[label] += 1

    cap.release()
    video_processing_end_time = time.time()
    print(f"Finished video processing. Total time: {video_processing_end_time - video_processing_start_time:.2f}s")


    if total_frames_processed == 0:
        if total_frames_read > 0:
             return "Error: No frames processed due to frame skipping. Try a smaller frame_skip value.", None, total_frames_read, "All frames skipped."
        return "Error: Video contains no processable frames.", None, 0, None

    final_prediction_label = max(prediction_counts, key=prediction_counts.get)
    
    video_confidence_metric = (prediction_counts[final_prediction_label] / total_frames_processed) if total_frames_processed > 0 else 0
    custom_message = f"Processed {total_frames_processed} out of {total_frames_read} frames (skipped {frame_skip-1} of every {frame_skip} frames)."
    
    if video_confidence_metric < CONFIDENCE_THRESHOLD:
        low_confidence_msg = (f"The dominant prediction ('{final_prediction_label}') appeared in "
                              f"{video_confidence_metric*100:.2f}% of processed frames. "
                              f"The video content might be mixed or ambiguous.")
        custom_message += " " + low_confidence_msg
        
    return final_prediction_label, prediction_counts, total_frames_processed, custom_message


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error="No file part in the request.")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No file selected.")
    if not model:
        return render_template('result.html', error="Model not loaded. Cannot make predictions. Please check server logs.")

    filename = secure_filename(file.filename)
    display_filename_for_html = None

    if allowed_file(filename, ALLOWED_EXTENSIONS_IMAGE):
        filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return render_template('result.html', error=f"Error saving uploaded image: {str(e)}")
        prediction_label, probabilities, custom_message = predict_on_image(filepath)
        display_filename_for_html = os.path.join('uploads', filename)
        return render_template('result.html',
                               prediction_label=prediction_label,
                               probabilities=probabilities,
                               filename=display_filename_for_html,
                               upload__type='image',
                               custom_message=custom_message)

    elif allowed_file(filename, ALLOWED_EXTENSIONS_VIDEO):
        temp_video_filepath = os.path.join(app.config['VIDEO_TEMP_FOLDER'], filename)
        try:
            file.save(temp_video_filepath)
        except Exception as e:
            return render_template('result.html', error=f"Error saving uploaded video: {str(e)}")
        
        FRAME_SKIP_RATE = 10 # Process 1 in every 10 frames. ADJUST THIS!
        final_prediction, prediction_counts, total_frames_proc, custom_message = predict_on_video(temp_video_filepath, frame_skip=FRAME_SKIP_RATE)
        
        try:
            if os.path.exists(temp_video_filepath):
                os.remove(temp_video_filepath)
        except Exception as e:
            print(f"Warning: Could not remove temporary video file {temp_video_filepath}: {e}")
        
        return render_template('result.html',
                               prediction_label=final_prediction,
                               video_counts=prediction_counts,
                               total_frames=total_frames_proc, # Show processed frames
                               upload_type='video',
                               custom_message=custom_message)
    else:
        return render_template('result.html', error="Invalid file type. Allowed images: .png, .jpg, .jpeg, .gif. Allowed videos: .mp4, .avi, .mov, .mkv")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False for production/timing