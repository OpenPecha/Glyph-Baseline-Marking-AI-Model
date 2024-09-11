import numpy as np
import cv2
import csv
import os
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import random


def combined_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    mse = MeanSquaredError()(y_true, y_pred)
    return bce + 0.01 * mse


def load_trained_model(model_save_path):
    return load_model(model_save_path, custom_objects={'combined_loss': combined_loss})


def apply_threshold(image, threshold=0.5):
    return (image > threshold).astype(np.float32)


def create_condition_image(glyph, font_path, image_size, font_size):
    """Create a condition image based on the provided glyph."""
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), glyph, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    cx = bbox[0] + text_width / 2
    cy = bbox[1] + text_height / 2

    x = (image_size[0] / 2) - cx
    y = (image_size[1] / 2) - cy

    draw.text((x, y), glyph, font=font, fill='black')

    return np.array(image) / 255.0


def prepare_inference_data(glyph_image_path, image_size, font_path, font_size):
    """Prepare data for inference by creating condition images."""
    glyph_image = Image.open(glyph_image_path).convert(
        'RGB').resize(image_size)
    glyph_image_array = np.array(glyph_image) / 255.0
    glyph = os.path.basename(glyph_image_path).split('_')[1].split('.')[0]

    condition_image_array = create_condition_image(
        glyph, font_path, image_size, font_size)
    condition_image_array = np.mean(
        condition_image_array, axis=-1, keepdims=True)
    condition_image_array = np.repeat(condition_image_array, 3, axis=-1)
    glyph_image_array = np.expand_dims(glyph_image_array, axis=0)

    input_data = np.concatenate(
        (glyph_image_array, condition_image_array[np.newaxis, ...]), axis=-1)

    return input_data, glyph_image.size


def run_inference(model, input_data, threshold=0.5):
    predictions = model.predict(input_data)
    return apply_threshold(predictions, threshold)


def save_predicted_image(predicted_image, output_path, resized_size):
    predicted_image = (predicted_image.squeeze() * 255).astype(np.uint8)
    predicted_image = Image.fromarray(predicted_image)
    predicted_image = predicted_image.resize(resized_size, Image.BILINEAR)
    predicted_image.save(output_path)
    print(f"Saved predicted image to {output_path}")


def find_bounding_box(image):
    image_cv = np.array(image)
    image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    return [[0, 0], [0, 0], [0, 0], [0, 0]]


def process_images_in_directory(input_glyph_image_dir, output_predicted_dir, model_save_path, font_path, csv_output_path, image_size, font_size, resized_size, threshold=0.5, num_samples=5, random_sampling=True):
    os.makedirs(output_predicted_dir, exist_ok=True)
    glyph_image_files = sorted([f for f in os.listdir(
        input_glyph_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    selected_glyph_files = random.sample(glyph_image_files, min(
        num_samples, len(glyph_image_files))) if random_sampling else glyph_image_files

    csv_dir = os.path.dirname(csv_output_path)
    os.makedirs(csv_dir, exist_ok=True)

    model = load_trained_model(model_save_path)

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "baseline_coordinates"])

        for glyph_file in selected_glyph_files:
            glyph_image_path = os.path.join(input_glyph_image_dir, glyph_file)
            input_data, original_size = prepare_inference_data(
                glyph_image_path, image_size, font_path, font_size)
            predicted_cleaned_image = run_inference(
                model, input_data, threshold)
            output_path = os.path.join(output_predicted_dir, f"{glyph_file}")
            save_predicted_image(predicted_cleaned_image,
                                 output_path, resized_size)

            predicted_image = Image.open(output_path)
            corners = find_bounding_box(predicted_image)
            coordinates = str(corners)
            writer.writerow([glyph_file, coordinates])


if __name__ == "__main__":
    input_glyph_image_dir = 'data/test_images/drepung/cleaned_images'
    output_predicted_dir = 'data/predicted_marking/drepung'
    model_save_path = 'model/baseline_marking_model.keras'
    font_path = 'data/MonlamTBslim.ttf'
    csv_output_path = 'data/baseline_coordinates/drepung_baseline_coordinates.csv'
    image_size = (256, 256)
    font_size = 80
    resized_size = (200, 256)

    process_images_in_directory(input_glyph_image_dir, output_predicted_dir, model_save_path, font_path,
                                csv_output_path, image_size, font_size, resized_size, threshold=0.5, num_samples=5, random_sampling=True)
