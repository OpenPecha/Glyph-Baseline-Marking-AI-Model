import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf

MODEL_SAVE_PATH = 'model/baseline_marking_model.keras'
FONT_PATH = 'data/MonlamTBslim.ttf'
IMAGE_SIZE = (256, 256)
FONT_SIZE = 80


def combined_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    mse = MeanSquaredError()(y_true, y_pred)
    return bce + 0.01 * mse


def load_trained_model():
    return load_model(MODEL_SAVE_PATH, custom_objects={'combined_loss': combined_loss})


def apply_threshold(image, threshold=0.5):
    return (image > threshold).astype(np.float32)


def create_condition_image(glyph, font_path=FONT_PATH, image_size=IMAGE_SIZE, font_size=FONT_SIZE):
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


def prepare_inference_data(glyph_image_path):
    """Prepare data for inference by creating condition images."""
    glyph_image = Image.open(glyph_image_path).convert('RGB').resize(IMAGE_SIZE)
    glyph_image_array = np.array(glyph_image) / 255.0
    glyph = os.path.basename(glyph_image_path).split('_')[1].split('.')[0]

    condition_image_array = create_condition_image(glyph)
    condition_image_array = np.mean(condition_image_array, axis=-1, keepdims=True)
    condition_image_array = np.repeat(condition_image_array, 3, axis=-1)
    glyph_image_array = np.expand_dims(glyph_image_array, axis=0)

    input_data = np.concatenate((glyph_image_array, condition_image_array[np.newaxis, ...]), axis=-1)

    return input_data, glyph_image.size


def run_inference(model, input_data, threshold=0.5):
    predictions = model.predict(input_data)
    return apply_threshold(predictions, threshold)


def save_predicted_image(predicted_image, output_path, original_size):
    predicted_image = (predicted_image.squeeze() * 255).astype(np.uint8)
    predicted_image = Image.fromarray(predicted_image)
    predicted_image = predicted_image.resize(original_size, Image.BILINEAR)
    predicted_image.save(output_path)
    print(f"Saved predicted image to {output_path}")


def process_images_in_directory(glyph_image_dir, output_dir, model, threshold=0.5, num_samples=5, random_sampling=True):
    os.makedirs(output_dir, exist_ok=True)
    glyph_image_files = sorted([f for f in os.listdir(glyph_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    selected_glyph_files = random.sample(glyph_image_files, min(
        num_samples, len(glyph_image_files))) if random_sampling else glyph_image_files

    for glyph_file in selected_glyph_files:
        glyph_image_path = os.path.join(glyph_image_dir, glyph_file)
        input_data, original_size = prepare_inference_data(glyph_image_path)
        predicted_cleaned_image = run_inference(model, input_data, threshold)
        output_path = os.path.join(output_dir, f"{glyph_file}")
        save_predicted_image(predicted_cleaned_image, output_path, original_size)


if __name__ == "__main__":
    model = load_trained_model()
    input_glyph_image_dir = 'data/test_images/derge/cleaned_images'
    output_predicted_dir = 'data/predicted_marking/derge'
    process_images_in_directory(input_glyph_image_dir, output_predicted_dir, model,
                                threshold=0.5, num_samples=5, random_sampling=True)
