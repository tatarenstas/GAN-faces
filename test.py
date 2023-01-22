import numpy as np
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt

generator = keras.models.load_model('face_generator.h5')

for i in range(1, 1001):
    SEED_SIZE = 100
    PREVIEW_ROWS = 1
    PREVIEW_COLS = 1
    PREVIEW_MARGIN = 1
    GENERATE_RES = 3
    GENERATE_SQUARE = 32 * GENERATE_RES
    IMAGE_CHANNELS = 3

    noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
    image_array = np.full((PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)),
          PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), IMAGE_CHANNELS),
          255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
          r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
          c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
          image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
          image_count += 1
    img = Image.fromarray(image_array)
    #plt.imshow(img)
    #plt.show()
    img.save(f"generated_images/generated_face_{i}.png")
