from PIL import Image, ImageOps
from typing import Dict, List
from tensorflow import keras
from pathlib import Path
import numpy as np
import cv2 as cv
import yaml


def read_config(path: str = "config.yaml") -> Dict[str, str]:
    """Read config from file"""

    # Read file content
    path = Path(path)
    content = path.read_text()

    # Parse file content
    config = yaml.safe_load(content)

    return config


def read_labels(config: Dict[str, str]) -> List[str]:
    """Read labels from file"""

    # Read file content
    path = Path.cwd() / Path(config["labels"])
    content = path.read_text()

    # Parse file content
    labels = content.splitlines()
    labels = [label.split(" ") for label in labels]
    labels = [label for _, label in labels]

    return labels


def main():
    """Main routine"""

    # Read config from file
    config = read_config()

    # Read labels from file
    labels = read_labels(config)

    # Open opencv camera feed
    camera = cv.VideoCapture(config["camera"]["id"])

    # Save current material, to display changes
    current_material = -1

    # Load model
    model = keras.models.load_model(config["model"])

    # Array to feed to the keras model
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)

    running = True
    while running:

        # Read frame from opencv video feed
        _, frame = camera.read()

        # Resize image
        image = Image.fromarray(frame)
        image = ImageOps.fit(image, config["camera"]["size"], Image.ANTIALIAS)

        # Convert the PIL image back to a numpy ndarray
        resized_frame = np.asarray(image)

        # Normalize the image
        normalized_frame = (resized_frame.astype(np.float32) / 127.0) - 1

        # Load the frame into te ndarray
        data[0] = normalized_frame

        # Run the inference
        prediction = model.predict(data)

        # Select best fitting option
        prediction = enumerate(prediction)
        top_prediction = max(prediction, key=lambda value: value[1])
        index = top_prediction[0]
        label = labels[index]

        # Display material changes
        if index != current_material:
            current_material = index
            print(label)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
