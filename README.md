<h1 align="center">Team Members: Yuri Armando, Simone Nardi, Francesco Bassignana, Pietro Jomini</h1>
<h3 align="center">ITIS M. DELPOZZO</h3>

- 🔭We are currently working on [Trash Sorter](https://github.com/PietroJominiITIS/trash-sorter)

- 🌱 We're currently using **Python, Teachable Machine, Keras**



<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

<p><img align="center" src="https://github-readme-stats.vercel.app/api/top-langs?username=pietrojominiitis&show_icons=true&locale=en&layout=compact" alt="pietrojominiitis" /></p>

# Trash sorter with keras and teachable machines


The project involves the recognition of an object (garbage) thanks to the webcam, with the subsequent categorization and sorting

## Making of (software) 
Finding and creating images out of junk
```bash
https://drive.google.com/drive/u/1/folders/1IaTeHUQ3Si47oypWl3FTsWGKIGZnvUij
```

Creation and training of image recognition through the site
```python
https://teachablemachine.withgoogle.com/
```
Writing python code to start the webcam 

```python
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

``` 
Writing Python code to start image recognition 
```bash
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
``` 
## Making of (Hardware)
Webcam
Raspberry Pi
Monitor


## Usage

```txt
Connect the hardware parts and initialize the program. 
Now a webcam window will open. Place the object in front of the webcam and wait for the recognition then repeat the process for the number of waste
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
