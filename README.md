# MNIST Digit Classifier

## Intro

As I'm a strong believer that understanding a topic means, above all, grasping it all the way down to its most basic concepts, I manually recoded the "Hello World!" of machine learning, an MNIST digit classifier. I find this exercise very useful for reminding myself of important ML architecture concepts, as well as key PyTorch syntax and concepts.

The code was entirely written by hand for the data loading, training, and evaluation parts. The Streamlit interface was produced with Claude Sonnet 5.5.

It is important to note that this project makes no claim to be an advanced machine learning project, and I am aware that I did not implement, for example, any kind of validation process to test different hyperparameters. As mentioned, this is primarily a learning and revision project.

## Short description

This project is a handwritten digit classifier trained on the MNIST dataset with PyTorch. It includes a command line pipeline for training and evaluation, as well as a web interface built with Streamlit that lets you draw a digit with your mouse and see how the model interprets it.

## Project structure

```
root/
├── requirements.txt
├── README.md
├── main.py
├── mnist_model.pth
├── src/
│   ├── data_downloader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── app/
    ├── app.py
    ├── predictor.py
    ├── pixel_canvas.py
    └── pixel_canvas_frontend/
        └── index.html
```

The `src` folder contains the learning logic, `main.py` orchestrates it, and the `app` folder gathers everything related to the interface. The `mnist_model.pth` file is generated automatically the first time the model is trained.

## Getting started

### Clone the repository

```
git clone <repository-url>
cd <repository-name>
```

### Set up an environment

Python 3 is required. Creating a virtual environment is recommended so the dependencies stay isolated from the rest of your system.

```
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

### Install the dependencies

```
pip install -r requirements.txt
```

### Train the model

From the root of the project, run the main script.

```
python main.py
```

The MNIST dataset is downloaded into a `data` folder on the first run. The model is then trained for 10 epochs and saved as `mnist_model.pth` at the root of the project. The script ends by evaluating the model on the test set and printing the accuracy it obtained.

### Launch the web interface

Still from the root of the project, start the application.

```
streamlit run app/app.py
```

Streamlit prints a local address in the terminal and usually opens it in your browser automatically. If it does not, open that address manually.

## Training and evaluation

When `main.py` finds a `mnist_model.pth` file at the root, it loads the saved weights instead of training again, then evaluates the model on the test set. To start over with a fresh training, delete that file and run the script again.

## Web interface

The interface needs the `mnist_model.pth` file to exist, so `main.py` must have been run at least once. The application never retrains the model, it only loads the saved weights.

Usage is intentionally simple. You draw a digit on the grid, click Predict to get the answer of the model, and click Reset to start again with an empty grid. The interface displays the predicted digit, the associated confidence, and the probability assigned to each of the ten digits.

## The model

The classifier is a multilayer perceptron. Each 28 by 28 image is flattened into a vector of 784 values, then goes through a configurable number of hidden layers, each followed by a ReLU activation. The last layer outputs ten values, one per digit, which are the logits of the model. By default, the network has two hidden layers of 128 neurons.

Training uses the Adam optimizer with a learning rate of 0.001, batches of 64 images and a cross-entropy loss.

## How the interface works

The drawing area is a grid of 28 by 28 large squares, and each square corresponds exactly to one pixel of the image sent to the model. The brush is soft, which produces shades of gray, with a white stroke on a dark background just like in MNIST.

When a prediction is requested, the image is first recentered on its center of mass, as the digits of the original dataset were. It is then converted to a tensor and passed to the model. The resulting logits are turned into probabilities with a softmax function, and the most likely digit is highlighted. If the drawing is modified after a prediction, the displayed result disappears so that it never stays attached to an image that no longer exists.

## Notes

If you change the architecture of the network, remember to adjust the default values of the `Predictor` class in `app/predictor.py`. Otherwise, loading the weights will fail.

The model is loaded only once when the application starts. After a new training, Streamlit must be restarted for the new weights to be taken into account.

For better results, draw a digit that is fairly large with a thick stroke, around two to three cells wide.