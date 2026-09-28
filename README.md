# Emotion Cat Meme Detector

A facial emotion recognition project built with a convolutional neural network (CNN). It reads a webcam feed, detects a face, predicts one of seven emotions, and shows a cat meme that matches the emotion.

## Features

- CNN trained on 48x48 grayscale face images
- Seven emotion classes: angry, disgust, fear, happy, neutral, sad and surprise
- Real-time webcam demo with face detection
- Matching cat meme displayed for each predicted emotion
- Two model versions: a baseline CNN and an improved CNN with batch normalization and data augmentation

## How it works

The live demo follows this pipeline:

1. Grab a frame from the webcam, flip it horizontally like a mirror and resize it to 700x500
2. Convert it to grayscale and detect faces with OpenCV's Haar cascade
3. Crop each detected face and resize it to 48x48
4. Divide the pixel values by 255
5. Feed the image to the CNN
6. Take the emotion with the highest probability
7. Draw a box and the emotion label on the frame, and overlay the matching cat meme on the right side

## Dataset

The model is trained on a facial emotion dataset of 48x48 grayscale images, split into a train folder and a test folder with one subfolder per emotion.

- Training images: 28,821
- Test images: 7,066
- Classes: 7
- Input shape: (48, 48, 1)

Dataset link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

The dataset is not included in this repository because of its size and licensing. To retrain the model, download it and arrange it like this:

```
images/
    train/
        angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
    test/
        angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

## Development process

### Step 1: Data loading

Image paths and labels are collected from each emotion folder into a pandas dataframe. Every image is loaded as a 48x48 grayscale array, stacked into a single NumPy array, and divided by 255 so that pixel values fall between 0 and 1. Labels are converted to numbers with scikit-learn's LabelEncoder and then to one-hot vectors with `to_categorical`.

### Step 2: Hardware

Training on a cousin's PC with an RTX 5060 GPU was the original plan, but the distance made that impossible. Both models were therefore trained on a laptop with no dedicated GPU, using only the CPU (Intel Core i5-1135G7). This was slower than GPU training but worked without problems.

### Step 3: Version 1, baseline CNN

The first model is a plain CNN:

- Four blocks of Conv2D (128, 256, 512, 512 filters), each followed by MaxPooling and Dropout 0.4
- Flatten
- Dense 512 with Dropout 0.4, then Dense 256 with Dropout 0.3
- Dense 7 with softmax

It was trained with the Adam optimizer, categorical crossentropy, batch size 120 and early stopping (patience 5). Training ran for the full 30 epochs and reached about 60.6% validation accuracy. The model is saved as `emotion_detector_model_0`.

Adding up the epoch times from the notebook, training took about 1 hour 6 minutes (roughly 2 minutes per epoch).

### Step 4: Webcam integration

Before improving the model, the full pipeline was connected end to end. `real_time_detection.py` loads the saved model (JSON architecture plus .h5 weights), finds faces with OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`), applies the same preprocessing used in training, and runs the model on each detected face. It draws a box and the predicted emotion on the frame and pastes a 240x240 meme from the `memes` folder onto the right side of the window. Pressing Esc closes the window.

Memes are loaded by name, so the `memes` folder needs one image per emotion: `angry.jpg`, `disgust.jpg`, `fear.jpg`, `happy.jpg`, `neutral.jpg`, `sad.jpg` and `surprise.jpg`. The label order in the script matches the alphabetical order used by LabelEncoder during training, which is required because the model outputs class numbers.

### Step 5: Version 2, improved CNN

Since about 60% accuracy left room for improvement, the architecture was redesigned instead of simply enlarged:

- Two convolution layers per block instead of one, with filters going 64, 128, 256, 512
- Batch normalization after every convolution, with ReLU applied after it
- GlobalAveragePooling2D instead of Flatten, which avoids the very large jump in dense layer parameters
- A smaller Dense 256 layer with L2 regularization (0.01) and Dropout 0.5
- Data augmentation on the training images: rotation up to 15 degrees, width and height shifts of 15%, zoom of 15%, and horizontal flips
- ReduceLROnPlateau, which halves the learning rate when validation loss stops improving (patience 3, minimum 1e-6)
- EarlyStopping on validation loss (patience 10)

Training ran for 70 epochs at batch size 120. Adding up the epoch times from the notebook, it took about 4 hours 10 minutes, with each epoch taking roughly 3 to 4 minutes. The model is saved as `emotion_detector_model_1`, and the webcam demo loads this version.

## Results

| Model | Epochs | Best validation accuracy |
|-------|--------|--------------------------|
| Version 1 (baseline CNN) | 30 | about 60.6% |
| Version 2 (BatchNorm, augmentation, GAP) | 70 | 66.47% (epoch 65) |

Observations:

- Version 2 improved on Version 1 by about 6 percentage points.
- In Version 2, training accuracy (about 66%) and validation accuracy (about 66.5%) ended up very close, which indicates the model is no longer overfitting. Augmentation and regularization were effective.
- Early stopping never triggered. The learning rate decayed to about 2e-6 and validation accuracy stayed flat from around epoch 40 onward, so training for longer would not have helped.

Note: the test set was used as the validation set during training (for early stopping and learning rate reduction), so the 66.5% figure is validation accuracy on that set and not a fully unbiased final test score. A stricter setup would use a separate validation split and keep the test set untouched until the end. The same data was used for both versions so that they could be compared fairly.

## Project structure

```
.
    model_training.ipynb       training notebook (data loading, V1, V2)
    real_time_detection.py     webcam demo
    requirements.txt           Python dependencies
    models/                    saved models (.json architecture and .h5 weights)
    memes/                     one cat meme per emotion
```

The `images` folder (the dataset) is not included, as explained above.

## Installation and usage

1. Clone the repository.
2. Create an environment and install the dependencies:

```
pip install -r requirements.txt
```

3. Start the webcam demo from the project root folder, since the script uses relative paths to find `models` and `memes`:

```
python real_time_detection.py
```

Press Esc to close the window.

The project was developed with TensorFlow 2.10.0 and NumPy 1.23.5 inside a conda environment. If version errors occur, try matching those versions.

To retrain the models, download the dataset, arrange the folders as shown above, and run `model_training.ipynb` from top to bottom.

## Limitations

- Accuracy is around 66%, and the model is weaker on emotions that look similar, such as fear, sad and angry.
- The dataset images are small and low quality, so the model can behave differently on a real webcam with different lighting and angles.
- The model outputs one of seven fixed emotions, so it always picks something, even for a face that fits none of them well.
- The Haar cascade face detector works best on faces looking straight at the camera in decent light, and it sometimes misses faces or gives false detections.
- Predictions are made on every frame with no smoothing, so the label and meme can flicker between emotions.
- Predictions may be less accurate when the person is wearing glasses.

## Built with

Python, TensorFlow and Keras, OpenCV, NumPy, pandas, scikit-learn, tqdm, Jupyter Notebook, VS Code.
