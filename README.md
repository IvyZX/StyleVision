# StyleVision

## Overview

StyleVision is an artwork style classifier that can determine the style of a painting or painting. Currently, this repository only contains a pre-trained model with 5 different styles. 

These styles are: Impressionism, Realism, Romanticism, Expressionism and Art Nouveau (Modern)

This tool is written in Python and utilized the Inception-V3, a pre-trained neural network for image object recognition. For more details, see https://tensorflow.org/tutorials/image_recognition/. 

## Dependencies

 - Python 2.7
 - TensorFlow
 - NumPy
 - Scikit-Learn

## How to Use

In the command line, type:

  ```
  python classify_image.py --image_file <path of the image, in JPEG format>
  ```

