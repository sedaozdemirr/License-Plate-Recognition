

# License-Plate-Recognition
License Plate Detection and Character Extraction
This project demonstrates license plate detection and character extraction from an image using Python and various computer vision techniques.


## Prerequisites
Before running the code, ensure you have the following dependencies installed:

Python 3.x
OpenCV (cv2)
Matplotlib
NumPy
## You can install the dependencies using the following command:
pip install opencv-python matplotlib numpy

## Getting Started
Clone the repository to your local machine.

Place your dataset images in the "veriseti" folder.

Run the Python script "license_plate_detection.py".

The script will process the images one by one and save the extracted characters in the "karakterseti" folder.
## Usage
Run the script "license_plate_detection.py" with the required dependencies installed.

The script will automatically process the images present in the "veriseti" folder.

It will perform the following steps:

Read an image from the dataset folder.
Detect the license plate region using "alg1_plaka_tespiti" (license plate detection algorithm).
Extract the license plate from the image.
Enhance the image resolution and convert it to grayscale.
Apply adaptive thresholding to segment the characters.
Remove noise using morphological operations.
Find contours of the characters on the license plate.
Filter and extract individual characters.
Save the extracted characters in the "karakterseti" folder.
The extracted characters will be saved as separate images with the format "original_filename_index.jpg" in the "karakterseti" folder.
## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, feel free to open a new issue or submit a pull request.
## Acknowledgments
The license plate detection algorithm "alg1_plaka_tespiti" used in this project is provided by sedaozdemirr.