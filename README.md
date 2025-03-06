# Lane Line Detection Technical Document

- [Lane Line Detection Technical Document](#lane-line-detection-technical-document)
  - [Introduction](#introduction)
  - [Download Everything You Need](#download-everything-you-need)
    - [Required Apps](#required-apps)
    - [Download and Configure VS Code](#download-and-configure-vs-code)
      - [Step 1: Download VS Code](#step-1-download-vs-code)
      - [Step 2: Install VS Code](#step-2-install-vs-code)
  - [Write the Python Program to Detecting Image](#write-the-python-program-to-detecting-image)
    - [Why Use Python Virtual Environments in VS Code?](#why-use-python-virtual-environments-in-vs-code)
    - [Import Everything You Need](#import-everything-you-need)
      - [The Libraries We Need](#the-libraries-we-need)
      - [How to Install These Libraries?](#how-to-install-these-libraries)
      - [Import These Libraries](#import-these-libraries)
    - [The Processes in the Detection](#the-processes-in-the-detection)
    - [Start Coding](#start-coding)
      - [1. Image Reading](#1-image-reading)
      - [2. Grayscale Processing](#2-grayscale-processing)
      - [3. Gaussian Smoothing](#3-gaussian-smoothing)
      - [4. Canny Edge Detecting](#4-canny-edge-detecting)
      - [5. Region Masking](#5-region-masking)
      - [6. Hough Transforming](#6-hough-transforming)
      - [7. Fusion Image and Output](#7-fusion-image-and-output)
      - [8. Main Function](#8-main-function)
    - [Test the Code](#test-the-code)
  - [What about Detecting Lane Line in a Video](#what-about-detecting-lane-line-in-a-video)
  - [Shortcomings](#shortcomings)


## Introduction
This project is used to detect the lane line based on the Python and Open CV.
__Project Adress:__ [Lane Line Detection](https://www.kaggle.com/code/soumya044/lane-line-detection)
***
## Download Everything You Need
### Required Apps
VS Code, Necessary Libraries for Python

### Download and Configure VS Code
#### Step 1: Download VS Code
Tap the following link and download the installer based on your operation system.
__Download Link:__ https://code.visualstudio.com/download
#### Step 2: Install VS Code
Visual Studio Code (VS Code for short) is a free and open-source code editor developed by Microsoft. It runs on multiple platforms such as Windows, macOS, and Linux, and supports multiple programming languages

__1. Click User Agreement__
<img src="Download VS Code\step1.png" alt="Click User Agreement">

__2. Choose Installation Location__
<img src="Download VS Code\step2.png" alt="Choose Installation Location">
==_Recommend installing VS Code on a disk other than the C Drive._==

__3. Create the Shortcut__
<img src="Download VS Code\step3.png" alt="Create the Shortcut">

__4. Additional Tasks__
<img src="Download VS Code\step4.png" alt="Additional Tasks">
==_Highly recommend choose the options as the picture showing._==

Now, you've well installed the VS Code. 
Let's continue to configure the environment of VS Code.
<img src="Configuire VS Code\Welcome Page.png" alt="Welcome to VS Code">

__5. Switch to Chinese (Optional)__
* Tap Extensions
* Search "Chinese"
* Tap install
* Restart

<img src="Configuire VS Code\Switch to Chinese.png" alt="Switch to Chinese">

***
## Write the Python Program to Detecting Image

__Environment: Python 3.12.0(.'venv')__
### Why Use Python Virtual Environments in VS Code?
__1. Isolation of Dependencies__
Each project can have its own virtual environment with specific package versions that won't interfere with other projects.
__2. Consistent Development Environment__
Ensures that everyone working on the project uses the same package versions, avoiding compatibility issues.
__3. Better Project Management__
Allows you to easily manage and reproduce the project's dependencies, making it easier to deploy or share the project.
__4. VS Code Integration__
VS Code has good support for Python virtual environments, making it easy to set up and switch between different environments.

### Import Everything You Need
#### The Libraries We Need
* __OpenCV-Python:__ OpenCV-Python is a library for computer vision tasks. It provides a wide range of functions for image and video processing, object detection, and more.
* __NumPY:__ NumPy is a fundamental library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
* __Matplotlib:__ Matplotlib is a plotting library for creating static, interactive, and animated visualizations in Python. It is widely used for data visualization and plotting graphs.
* __Math:__ The math module is a built-in Python module that provides mathematical functions and constants. It includes functions for trigonometry, logarithms, and more.
* __OS:__ The os module provides a way to interact with the operating system. It includes functions for file and directory operations, environment variables, and more.

#### How to Install These Libraries?
__1. Set Up Python Environment:__
Make sure you have Python installed on your system.
Install the Python extension in VS Code.
Open the command palette (Ctrl+Shift+P) and select "Python: Select Interpreter" to choose the correct Python environment.

<img src="Install Pip\Select Interpreter.png" alt="Python: Select Interpreter">
<img src="Install Pip\Python venv.png" alt="Choose Virtual Environment">

__2. Install pip:__
If pip is not installed, run the following command in the VS Code terminal:
python -m ensurepip --upgrade

__3. Install Packages:__
Use the pip install package_name command in the VS Code terminal to install any required packages.
```python
pip install opencv-python numpy matplotlib 
"""
"Math" and "OS" are in the Python Standard Library, 
we can import it using "import math" and "import os"
"""
```

#### Import These Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
```

### The Processes in the Detection
__1. Gray Scale__
__2. Gaussian Smoothing__
__3. Canny Edge Detection__
__4. Region Masking__
__5. Hough Transform__
__6. Draw Lines [Mark Lane Lines with Different Color]__

### Start Coding
==_In the following code, I will assume that you have correctly imported all the required Python libraries._==
#### 1. Image Reading

```python
img1 = mping.imread('location/pic_name.jpg')
img2 = cv2.imread('location/pic_name.jpg')
"""
The Differences:
1. mping.imread(): Read the image in RGB format. 
   Returns an array of type 'float32'. 
   The pixel value range is [0, 1]
2. cv2.imread(): Read the image in BGR format. 
   Returns an array of type 'uint8'. The pixel value range is [0, 255]. 
   If we want to read in RGB format, we can convert it 
   by using 'cv2.cvtColor(image, cv2.COLOR_BGR2RGB)'.
"""
```

#### 2. Grayscale Processing
__Grayscale processing__ converts an RGB image to grayscale, representing pixel brightness on a scale from 0 (black) to 255 (white). This step is crucial for lane line detection as it:
__Reduces Computational Complexity:__ Grayscale images have one channel instead of three (RGB), speeding up processing.
__Enhances Edge Detection:__ Many edge detection algorithms (e.g., Canny) work better on grayscale images.
__Removes Irrelevant Color Information:__ Lane lines are typically uniform in color, so grayscale preserves essential brightness variations.

```python
def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 3. Gaussian Smoothing
__Gaussian Blur__ is used to smooth images in lane line detection. It helps to:
__Reduce Noise:__ Minimizes random variations in pixel intensities
__Reduce Detail:__ Simplifies the image while preserving essential features
__Improve Robustness:__ Makes the detection less sensitive to small variations.

```python
def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
#### 4. Canny Edge Detecting
__The Canny Edge Detection__ algorithm is used to identify edges in images. In lane line detection, it helps to:
__Identify Lane Line Edges:__ Accurately detect the boundaries of lane lines
__Reduce Noise:__ Minimize false detection of lane lines
__Improve Accuracy:__ Provide precise edge information for further processing.

```python
def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform.
    """
    return cv2.Canny(img, low_threshold, high_threshold)
```

#### 5. Region Masking
__Region masking__ is used to focus on specific regions of an image while ignoring others. In lane line detection, it helps to:
__Focus on Relevant Area:__ Isolate the region where lane lines are likely to appear
__Remove Distractions:__ Ignore irrelevant parts of the image like the sky or other vehicles
__Improve Detection Accuracy:__ Reduce false positives by limiting processing to the area of interest.

```python
def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.15, rows]
    top_left     = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

#### 6. Hough Transforming
__The Hough Transform__ is used to detect lines in images. In lane line detection, it helps to:
__Identify Lane Lines:__ Convert edge-detected images into lines
__Handle Noise:__ Ignore small gaps in lane lines
__Improve Robustness:__ Detect lines even if they are partially occluded.

```python
def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image,lines):
    
    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]

    left_lines = [] # Like /
    right_lines = [] # Like \
    for line in lines:
        for x1,y1,x2,y2 in line:

            if x1 == x2:
                pass #Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m,c))
                elif m >= 0:
                    right_lines.append((m,c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    #print(left_line, right_line)

    for slope, intercept in [left_line, right_line]:

        #getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1= int(rows) #image.shape[0]

        #taking y2 upto 60% of actual height or 60% of y1
        y2= int(rows*0.6) #int(0.6*y1)

        #we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1,y1,x2,y2]]]))
    
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts = np.array([poly_vertices],'int32'), color = (0,255,0))
    return cv2.addWeighted(image,0.7,img,0.4,0.)
    
    #cv2.polylines(img,np.array([poly_vertices],'int32'), True, (0,0,255), 10)
    #print(poly_vertices)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    line_img = slope_lines(line_img,lines)
    return line_img
```

#### 7. Fusion Image and Output
__Fusion Image__ is used to blend two images together. In lane line detection, it is commonly used to:
__Overlay Detected Lane Lines:__ Combine the detected lane lines with the original image
__Adjust Opacity:__ Control the visibility of the lane lines overlay
__Enhance Visualization:__ Create a more intuitive result by showing both the original image and the detected lane lines.

```python
def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    #lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges
```

#### 8. Main Function

```python
def lane_finding_pipeline(image):
    
    #Grayscale
    gray_img = grayscale(image)
    #Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 5)
    #Canny Edge Detection
    canny_img = canny(img = smoothed_img, low_threshold = 180, high_threshold = 240)
    #Masked Image Within a Polygon
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image))
    #Hough Transform Lines
    houghed_lines = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    #Draw lines on edges
    output = weighted_img(img = houghed_lines, initial_img = image, α=0.8, β=1., γ=0.)
    
    return output
```

### Test the Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
from LLD5_Lane_Detection_Pipeline import lane_finding_pipeline
# "LLD5_Lane_Detection_Pipeline" is the file name that store the main function

for image_path in list(os.listdir('CarND-LaneLines-P1/test_images')):
    fig = plt.figure(figsize=(20, 10))
    image = mpimg.imread(f'CarND-LaneLines-P1/test_images/{image_path}')
    ax = fig.add_subplot(1, 2, 1,xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title("Input Image")
    ax = fig.add_subplot(1, 2, 2,xticks=[], yticks=[])
    plt.imshow(lane_finding_pipeline(image))
    ax.set_title("Output Image [Lane Line Detected]")
    plt.show()
```

> __If the code runs well, we will get the following figure.__
> <img src="Detected Image.png" alt="Detected Image">

***
## What about Detecting Lane Line in a Video
__Reason for Not Using MoviePy:__
* MoviePy might not be compatible with Python 3.12 due to dependencies or API changes in newer Python versions.

__Why Use OpenCV (cv2):__
* OpenCV is a powerful library for computer vision tasks, including video processing.
* It provides direct access to video frames, making it efficient for real-time processing.
* OpenCV supports a wide range of video codecs and formats.
```python
import cv2
import time
import numpy as np
from LLD5_Lane_Detection_Pipeline import lane_finding_pipeline

# Set the path of the input and ouput video
input_video_path = "CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4"
output_video_path = "solidWhiteRight_fixed.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the parameters of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create video writing object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 encoding format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing Video: {input_video_path} -> {output_video_path}")
print(f"FPS: {fps}, Resolution: {width}x{height}, Total Frames: {frame_count}")

# Record start time
start_time = time.time()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Read end

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    processed_frame = lane_finding_pipeline(frame)  # Processing frame
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    out.write(processed_frame)  # Write a new video
    
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

# Release resources
cap.release()
out.release()

# Record end time
end_time = time.time()
print(f"Video processing completed! Total time of {end_time - start_time:.2f} s")
```

> __If the code runs well, we will get the following video.__
> <video controls width="100%">
  <source src="solidWhiteRight_fixed.mp4" type="video/mp4">
</video>

## Shortcomings
__1. Hough Transform is fit for Straight Lines only but in reality curved lane lines exists where this will fail.__
__2. There are many roads which don't have lane markings where this will fail.__