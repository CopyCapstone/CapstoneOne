# Metal Color and Glossiness Evaluation System
This repository contains a Computer Vision-based Web Application designed for the continuous monitoring and analysis of metal surfaces. Developed as a Computer Engineering Capstone Project at Chulalongkorn University, the system provides a low-cost, accessible alternative to industrial spectrophotometers for evaluating color shifts and surface glossiness over time.

## Key Features
- Automated Object Detection: Utilizes YOLOv8 to automatically locate and isolate target objects (such as copper or brass samples) within a video frame.  
- Chromatic Adaptation: Implements a Chromatic Adaptation Transform (CAT) to normalize captured color data to the standard D65 illuminant, ensuring consistency across varying lighting conditions.  
- Color & Gloss Segmentation: Employs K-means clustering to bifurcate pixels into diffuse (intrinsic color) and specular (gloss) components.  
- CIELAB Color Estimation: Converts RGB data to the device-independent CIELAB *L*\**a*\**b*\* color space using a Neural Network regression model for high accuracy.  
- Predictive Forecasting: Features a Polynomial Regression model to forecast future color degradation and tarnish trends based on historical time-series data.

## Installation
Follow these steps to set up the environment and run the application locally.
### 1. Clone the Repository
First, clone the repository and navigate into the project directory:
```
git clone https://github.com/CopyCapstone/CapstoneOne.git
cd CapstoneOne
```
### 2. Choose Installation Option
Option 1: Local Environment (Requires Python 3.11)

This method uses a Python virtual environment to manage dependencies locally.
1. Create a virtual environment:
   ```
   python -m venv .venv
   ```
2. Activate the environment:
   
   Windows:
   ```
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```
   source .venv/bin/activate
   ```
4. Install required libraries:
   ```
   pip install -r requirements.txt
   ```
5. Launch the Web App:
   ```
   streamlit run main.py
   ```
Option 2: Docker (Recommended for Portability)

This project supports deployment via Docker to prevent dependency conflicts and ensure environment consistency.  

1. Build the Docker image:
   ```
   docker build -t my-streamlit-app .
   ```
2. Run the container:
   
   Map the default Streamlit port (8501) to your local machine:
   ```
   docker run -p 8501:8501 my-streamlit-app
   ```
