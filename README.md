Project

# Project features:

- Face Morphing: using facial landmarks extracted with Mediapipe
- General Image Morphing: using Piecewise Affine Transform from scikit-image

---

# Requirements

To run morphing features(face-morphing), you will need to download Python 3.11.9:

- Go to https://www.python.org/downloads/release/python-3119/ and download Python 3.11

You will need to install the following libraries: opencv-python (cv2),numpy,mediapipe,matplotlib,scipy

- Install Python libraries using pip:

Open terminal and run the following command:

```
pip install numpy matplotlib opencv-python mediapipe scikit-image
```

# How to Run

The Morphing feature is suggested to launch via Jupyter notebook:

1. Launch Jupyter Notebook

Open terminal and run the following command:

```
jupyter notebook
```

2. Open either `morphing_face.ipynb` or `morphing_image.ipynb`
3. Run each cell in order
4. Replace input images with your own as needed:
   `face1.jpg`, `face2.jpg` in `morphing_face.ipynb` or
   `example_img1.png` in `morphing_image.ipynb`

# Tips

- Ensure images are of similar size for better morphing results
- Use frontal, well-lit face images for optimal facial landmark detection
- RGB format is required for matplotlib; use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` if using OpenCV
