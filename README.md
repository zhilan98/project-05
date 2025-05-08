# Project-Proposal-25
Final research code and manual

**Project features:**
The de-hazing function is achieved by processing the brightness and transmittance of the image to restore the clarity of the image. This feature uses an atmospheric scattering model with adaptive transmittance estimation to achieve the de-hazing effect.

**Requirements**
To run the image de-hazing function, you first need to install Python 3.x via the Python website https://www.python.org/downloads/release/python-3119/
Then download the libraries by “pip” to load libraries like opencv-python (cv2), numpy, scipy, Pillow (PIL), matplotlib.
You need to write the comment 

```python
pip install numpy matplotlib opencv-python scipy Pillow
```
in the terminal to download these libraries.

**How to Run**
De-hazing feature is suggest to run by Jupyter Notebook to run.
First, launch Jupyter Notebook.
Run the comment 

```python
jupyter notebook
``` 
in the terminal to open Jupyter Notebook.
Second, open the de-hazing.ipynb file.
Third, run each code unit in sequence.
Fourth, Replace the input image path and output image path as needed, e.g. /User/input_image.jpg.
Then you will get the output of the de-hazing feature.

**Tips**
Image Size: Make sure the input image is a reasonable size. If the image size is too large, it can be resized first to improve the processing speed.

Brightness Estimation: The defogging algorithm relies on image brightness estimation to ensure that the brightness distribution of the input image is reasonable in order to get a better defogging effect.

