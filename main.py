import os
import io
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from PIL import Image
from io import BytesIO

# Import functions from your image processing modules
from morphing_face import morphing_face
from morphing_image import piecewise_affine_warp
from de_hazing import dehaze_image as dehaze_img
from deblur import deblur_process_image, resize_image

app = Flask(__name__)

# Create folders for uploads and output
OUTPUT_FOLDER = 'output_images'
UPLOAD_FOLDER = 'uploads'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process_image():
    method = request.form['method']
    
    # Different image processing methods require different handling
    if method == 'dehazing':
        # Handle dehazing process
        if 'image' not in request.files:
            return "No image uploaded", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No image selected", 400
            
        # Save the uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, 'input_dehazing.jpg')
        file.save(input_path)
        
        # Process the image
        try:
            result_img = dehaze_img(input_path)
            output_path = os.path.join(OUTPUT_FOLDER, 'result_dehazing.jpg')
            result_img.save(output_path)
            
            # Return processed image
            img_io = BytesIO()
            result_img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        except Exception as e:
            return f"Image processing failed: {str(e)}", 500
    
    elif method == 'deblur':
        # Handle deblurring process
        if 'image' not in request.files:
            return "No image uploaded", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No image selected", 400
            
        # Save the uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, 'input_deblur.jpg')
        file.save(input_path)
        
        # Process the image
        try:
            # Read image with OpenCV
            img = cv2.imread(input_path)
            if img is None:
                return "Failed to read image", 400
                
            # Resize if needed (optional)
            img = resize_image(img, width=800)  # Resize to reasonable width
            
            # Process with deblur function
            output_img = deblur_process_image(img, output_folder=OUTPUT_FOLDER, image_name='result_deblur')
            
            # Convert and return
            _, buffer = cv2.imencode('.jpg', output_img)
            img_io = BytesIO(buffer)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        except Exception as e:
            return f"Deblurring failed: {str(e)}", 500
    
    elif method == 'morphing_face':
        # Handle face morphing process - requires two images
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Two face images are required", 400
            
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        if file1.filename == '' or file2.filename == '':
            return "Both images must be selected", 400
            
        # Save the uploaded images
        path1 = os.path.join(UPLOAD_FOLDER, 'face1.jpg')
        path2 = os.path.join(UPLOAD_FOLDER, 'face2.jpg')
        file1.save(path1)
        file2.save(path2)
        
        # Get alpha parameter (default to 0.5 if not provided)
        alpha = float(request.form.get('alpha', 0.5))
        
        # Process the images
        try:
            result = morphing_face(path1, path2, alpha=alpha)
            
            # Convert result to image and return
            output_path = os.path.join(OUTPUT_FOLDER, 'result_morphface.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            # Return as file
            img_pil = Image.fromarray(result)
            img_bytes = BytesIO()
            img_pil.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/jpeg')
        except Exception as e:
            return f"Face morphing failed: {str(e)}", 500
    
    elif method == 'morphing_image':
        # Handle image morphing/warping process
        if 'image' not in request.files:
            return "No image uploaded", 400
            
        file = request.files['image']
        if file.filename == '':
            return "No image selected", 400
            
        # Save the uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, 'input_morphimg.jpg')
        file.save(input_path)
        
        # Get parameters with defaults
        mode = request.form.get('mode', 'sine')
        strength = int(request.form.get('strength', 10))
        grid_size = int(request.form.get('grid_size', 8))
        
        # Process the image
        try:
            # The function returns a numpy array with values in [0,1]
            warped_img = piecewise_affine_warp(
                input_path, 
                mode=mode, 
                strength=strength, 
                grid_size=grid_size,
                show=False  # Don't show the image, just return it
            )
            
            # Convert float array to uint8 for saving
            # Values are in range [0,1] so multiply by 255
            warped_img_uint8 = (warped_img * 255).astype(np.uint8)
            
            # Save the image - No color conversion needed since we're using PIL
            output_path = os.path.join(OUTPUT_FOLDER, 'result_morphimg.jpg')
            
            # Use PIL to save the image which maintains correct color space
            pil_img = Image.fromarray(warped_img_uint8)
            pil_img.save(output_path)
            
            # Return as file
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/jpeg')
        except Exception as e:
            return f"Image morphing failed: {str(e)}", 500
    
    else:
        return f"Unknown processing method: {method}", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)