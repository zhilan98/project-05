from flask import Flask, request, send_file, render_template
from io import BytesIO
import cv2
import numpy as np
from de_hazing import dehaze_image
import os
import io
from PIL import Image
from morphing_face import morphing_face
from morphing_image import piecewise_affine_warp
from deblur import deblur_process_image

app = Flask(__name__)
OUTPUT_FOLDER = 'output_images'
UPLOAD_FOLDER = 'img'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    method = request.form['method']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if method == 'dehazing':
        processed = dehaze_image(img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'uploaded_dehazed.jpg'), processed)

    elif method == 'deblur':
        processed = deblur_process_image(img, output_folder=OUTPUT_FOLDER, image_name='uploaded')

    elif method == 'morphing_face':
        image_path1 = request.files['image'] 
        image_path2 = request.files['image2'] 
        
        # 保存上传的图片
        path1 = os.path.join(UPLOAD_FOLDER, 'face1.jpg')
        path2 = os.path.join(UPLOAD_FOLDER, 'face2.jpg')

        image_path1.save(path1)
        image_path2.save(path2)
        
        try:
            result = morphing_face(path1, path2, alpha=0.5)
        except Exception as e:
            return f"处理图像失败：{str(e)}", 500

        # 转换 numpy array 为图像字节流返回
        img_pil = Image.fromarray(result)
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype='image/jpeg')
    
        # processed = morphing_face(path1, path2, alpha=0.5)
        # cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'uploaded_morphface.jpg'), processed)

    elif method == 'morphing_image':
        processed = piecewise_affine_warp(img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'uploaded_morphimg.jpg'), processed)

    # Return image as response
    _, buf = cv2.imencode('.jpg', processed)
    return send_file(BytesIO(buf.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
