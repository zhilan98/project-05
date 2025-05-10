from flask import Flask, request, send_file, render_template
from io import BytesIO
import cv2
import numpy as np
from de_hazing import dehaze_image
import os
import io
import time
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
    saved_filename = None  # 🟡 初始化文件名变量

    file = request.files['image']
    method = request.form['method']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        if method == 'dehazing':
            processed = dehaze_image(img)
            if processed is None:
                raise ValueError("Dehazing failed: result is None")
            original_name = file.filename.rsplit('.', 1)[0]
            saved_filename = f"{original_name}_dehazed.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, saved_filename), processed)

        elif method == 'deblur':
            original_name = file.filename.rsplit('.', 1)[0]
            saved_filename = f"{original_name}_deblur.jpg"
            processed = deblur_process_image(img, output_folder=OUTPUT_FOLDER, image_name=original_name)

        elif method == 'morphing_face':
            image_path1 = request.files['image']
            image_path2 = request.files['image2']

            path1 = os.path.join(UPLOAD_FOLDER, 'face1.jpg')
            path2 = os.path.join(UPLOAD_FOLDER, 'face2.jpg')
            image_path1.save(path1)
            image_path2.save(path2)

            processed = morphing_face(path1, path2, alpha=0.5)
            saved_filename = "morphing_face_result.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, saved_filename), processed)

        elif method == 'morphing_image':
            processed = piecewise_affine_warp(img)
            original_name = file.filename.rsplit('.', 1)[0]
            saved_filename = f"{original_name}_morphimg.jpg"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, saved_filename), processed)

        # ✅ 输出保存成功的文件名（调试或记录用途）
        if saved_filename:
            print(f"图像已保存为：{saved_filename}")

        # 返回图像数据
        _, buf = cv2.imencode('.jpg', processed)
        return send_file(BytesIO(buf.tobytes()), mimetype='image/jpeg')

    except Exception as e:
        return f"图像处理出错：{str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
