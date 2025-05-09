from flask import Flask, request, send_file
from io import BytesIO
import cv2
import numpy as np
from dehazing import dehazing
from morphing_face import morphing_face
from morphing_image import morphing_image
from deblur import deblur

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    method = request.form['method']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if method == 'dehazing':
        processed = dehazing(img)
    elif method == 'deblur':
        processed = deblur(img)
    elif method == 'morphing_face':
        processed = morphing_face(img)
    elif method == 'morphing_image':
        processed = morphing_image(img)

    _, buf = cv2.imencode('.jpg', processed)
    return send_file(BytesIO(buf.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
