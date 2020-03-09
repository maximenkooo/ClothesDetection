import numpy as np
import flask
from flask import Flask, request, redirect, url_for, send_from_directory, send_file, make_response
import os
import json
from keras.models import load_model
from PIL import Image
import io
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes

app = Flask(__name__)
CONFIG_FILE = '/home/helga/project/ClothesDetection/config.json'
app.config['CONFIG_FILE'] = CONFIG_FILE

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        imagefile = flask.request.files.get('file', '')
        imagefile_array = np.array(Image.open(imagefile))
        return predict(image=imagefile_array)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

def predict(image):
    with open(app.config['CONFIG_FILE']) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################

            # predict the bounding boxes
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
    
            # output the image with bounding boxes to file
    img = Image.fromarray(image)

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG', as_attachment=True, attachment_filename='result.png')   

if __name__ == "__main__":
    app.run(debug=True, port=5000)