import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle
import subprocess
import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return print('Hello lauser')#render_template('index.html')

# @app.route('/query-example')
# def query_example():
#     return 'Todo...'
# @app.route('/predict')
# def predict():
#     image_i = request.args.get('image_input') #if key doesn't exist, returns None
#     print(image_output)
#     # image_o = request.args.get('image_output')
#     # print(image_output)
#     # subprocess.run(f'python3 ../predict.py -c config.json -i {image_input} -o {image_output}')
#     # print('''<h1>The image {} processing and output to {} </h1>'''.format( image_i, image_o))
#     return '''<h1>The image {} processing and output to /h1>'''.format(image_i)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        config_path  = request.form.get('config')
        image_path   = request.form.get('image_path')
        output_path  = request.form.get('image_output')
        # import pdb; pdb.set_trace()
        with open(config_path) as config_buffer:    
            config = json.load(config_buffer)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

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
        
        # if not os.path.exists(image_path):
        #     import pdb; pdb.set_trace()
        image = cv2.imread(image_path)
                
        print(image_path)

                # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

                # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
        
                # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))  
    return '''<form method="POST">
                  config: <input type="text" name="config"><br>
                  image_input: <input type="text" name="image_path"><br>
                  image_output: <input type="text" name="image_output"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''    

# @app.route('/form', methods=['GET', 'POST']) #allow both GET and POST requests
# def form_example():
#     if request.method == 'POST':  #this block is only entered when the form is submitted
#         # config = request.form.get('config')
#         image_input = request.form.get('image_input')
#         image_output = request.form['image_output']
#         subprocess.run(f'python3 predict.py -c config.json -i {image_input} -o {image_output}')
#         return '''<h1>The image_input value is: {}</h1>
#                   <h1>The image_output value is: {}</h1>'''.format(image_input, image_output)

#     return '''<form method="POST">
#                   image_input: <input type="text" name="image_input"><br>
#                   image_output: <input type="text" name="image_output"><br>
#                   <input type="submit" value="Submit"><br>
#               </form>'''

if __name__ == "__main__":
    app.run(debug=True, port=5000)