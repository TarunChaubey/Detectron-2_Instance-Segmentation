from flask import Flask, request,jsonify,render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import os

import numpy as np


import logging

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


os.makedirs('logs',exist_ok=True)
logging.basicConfig(filename='logs/logfile.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

logging.info('API Statred')
logging.info('module import successfully')

upload_dir = "./APIPredictedImages/"
os.makedirs(upload_dir,exist_ok=True)
logging.info('now working on flask app')


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# cfg.MODEL.WEIGHTS = "./OutputModels/model_final.pth"
cfg.MODEL.WEIGHTS = "./OutputModelsSegmentation/model_final.pth"
cfg.MODEL.DEVICE = "cpu"



classes = ["benign","malignant"]
dataset_name = "val"
MetadataCatalog.get(dataset_name).set(things_classes = classes)
register_coco_instances("val",{},"Data/val.json","Data/val")
val_metadata = MetadataCatalog.get('val')

predictor = DefaultPredictor(cfg)

app = Flask(__name__)

# CORS(app)

def CalculateArea(mask_tensor):
  # Move tensor to CPU if necessary
  if mask_tensor.device.type != "cpu":
      mask_tensor = mask_tensor.cpu()

  # Convert mask tensor to numpy array
  mask_array = mask_tensor.numpy()

  # Compute pixel size (assuming square pixels)
  # image_width = outputs["instances"].image_width
  # image_height = outputs["instances"].image_height

  image_height = 300
  image_width = 300
  pixel_size = (image_height*image_width) / np.prod(mask_array.shape[1:])
  # Compute area of each mask
  mask_areas = []
  for mask in mask_array:
      num_pixels = np.count_nonzero(mask)
      mask_area = num_pixels * pixel_size
      mask_areas.append(mask_area)

  return mask_areas
  # print("Mask areas:", mask_areas)

logging.info('index url called')
@app.route("/")
def index():
  return render_template('upload.html')


logging.info('predict url called')
@app.route("/predict",methods=['POST'])
def predict():
  file = request.files['image']
  file.save(upload_dir+file.filename)

  logging.info(f"image upload at {upload_dir+file.filename}")
  img = Image.open(upload_dir+file.filename)
  imgarray = np.array(img)[:, :, ::-1]
  res = predictor(imgarray)

  logging.info(f"working to predict {file.filename}")
  pred_classes = res["instances"].to("cpu").pred_classes.tolist()[0]
  pred_scores = res["instances"].to("cpu").scores.tolist()[0]
  pred_boxes = res["instances"].to("cpu").pred_boxes
  pred_masks = res["instances"].to("cpu").pred_masks.tolist()[0]

  image_area = CalculateArea(res["instances"].to("cpu").pred_masks)

  v = Visualizer(imgarray, metadata=val_metadata, scale=0.8)
  v = v.draw_instance_predictions(res["instances"].to("cpu"))
  cv2.imwrite(f"{upload_dir}{file.filename}",v.get_image()[:, :, ::-1])

  # Convert the predicted class labels and scores to a dictionary
  output = {
      "Actual File":file.filename,
      "Predicted": pred_classes,
      "scores": pred_scores,
      "area":image_area,
      "pred_masks":pred_masks

  }

  print(output)

  return jsonify(output)


# running port http://127.0.0.1:5000/
if __name__ == "__main__":
  app.run(debug=True) 