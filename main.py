# uvicorn FastAPI:app

from fastapi.responses import HTMLResponse, JSONResponse,FileResponse
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import uvicorn
import cv2
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
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

app = FastAPI()
templates = Jinja2Templates(directory="templates")
classes = ["benign","malignant"]

upload_dir = "./uploadeImages/"
predict_dir = "./APIPredictedImages/"
os.makedirs(upload_dir,exist_ok=True)
os.makedirs(predict_dir,exist_ok=True)

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



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict")
# async def predict(image: UploadFile = File(...)):
async def predict(request: Request,image: UploadFile = File(...)):
    contents = await image.read()
    file_path = os.path.join(upload_dir, image.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    img = cv2.imread(file_path)
    # perform prediction on the image
    # print(file_path)

    img = Image.open(file_path)
    imgarray = np.array(img)[:, :, ::-1]
    res = predictor(imgarray)


    v = Visualizer(img, metadata=val_metadata, scale=0.8)
    v = v.draw_instance_predictions(res["instances"].to("cpu"))
    cv2.imwrite(f"{upload_dir}{image.filename}",v.get_image()[:, :, ::-1])

    print(res["instances"].to("cpu"))
    
    pred_classes = res["instances"].to("cpu").pred_classes
    pred_scores = res["instances"].to("cpu").scores
    pred_boxes = res["instances"].to("cpu").pred_boxes
    pred_masks = res["instances"].to("cpu").pred_masks

    image_area = CalculateArea(pred_masks)

  # Convert the predicted class labels and scores to a dictionary
    output = {
        "Predicted": pred_classes,
        "scores": pred_scores,
        "area":image_area,
        "pred_boxes":pred_boxes,
        "pred_masks":pred_masks
        }
        
    return templates.TemplateResponse(
        "result.html",
        {"request": request,"text": output,"filename":image.filename}
        )

if __name__ == "__main__":
    uvicorn.run(app, debug=True)