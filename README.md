# YOLOv5 Emotion Classification and Clothing Detection

This repository contains code to train and deploy YOLOv5 models for emotion classification and clothing detection.

## Clone YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
```
## Folder Structure

yolov5-vision/
├── datasets/
│   ├── fer2013
│   ├── deepfashion2
│   └── deepfashion2.yaml
├── utils/
│   └── ...
├── weights/
│   └── ...
├── yolov5/
│   └── ...
├── Dockerfile
├── README.md
└── requirements.txt

## Build Docker Image
```bash
docker build -t yolov5 .
```

## Run Docker Container
```bash
docker run --gpus all -it --ipc=host --name yolov5 -v <path-to-repo>/yolov5-vision:/app yolov5
cd yolov5
```

## Emotion Classification
```bash
python classify/train.py --model yolov5s-cls.pt --data ../datasets/fer2013 --epochs 150 --imgsz 224 --batch 64 --workers 2 --name fer2013
```

### Export to ONNX and TensorFlow Lite
```bash
python export.py --weights runs/train-cls/fer2013/weights/best.pt --imgsz 224 --include onnx
onnx-tf convert -i runs/train-cls/fer2013/weights/best.onnx -o runs/train-cls/fer2013/weights/saved_model
python ../utils/to_tflite.py

python export.py --weights runs/train-cls/fer2013/weights/best.pt --imgsz 224 --optimize --include torchscript
```

## Clothing Detection
```bash
python train.py --weights yolov5s.pt --data ../datasets/fashionpedia.yaml --epochs 100 --imgsz 320 --batch-size 32 --workers 4 --name fashionpedia 
```

### Export to TensorFlow Lite and TorchScript
```bash
python export.py --weights runs/train/fashionpedia/weights/best.pt --imgsz 320 --include tflite #--nms --agnostic-nms
python export.py --weights runs/train/fashionpedia/weights/best.pt --imgsz 320 --include torchscript --optimize
```

**Note:** Make sure to replace file paths and dataset names as per your specific setup.