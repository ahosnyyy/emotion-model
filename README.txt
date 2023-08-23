docker build -t emotion-model .
docker run --gpus all -it -v C:/Users/ahosny/Documents/Projects/emotion-model:/emotion-model -p 8888:8888 emotion-model

