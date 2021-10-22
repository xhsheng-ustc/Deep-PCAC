# Deep-PCAC: An End-to-End Deep Lossy Compression Framework for Point Cloud Attributes
This repo holds the code for the paper:
Deep-PCAC: An End-to-End Deep Lossy Compression Framework for Point Cloud Attributes https://ieeexplore.ieee.org/document/9447226

## Requirments
tensorflow-gpu 1.13.1

open3d

nvidia-docker2
##Docker
I have built a docker to aviod setting up the environment.
You can pull the docker frome dockerhub.

https://hub.docker.com/r/xhsheng/deep_pcac

docker pull xhsheng/deep_pcac

Then you need to run the nvidia-docker and mount the local directory using the following command:

sudo nvidia-docker run -v /home/Deep-PCAC-main/:/media/deep-pcac -it xhsheng/deep_pcac:latest

After entering the container, you can run the code using the following command.

## Pre-trained models
We trained four models with four different latent_points (latent_points=256,200,128,64). You can modify the latent_points to change the rate.

Pre-trained models are stored in /model/256, /model/200, /model/128, /model/64. Please modify the "checkpoint" file in these folders and change the absolute path to find the ckpt.

## Encoding
python mycodec.py compress --input="./testdata/soldier_vox10_0690.ply" --ckpt_dir='./model/256/' --latent_points=256
## Decoding
python mycodec.py decompress --input="./testdata/soldier_vox10_0690" --ckpt_dir='./model/256/' --latent_points=256

## Note
I am sorry that I have no time to sort my code becuase I am being an intern. Therefore, I can only release a .so file of my model. If you have any questions, please contact me (xhsheng@mail.ustc.edu.cn). I will try my best to solve your concerns. 

