# Deep-PCAC: An End-to-End Deep Lossy Compression Framework for Point Cloud Attributes
This repo holds the code for the paper:
Deep-PCAC: An End-to-End Deep Lossy Compression Framework for Point Cloud Attributes https://ieeexplore.ieee.org/document/9447226

## Requirments
Tensorflow-gpu 1.13.1
open3d

##Pre-trained models are stored in ./models
## We trained four models with four different latent_points (latent_points=256,200,128,64). You can modify the latent_points to change the rate.
## Encoding
python mycodec.py compress --input="./testdata/soldier_vox10_0690.ply" --ckpt_dir='./model/256/' --latent_points=256
## Decoding
python mycodec.py decompress --input="./testdata/soldier_vox10_0690" --ckpt_dir='./model/256/' --latent_points=256

