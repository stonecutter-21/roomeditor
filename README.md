# [NeurIPS 2025] RoomEditor: High-Fidelity Furniture Synthesis with Parameter-Sharing U-Net

![alt text](assets/teaser.png)

## 🔥 Updates
[2025/12/15] Release the RoomEditor model, the RoomBench dataset, and the evaluation code.
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Download Checkpoints 

Download SD-1.5-inpainting checkpoint: 

* You could download it from HuggingFace [stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

Download Roomeditor: 

* You could download the model weights and test images from  [here](https://share.multcloud.link/share/74f20132-4269-41d0-8355-0c39106de6b0)



## Evaluation

Run inference with 

```eval
bash run.sh 
```



## Gradio Demo

You can run this script:

```setup
python run_gradio_demo.py
```

