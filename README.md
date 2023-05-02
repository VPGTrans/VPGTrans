# VPGTrans: Transfer Visual Prompt Generator across LLMs
[Ao Zhang](https://waxnkw.github.io/), [Hao Fei](https://haofei.vip/)\*, [Yuan Yao](https://yaoyuanthu.github.io/)\*, [Wei Ji](https://jiwei0523.github.io/), [Li Li](https://vpgtrans.github.io/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/) and [Tat-Seng Chua](https://www.chuatatseng.com/). *Correspondence 

**Sea-NExT Joint Lab, National University of Singapore**


**Technology, Tsinghua University**


<a href='https://vpgtrans.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://vpgtrans.github.io/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://ee569fe29733644a33.gradio.live'><img src='https://img.shields.io/badge/Demo-Page-blue'></a> 

## What's New: 🎉 
- [x] 2023.05.01 Rlease the code.

## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [VL-Vicuna Demo](#vl-vicuna-demo)
  - [Evaluation](#evaluation)
  - [Training](#training)
    - [VL-LLaMA Training](#vl-llama-training)
    - [VL-Vicuna Training](#vl-vicuna-training)
  -  [Acknowledgement](#acknowledgement)
  - [License](#license)

## Introduction
We develop a VPGTrans framework for transferring visual prompt generator (VPG) across LLMs to build VL-LLMs.
The GPU hours can be reduced over 10 times and the training data can be reduced to around 10%, like:
[![demo](figs/cost.png)](https://github.com/VPGTrans/VPGTrans)

Our framework consists of two stages:
[![demo](figs/VPGTrans.png)](https://github.com/VPGTrans/VPGTrans)

The most exciting thing is that VPGTrans enable us to **customize new VL-LLMs with newly released LLMs**.
For example, we release a **VL-LLaMA and VL-Vicuna**.
The performance of VL-LLaMA looks like:
<div>
  <a><img src="./figs/performance.png" width = "500" height = "200" alt="" align=left /></a>
  <br><br><br><br><br><br><br><br><br>
</div>


Checkout the image for our demo of VL-Vicuna:
[![demo](figs/online_demo.png)](https://ee569fe29733644a33.gradio.live)  


## Installation
**1. Prepare the code**

```bash
git clone https://github.com/VPGTrans/VPGTrans.git
cd VPGTrans
pip install -r requirements.txt
pip install -e .
```


## VL-Vicuna Demo

**1. Prepare the pretrained Vicuna weights**  
To run VL-Vicuna locally, you need to first prepare the  
The current version of VL-Vicuna is built on the v0 versoin of Vicuna-7B.
Please refer to the instruction [here](PrepareVicuna.md) 
to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```
 
**2. Run the code**  
Please modify the `llama_model`
[lavis/projects/blip2/demo/vl_vicuna_demo.yaml](lavis/projects/blip2/demo/vl_vicuna_demo.yaml)
line 12 to your vicuna checkpoint.   

Then, run:
```bash
python webui_demo.py --cfg-path lavis/projects/blip2/demo/vl_vicuna_demo.yaml  --gpu-id 0
```

Note the checkpoint will be automatically downloaded.

## Evaluation

**Prepare the data**  
Please first refer to [here](prepare_data/README.md) to prepare the dataset you want to evaluate.
We use COCO caption, NoCaps, VQAv2, GQA, and OK-VQA in our paper.

### VL-LLaMA Evaluation
**1. Prepare the pretrained LLaMA weights**  
LLaMA checkpoint is the start of Vicuna.
Please also refer to the instruction [here](PrepareVicuna.md) 
to prepare the LLaMA weights.

**2. Run the code**  
Please run:
```bash
bash run_scripts/blip2/scale_up_eval/eval_vqa_llama.sh /path/to/llama_7b_dir/ # zero-shot vqav2 eval
bash run_scripts/blip2/scale_up_eval/eval_gqa_llama.sh /path/to/llama_7b_dir/ # zero-shot gqa eval
bash run_scripts/blip2/scale_up_eval/eval_okvqa_llama.sh /path/to/llama_7b_dir/ # zero-shot okvqa eval
```

[//]: # (### BLIP-2 OPT 6.7B Evaluation)
[//]: # (### BLIP2 FlanT5-XXL Evaluation)

## Training
The stage-1 pre-training requires COCO caption and SBU.
The stage-2 also requires VG caption and Laion-COCO.
Please refer to [here](prepare_data/README.md) for data downloading.  

If you want to remove some datasets in the training.
Please comment the items in the config file, like:
```yaml
# lavis/projects/blip2/train/llama_vpgtrans_step1_proj_warmup.yaml
datasets:
  coco_caption:
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224
        eval:
          name: "blip_image_eval"
          image_size: 224
    text_processor:
        train:
          name: "blip_caption"
        eval:
          name: "blip_caption"
#  the SBU will not be used in the pre-training
#  sbu_caption:
#    vis_processor:
#        train:
#          name: "blip2_image_train"
#          image_size: 224
#    text_processor:
#        train:
#          name: "blip_caption"
```

### VL-LLaMA Training

**1. Stage-1 Projector Warm-up**  
First, you need to download the [BLIP2 OPT-6.7B checkpoint](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth).

The 1st thing is to initialize the projector with word convertor. (Optional)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/linear_proj/train_linear_proj_opt_and_llama.py \
  facebook/opt-6.7b \
  /path/to/llama_7b_dir/ \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/output_dir
# example:
# CUDA_VISIBLE_DEVICES=0 python tools/linear_proj/train_linear_proj_opt_and_llama.py \
#   facebook/opt-6.7b \
#   ./llama-7b/ \
#   ./blip2_pretrained_opt6.7b.pth  \
#   ./lavis/output/proj_init
```  
Then, run the projector warm-up:
```bash
bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step1_proj_warmup.sh \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/projector_init_weight \
  /path/to/llama_7b_dir/
 # example:
# bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step1_proj_warmup.sh \
#  ./blip2_pretrained_opt6.7b.pth \
#  ./lavis/output/proj_init/xxx.pth \
#  ./llama-7b/
```

**2. Stage-2 Direct Fine-tuning**  
Please run:
```bash
bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step2_direct_finetune.sh \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/stage1_proj_warmup_checkpoint \
  /path/to/llama_7b_dir/
 # example:
# bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step2_direct_finetune.sh \
#  ./blip2_pretrained_opt6.7b.pth \
#  ./lavis/output/vl-llama_stage1/checkpoint_0.pth \
#  ./llama-7b/
```

### VL-Vicuna Training
Vicuna is the instruction-tuning version of LLaMA.
Most of the scripts are similar with training LLaMA.

**1. Stage-1 Projector Warm-up**  
First, you need to download the [BLIP2 OPT-6.7B checkpoint](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth).

The 1st thing is to initialize the projector with word convertor. (Optional)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/linear_proj/train_linear_proj_opt_and_llama.py \
  facebook/opt-6.7b \
  /path/to/vicuna_7b_dir/ \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/output_dir
# example:
# CUDA_VISIBLE_DEVICES=0 python tools/linear_proj/train_linear_proj_opt_and_llama.py \
#   facebook/opt-6.7b \
#   ./vicuna-7b/ \
#   ./blip2_pretrained_opt6.7b.pth  \
#   ./lavis/output/proj_init
```  
Then, run the projector warm-up:
```bash
bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step1_proj_warmup.sh \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/projector_init_weight \
  /path/to/vicuna_7b_dir/
 # example:
# bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step1_proj_warmup.sh \
#  ./blip2_pretrained_opt6.7b.pth \
#  ./lavis/output/proj_init/xxx.pth \
#  ./vicuna-7b/
```

**2. Stage-2 Direct Fine-tuning**  
Please run:
```bash
bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step2_direct_finetune.sh \
  /path/to/blip2_opt6.7b_ckpt \
  /path/to/stage1_proj_warmup_checkpoint \
  /path/to/vicuna_7b_dir/
 # example:
# bash run_scripts/blip2/scale_up_train/llama_vpgtrans_step2_direct_finetune.sh \
#  ./blip2_pretrained_opt6.7b.pth \
#  ./lavis/output/vl-vicuna_stage1/checkpoint_0.pth \
#  ./vicuna-7b/
```

**3. Stage-3 Visual Instruction Tuning**  
To align with conversation scenario, we conduct a short tuning using MiniGPT-4's self-instruct data (around 3,000 images).
Please refer to [instruction](prepare_data/README.md) for downloading it.
Please run:
```bash
bash run_scripts/blip2/scale_up_train/vicuna_vpgtrans_step3_self_instruct.sh \
  /path/to/stage2_direct_tuning_checkpoint \
  /path/to/vicuna_7b_dir/
 # example:
# bash run_scripts/blip2/scale_up_train/vicuna_vpgtrans_step3_self_instruct.sh \
#  ./lavis/output/vl-vicuna_stage2/checkpoint_50000.pth \
#  ./vicuna-7b/
```


## Acknowledgement

+ [Lavis](https://github.com/salesforce/LAVIS) This repository is built upon Lavis!
+ [Vicuna](https://github.com/lm-sys/FastChat) We build the model based.
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) The web UI and part of README are based on MiniGPT-4.


If you're using VPGTrans in your research or applications, please cite using this BibTeX:
```bibtex
@article{2023vpgtrans,
  author      = {Ao Zhang and Hao Fei and Yuan Yao and Wei Ji and Li Li and Zhiyuan Liu and Tat-Seng Chua},
  title       = {Transfer Visual Prompt Generator across LLMs},
  year        = {2023},
}
```

## License
[BSD 3-Clause License](LICENSE.txt)
