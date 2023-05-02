# Download the Datasets
Before coming to the download, please modify the `cache_root` in [lavis/configs/defaults.yaml](../lavis/configs/default.yaml#10).
The `cache_root` is the repo's **default data dir**.

## Download Downstream Task Data: COCO, NoCaps, VQAv2, GQA, and OK-VQA
We use the [BLIP-2](https://github.com/salesforce/LAVIS)'s 5 tasks (COCO, NoCaps, VQAv2, GQA, OK-VQA) for evaluation.
COCO, VQAv2 and OK-VQA share the same COCO images.
To download, just run the scrips under [prepare_data](./):
```bash
python download_coco.py # COCO images
python download_nocaps.py # nocaps images
python download_gqa.py # gpa images
```
The annotation files will be automatically downloaded, when you run the training or evaluation code.

## Download Pre-training Data: COCO, VG, SBU
We have already discussed how to download COCO.
To download VG and SBU (around 26G disk space), please also run:
```bash
python download_vg.py
python download_sbu.py
```
The annotation files will be automatically downloaded, when you run the training or evaluation code.


Note that it might be difficult for some researchers to download the SBU from urls.
We provide an alternative [sbu.tar.gz](https://thunlp.oss-cn-qingdao.aliyuncs.com/sbu.tar.gz) with around 0.9M data in a single compressed file.
There are also alternative links for VG ([part1](https://thunlp.oss-cn-qingdao.aliyuncs.com/images.zip), [part2](https://thunlp.oss-cn-qingdao.aliyuncs.com/images2.zip
)).
If you use alternative links, please **do not forget to put the images at your_cache_root/sbu_captions/images and your_cache_root/vg/images/**.

## Download Pre-training Data: Laion-COCO
[Laion-COCO](https://laion.ai/blog/laion-coco/) is a dataset that simulating the [BLIP](https://arxiv.org/pdf/2201.12086.pdf)'s synthetic image-caption pairs.


For Laion-COCO, we save them as several `.tar` files rather than single `.jpg` files, to avoid too much small files.
The tar files can be read using [webdataset](https://github.com/webdataset/webdataset), which is a library designed for pytorch based large-scale data loading.
I highly recommend the combination of [webdataset](https://github.com/webdataset/webdataset) (load data) and [img2dataset](https://github.com/rom1504/img2dataset) (download data) for pytorch based image-text pre-training.
To download the images and annotations, please run the following scripts:
```bash
python download_laion_coco/step1_download_laion_coco_meta.py # download the meta info of the images
python download_laion_coco/step2_download_laion_coco.sh # use img2dataest to download images and annotations into .tar based on the meta infos
```

**Please remember to modify the** `storage` **in** [lavis/configs/datasets/laion/defaults_coco.yaml](../lavis/configs/datasets/laion/defaults_coco.yaml) line 13.
We give the example about what we used in our training: `/home/zhangao/data/laion/{00000..00200}.tar`.
The `{00000..00200}.tar` means using the tar files from `00000.tar` to `00200.tar`.  


## Download MiniGPT-4 Self-Instruct Data
MiniGPT-4 create a very small set of self-instruct dataset for aligning VL-LLM with conversational scenarios.
To download, please download our preprocessed [data](https://drive.google.com/file/d/1cILZFqjvbfBEImIL2tMn1Ld8MJDAWF21/view?usp=share_link).

**Please remember to modify the** `storage` **in** [lavis/configs/defaults.yaml](../lavis/configs/datasets/minigpt4/align.yaml) line 6 to your downloaded `cc_sbu_align` dir.



