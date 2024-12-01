# NPRDreamer: Text-guided Reference-based Non-Photorealistic Gaussian Splatting Using 2D Diffusion

<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

**This repository is the official repository of the paper, *NPRDreamer: Text-guided Reference-based Non-Photorealistic Gaussian Splatting Using 2D Diffusion*.**

[Cailin Zhuang](http://journey-zhuang.github.io/)<sup>1,5</sup>,
[Yaoqi Hu](https://hanhung.github.io/)<sup>2,5</sup>,
[Jiacheng Bao](https://angelxuanchang.github.io/)<sup>1,5</sup>
[Lan Xu](https://angelxuanchang.github.io/)<sup>1</sup>
[Ming Li](https://angelxuanchang.github.io/)<sup>3,4</sup>

<sup>1</sup>ShanghaiTech University, 
<sup>2</sup>Independent Researcher, 
<sup>3</sup>National University of Singapore, 
<sup>4</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ) (Guangming Laboratory), 
<sup>5</sup>AIGC Research

### [Project Page](https://NPRDreamer.github.io/)｜[Paper (ArXiv)](https://arxiv.org/abs/2408.03178)｜[YouTobe]()｜[Twitter thread](https://x.com/yan_xg/status/1825830023636631990)｜[Bilibili]()


![teaser](imgs/igs2gs_teaser.png)

This codebase is a fork of the original [Instruct-NeRF2NeRF](https://github.com/ayaanzhaque/instruct-nerf2nerf) repository.

# Installation

## 1. Install Nerfstudio dependencies

Instruct-GS2GS is build on Nerfstudio and therefore has the same dependency reqirements. Specfically [PyTorch](https://pytorch.org/) and [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) are required.

Follow the instructions [at this link](https://docs.nerf.studio/quickstart/installation.html) to create the environment and install dependencies. Only follow the commands up to tinycudann. After the dependencies have been installed, return here.

## 2. Installing Instruct-GS2GS

Once you have finished installing dependencies, including those for gsplat, you can install Instruct-GS2GS using the following command:
```bash
pip install git+https://github.com/cvachha/instruct-gs2gs
```

_Optional_: If you would like to work with the code directly, clone then install the repo:
```bash
git clone https://github.com/cvachha/instruct-gs2gs.git
cd instruct-gs2gs
pip install --upgrade pip setuptools
pip install -e .
```

## 3. Checking the install

The following command should include `igs2gs` as one of the options:
```bash
ns-train -h
```

# Using Instruct-GS2GS

![teaser](imgs/igs2gs_pipeline.png)

To edit a GS, you must first train a regular `splatfacto` scene using your data. To process your custom data, please refer to [this](https://docs.nerf.studio/quickstart/custom_dataset.html) documentation.

Once you have your custom data, you can train your initial GS with the following command:

```bash
ns-train splatfacto --data {PROCESSED_DATA_DIR}
```

For more details on training a GS, see [Nerfstudio documentation](https://docs.nerf.studio/quickstart/first_nerf.html).

Once you have trained your scene for 20k iterations, the checkpoints will be saved to the `outputs` directory. Copy the path to the `nerfstudio_models` folder. (Note: We noticed that training for 20k iterations rather than 30k seemed to run more reliably)

To start training for editing the GS, run the following command:

```bash
ns-train igs2gs_styleshot --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5
```

The `{PROCESSED_DATA_DIR}` must be the same path as used in training the original GS. Using the CLI commands, you can choose the prompt and the guidance scales used for InstructPix2Pix.

After the GS is trained, you can render the GS using the standard Nerfstudio workflow, found [here](https://docs.nerf.studio/quickstart/viewer_quickstart.html).

## Training Notes

***Important***
Please note that training the GS on images with resolution larger than 512 will likely cause InstructPix2Pix to throw OOM errors. Moreover, it seems InstructPix2Pix performs significantly worse on images at higher resolution. We suggest training with a resolution that is around 512 (max dimension), so add the following tag to the end of both your `splatfacto` and `igs2gs` training command: `nerfstudio-data --downscale-factor {2,4,6,8}` to the end of your `ns-train` commands. Alternatively, you can downscale your dataset yourself and update your `transforms.json` file (scale down w, h, fl_x, fl_y, cx, cy), or you can use a smaller image scale provided by Nerfstudio.

If you have multiple GPUs, training can be sped up by placing InstructPix2Pix on a separate GPU. To do so, add `--pipeline.ip2p-device cuda:{device-number}` to your training command.

| Method | Description | Memory | Quality |
| ---------------------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------- | ----------------------- |
| `igs2gs` | Full model, used in paper | ~15GB | Best |

Currently, we set the max number of iterations for `igs2gs` training to be 7.5k iteratios. Most often, the edit will look good after ~5k iterations. If you would like to train for longer, just reload your last `igs2gs` checkpoint and continue training, or change `--max-num-iterations 10000`.

## Tips

If your edit isn't working as you desire, it is likely because InstructPix2Pix struggles with your images and prompt. We recommend taking one of your training views and trying to edit it in 2D first with InstructPix2Pix, which can be done at [this](https://huggingface.co/spaces/timbrooks/instruct-pix2pix) HuggingFace space. More tips on getting a good edit can be found [here](https://github.com/timothybrooks/instruct-pix2pix#tips).

# Extending Instruct-GS2GS

### Issues
Please open Github issues for any installation/usage problems you run into. We've tried to support as broad a range of GPUs as possible, but it might be necessary to provide even more low-footprint versions. Please contribute with any changes to improve memory usage!

### Code structure
To build off Instruct-GS2GS, we provide explanations of the core code components.

`igs2gs_datamanager.py`: This file is almost identical to the `base_datamanager.py` in Nerfstudio. The main difference is that the entire dataset tensor is pre-computed in the `setup_train` method as opposed to being sampled in the `next_train` method each time.

`igs2gs_pipeline.py`: This file builds on the pipeline module in Nerfstudio. The `get_train_loss_dict` method samples images and places edited images back into the dataset.

`ip2p.py`: This file houses the InstructPix2Pix model (using the `diffusers` implementation). The `edit_image` method is where an image is denoised using the diffusion model, and a variety of helper methods are contained in this file as well.

`igs2gs.py`: We overwrite the `get_loss_dict` method to use LPIPs loss and L1Loss.

## Bibtex
If you use this work or find it helpful, please consider citing: (bibtex)
<pre id="codecell0">@misc{igs2gs,
&nbsp;author = {Vachha, Cyrus and Haque, Ayaan},
&nbsp;title = {Instruct-GS2GS: Editing 3D Gaussian Splats with Instructions},
&nbsp;year = {2024},
&nbsp;url = {https://instruct-gs2gs.github.io/}
} </pre>
## 修改过程
1. styleshotp.py 用来代替ip2p.py代表 Styleshot pipeline
2. sigs2gs_pipeline.py是原本ig2gs的滚到

question:

styleshot input 994X738 output 992X736

igs2gs    input 994x738 output 736, 992 

## Usage
### 0. 提交保存
```bash
cd ./NPRDreamer 
pip install . 
or
pip install -e . # (推荐)可编辑模式，具体见下文
```

在开发过程中，可以使用 `editable` 模式来避免每次修改代码后都需要重新安装 package。`pip` 提供了一个 `-e` 或者 `--editable` 选项，可以让你在开发过程中直接使用本地代码。

以下是步骤：

1. **进入项目目录**：
   首先，确保你在项目的根目录下。
   ```bash
   cd ./NPRDreamer
   ```

2. **安装项目为可编辑模式**：
   使用以下命令：
   ```bash
   pip install -e .
   ```

这样，`pip` 会在开发者模式下安装你的项目，任何对代码的更改都会立即体现，而不需要每次重新安装。

### 1. 数据处理：图像/视频数据 ——> Colmap点云数据
```bash
# Format: 
ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
# Example:
ns-process-data images --data data/face/images --output-dir data_processed/face
```
### 2. 3D重建：Colmap点云数据 ——> 初始3DGS
```bash
# Format: 
ns-train splatfacto --data {PROCESSED_DATA_DIR}
# Example:
ns-train splatfacto --data data_processed/face
```
### 3. 3DGS风格化
```bash
# Format: 
ns-train nprdreamer --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5
# Example:
ns-train nprdreamer --data data_processed/face --load-dir outputs/face/splatfacto/2024-xxx-xxx/nerfstudio_models --pipeline.prompt {"xxx"}
```

### 4. 再次查看
```bash
# Format:
ns-viewer --load-config {outputs/.../config.yml}
# Example:
ns-viewer --load-config outputs/face/splatfacto/2024-xxx-xxx/config.yml
```

### 5. 渲染成视频
```bash
# Format:
ns-render camera-path --load-config {outputs/.../config.yml} --camera-path-filename {PROCESSED_DATA_DIR/camera_paths/final-path.json} --output-path {outputs/.../renders.mp4}
# Example:
ns-render camera-path --load-config outputs/face/nprdreamer/2024-09-14_143901/config.yml --camera-path-filename data_processed/face/camera_paths/final-path.json --output-path outputs/face/nprdreamer/2024-09-14_143901/render.mp4
```



## debug 调试
为了在运行 `ns-train` 命令时进行调试，你可以使用以下几种方法：

## 使用 IDE 的调试器

如果你用的是一个支持 Python 的集成开发环境（IDE），比如 PyCharm 或 VSCode，可以设置一个调试配置来运行 `ns-train` 命令。

1. **PyCharm**:
   - 打开你的项目。
   - 选择 `Run > Edit Configurations...`。
   - 点击 `+` 号并选择 `Python`。
   - 设置Script Path为你运行 `ns-train` 的脚本路径，并在 Parameters 中填入你原本的命令行参数：
     ```
     nprdreamer --data data_processed/nerf_synthetic/hotdog_process --load-dir outputs/hotdog_process/splatfacto/2024-08-27_163840/nerfstudio_models --pipeline.prompt {"a hotdog"} --vis viewer+wandb
     ```
   - 设置工作目录为你的项目根目录。
   - 点击 `Apply` 和 `OK`。
   - 在代码中设置断点，然后点击 `Debug` 按钮进行调试。

2. **VSCode**:
   - 打开你的项目。
   - 创建或编辑 `.vscode/launch.json` 文件，添加以下配置：
     ```json
     {
       "version": "0.2.0",
       "configurations": [
         {
           "name": "Python: ns-train",
           "type": "python",
           "request": "launch",
           "program": "${workspaceFolder}/path_to_ns_train_script",  // 修改为你的ns-train脚本路径
           "args": [
             "nprdreamer",
             "--data", "data_processed/nerf_synthetic/hotdog_process",
             "--load-dir", "outputs/hotdog_process/splatfacto/2024-08-27_163840/nerfstudio_models",
             "--pipeline.prompt", "{\"a hotdog\"}",
             "--vis", "viewer+wandb"
           ],
           "console": "integratedTerminal"
         }
       ]
     }
     ```
   - 设置断点，然后按 `F5` 开始调试。
