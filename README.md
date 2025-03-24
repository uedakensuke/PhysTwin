# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

<span class="author-block">
<a target="_blank" href="https://jianghanxiao.github.io/">Hanxiao Jiang</a><sup>1,2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://haoyuhsu.github.io/">Hao-Yu Hsu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.linkedin.com/in/hnyu/">Hsin-Ni Yu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign,</span>

### [Website](https://jianghanxiao.github.io/phystwin-web/) | [Paper](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf) | [Arxiv]()

### Overview
This repository contains the official implementation of the **PhysTwin** framework.

![TEASER](./assets/teaser.png)

### Setup
```
# Here we use cuda-12.1
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install the packages
bash ./env_install/env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

### Download the PhysTwin Data
Download the original data, processed data and the results
```
# Download the source data


# download pretrained gaussians from:
# https://drive.google.com/file/d/1ffNOGlPPJ20XKjxfR-sSl6K6zodpaRHK/view?usp=sharing
# and place it under the gaussian_splatting folder
```

### Play with the Interactive Playground
Use the previously constructed PhysTwin to explore the interactive playground. Users can interact with the pre-built PhysTwin using keyboard. The following sections will provide a detailed guide on constructing the PhysTwin from the original data.

TODO: add a video here

```


```