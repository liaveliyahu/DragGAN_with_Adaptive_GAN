<p align="center">

  <h1 align="center">Drag Your GAN with Addaptive GAN</h1>
  <h2 align="center">All rights reserved to the original Drag Your GAN work</h2>
  <p align="center">
    <a href="https://arxiv.org/abs/2305.10973/"><strong>Paper link</strong></a>
    ·
    <a href="https://github.com/XingangPan/DragGAN/"><strong>Code link</strong></a>
    ·
    <a href="https://vcai.mpi-inf.mpg.de/projects/DragGAN/"><strong>Project link</strong></a>
  </p>
  <p align="center">
    <a href="https://xingangpan.github.io/"><strong>Xingang Pan</strong></a>
    ·
    <a href="https://ayushtewari.com/"><strong>Ayush Tewari</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~tleimkue/"><strong>Thomas Leimkühler</strong></a>
    ·
    <a href="https://lingjie0206.github.io/"><strong>Lingjie Liu</strong></a>
    ·
    <a href="https://www.meka.page/"><strong>Abhimitra Meka</strong></a>
    ·
    <a href="http://www.mpi-inf.mpg.de/~theobalt/"><strong>Christian Theobalt</strong></a>
  </p>
  <h3 align="center">SIGGRAPH 2023 Conference Proceedings</h3>
  <div align="center">
    <img src="DragGAN.gif", width="600">
  </div>

  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://twitter.com/XingangP"><img alt='Twitter' src="https://img.shields.io/twitter/follow/XingangP?label=%40XingangP"></a>
    <a href="https://arxiv.org/abs/2305.10973">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/DragGAN/'>
      <img src='https://img.shields.io/badge/DragGAN-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://colab.research.google.com/github/liaveliyahu/DragGAN_with_Adaptive_GAN/blob/main/DragGan_AdaptiveGan_collab_gradio.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </p>
</p>

## Problem

When using the tool, can notice that the object may loss its essential characteristics during the iterations. 
For example, when rotating a car, it can end up with a different model than the original car model in the first iteration. 
Another example, when moving a person arm, it caused her hair to turn blonde.

<div align="center">
  <img src="rotatecar.gif", width="400">
  <img src="woman.png", width="400">
</div>

## Solution

The soultion found for this problem is to use method based on Adaptive GAN (by Shady Abu Hussein). With this method, the generator trained on the input image before running the manipulation. After a few iterations of training the happens only once per image, can work with the tool as usual.

<a href="https://arxiv.org/pdf/1906.05284.pdf/"><strong>Paper link</strong></a>
·
<a href="https://github.com/shadyabh/IAGAN/"><strong>Code link</strong></a>
·
    
<div align="center">
  <img src="flow.png", width="400">
</div>

## Results

With the results we can see a minor improvement of the problem, and see that the main characteristics were preserved a little better. Adjusting the weights of the generator may potentially induce artifacts or distortions in the resulting images.
Need to fine tuning the training hyperparameters per class.

<div align="center">
  <img src="results.png", width="400">
</div>

## How to Run

<p align="center">
  <br>
    <a href="https://colab.research.google.com/github/liaveliyahu/DragGAN_with_Adaptive_GAN/blob/main/DragGan_AdaptiveGan_collab_gradio.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

* Click on the Google Colab Link.
* Connect to GPU (T4).
* Run all lines.
* Open the Gradio Application link.
* Start to play with the tool :).
* You can change number of iterations or/and learning rate at "viz/renderer.py" lines 269 & 272.
