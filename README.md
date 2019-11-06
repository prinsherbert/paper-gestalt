
# Deep Paper Gestalt

## Abstract

Recent years have witnessed a significant increase in the number of paper submissions to computer vision conferences. The sheer volume of paper submissions and the insufficient number of competent reviewers cause a considerable burden for the current peer review system. In this paper, we learn a classifier to predict whether a paper should be accepted or rejected based solely on the visual appearance of the paper (i.e., the gestalt of a paper). Experimental results show that our classifier can safely reject 50% of the bad papers while wrongly reject only 0.4% of the good papers, and thus dramatically reduce the workload of the reviewers. We also provide tools for providing suggestions to authors so that they can improve the gestalt of their papers.

### Technical report: 
[[arXiv]](https://arxiv.org/pdf/1812.08775.pdf)

<img src=http://filebox.ece.vt.edu/~jbhuang/project/gestalt/this_paper.png>

## Dataset

Computer Vision Paper Gastalt dataset 
- Low resolution (680 x 440) [[link]](http://filebox.ece.vt.edu/~jbhuang/project/gestalt/CVPG_Dataset_LowRes.zip) (930 MB)
- High resolution (3400 x 2200) - Coming soon

## Pre-trained weights

Pre-trained weight for good/bad paper classifier [[link]](http://filebox.ece.vt.edu/~jbhuang/project/gestalt/PaperNet.pth) (44 MB)
- See Section 3.3 and 3.4 of the [paper](https://arxiv.org/pdf/1812.08775.pdf).

Pre-trained weight for random CVPR/ICCV paper generator [[link]](http://filebox.ece.vt.edu/~jbhuang/project/gestalt/network-snapshot-011203.pkl) (270 MB)
- See Section 4.1: What does a good paper look like?

Pre-trained weight for bad-to-good paper generator [[link]](http://filebox.ece.vt.edu/~jbhuang/project/gestalt/latest_net_G_A.pth) (44 MB)
- See Section 4.2: Learning bad-to-good paper translation

## Latent space interpolation of accepted CVPR/ICCV papers
[[link]](https://www.youtube.com/watch?v=yQLsZLf02yg)

## Instructions for running the pre-trained models on your own papers.

**These instructions are reversed engineered, and hence not from the original authors.**

**TODO:** include preprocessing starting from a pdf
**TODO:** do inference

The pre-trained model runs as `resnet18` in pytorch and require pypi's `torch` and `torchvision`. You also need to download the pre-trained weights.

    # Install torch and torchvision
    !pip3 install torch torchvision
    
    # download weights
    !wget http://filebox.ece.vt.edu/~jbhuang/project/gestalt/PaperNet.pth
 
Load the model using the downloaded `PaperNet.pth` file.

    import torch
    from torchvision.models import resnet18

    papernet = resnet18(num_classes=2)
    parameters = torch.load('PaperNet.pth')
    papernet.load_state_dict(parameters)

For accepted vs. rejected classification use `papernet.forward`

   papernet.forward # TODO: provide proper arguments

Stay tuned, not finished! 
