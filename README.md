# AttSwinUNet
Official implementation code for [_Attention Swin U-Net: Cross-Contextual Attention Mechanism for Skin Lesion Segmentation_](https://arxiv.org/abs/2210.16898) paper

---
:fire: __The code will be released after the acceptance of the paper.__
![Proposed Model](./images/proposed_method_v2.png)

---
## Prepare data and pretrained model

* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"
---
## Environment and Installation

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
---
## Train and Test
- The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train 

```bash
 python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24 --mode cross_contextual_attention --spatial_attention 1
```

- Test 

```bash
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24 --mode cross_contextual_attention --spatial_attention 1
```
- For Ablation study states, skip_num can be used to determine which skip connection the proposed module will be run on, which is 3 by default, that is, it will be run in all skip connections. To remove spatial attention, just set its flag to zero. Use swin mode to remove cross contextual attention module.
---
## Updates
- October 24, 2022: Submitted to ISBI2023 [Under Review].
---
## References
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [CrossViT](https://github.com/IBM/CrossViT)
---
## Citation
```
@article{aghdam2022attention,
  title={Attention Swin U-Net: Cross-Contextual Attention Mechanism for Skin Lesion Segmentation},
  author={Aghdam, Ehsan Khodapanah and Azad, Reza and Zarvani, Maral and Merhof, Dorit},
  journal={arXiv preprint arXiv:2210.16898},
  year={2022}
}
```
