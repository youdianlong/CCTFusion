
# CCTFusion
## Description
This package includes the python code of the CCTFusion(CCTFusion: A Cyclic-cross Transformer for Multimodal Image
Fusion) which focus on the problem of multimodal image fusion.
## Visible and Infrared Image Fusion
### Train
Download the training dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_cctfusion.py --opt options/swinir/train_cctfusion_vif.json  --dist True

### Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

    python test_cctfusion.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --iter_number=10000 --dataset=MSRS --A_dir=IR  --B_dir=VI_Y

## Medical Image Fusion
### Train
Download the training dataset from [**Harvard medical dataset**](http://www.med.harvard.edu/AANLIB/home.html), and put it in **./Dataset/trainsets/PET-MRI/** or **./Dataset/trainsets/CT-MRI/**. 

    python -m torch.distributed.launch --nproc_per_node=3 --master_port=1234 main_train_cctfusion.py --opt options/swinir/train_cctfusion_med.json  --dist True
    
### Test
Download the training dataset from [**Harvard medical dataset**](http://matthewalunbrown.com/nirscene/nirscene.html), and put it in **./Dataset/testsets/PET-MRI/** or **./Dataset/testsets/CT-MRI/**. 

    python test_cctfusion.py --model_path=./Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/  --iter_number=10000 --dataset=CT-MRI--A_dir=MRI --B_dir=PET_Y
**or** 

    python test_cctfusion.py --model_path=./Model/Medical_Fusion-CT-MRI/Medical_Fusion/models/ --iter_number=10000 --dataset=CT-MRI--A_dir=MRI --B_dir=CT


## Recommended Environment

 - [x] torch 1.11.0
 - [x] torchvision 0.12.0
 - [x] tensorboard  2.7.0
 - [x] numpy 1.21.2

## Notes
This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Dian-Long You (youdianlong@sina.com).
This package was developed by Ms. Tian-Yuan Zhou (tianyuan_zhou_edu@163.com). For any problem concerning the code, please feel free to contact Ms. Zhou.