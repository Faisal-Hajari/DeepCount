# DeepCount
Object Counting framework based on pytorch. This framework focuses on point prediction (i.e. the output of the model is points). 


# Data and Augmentation 
## Datasets: 
The framework provides multiple datasets formats that can be handeled as PyTorch Dataset: 
### JasonDataset
 similar to coco formated datasets. this dataset expects a path to a folder where an json file contains the relative paths to the images and the points labels for the image

**example json**
```json 
[
    {
    "image":"train/DSC_955.jpeg",
    "points":[
        [120, 120], 
        [55, 55]
        ]
    },
    {
    "image":"train/DSC_9625.jpg",
    "points":[
        [120, 120], 
        [55, 55]
        ]
    }
]
```

## Augmentation 
The framewrok mainly relay on [Albumnation](https://github.com/albumentations-team/albumentations) for most of the augmentaion. please refer to their doc and keep in mind we need suport for kyepoints such that the lable reflects the augmented image. In addtion to albumnation the framework also provides some augmentation of it's on, or a fixed transformation. all can be found under `deepcount.data.transforms`. 
Here is a list of transformations available in `deepcount.data.transforms`:
| transforms            | usecase                             | type        |          
|-----------------------|-------------------------------------|-------------|
| ToTensor              | Convert the image to pytorch tensor | Fix         |
| CutMix                | implementation to [CutMix](https://arxiv.org/abs/1905.04899) while preserving point locations| Batch|

* Fix: Albumnation has a version of it that some time breaks with the framework
* Btach: Augmentation that is apllied sepreatly after getting the batch from the data loader
* New: New dataset level augmentation that can be passed inside A.Compose()

Feel free to explore these augmentations to enhance your object counting framework!

**Example code**
``` python 

from deepcount.data.datasets import JasonDataset  
import albumentations as A
from deepcount.data.transforms.transforms import ToTensor, CutMix
transform = A.Compose(
    [
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(),
    A.RandomSizedCrop(min_max_height=(112, 235), width=224, height=224,),
    ToTensor()
    ], 
    #this is an important part to add to your compose to recongnise keypoints
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
)

#this datase retruns a image and kpoints, where kpoints is binary map of shape (HW). 
dataset = JasonDataset("path/to/dataset/folder", json_file_name="annotation.json", transforms=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
cutmix= CutMix(1.0)
for epoch in epochs: 
    for batch in dataloader: 
        images, kpoints = batch 
        images, kpoints = cutmix(images, kpoints)
        ... 
        #if you need x,y of the points just use. Note: this works on sample level if done on batches you'll get [sample_numper, x, y]: 
        points_xy = kpoints[1, :, :].nonzero()
