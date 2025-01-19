# An Official Code Implementation of GraspGPT & FoundationGrasp

# GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping (RA-L 2023)
[[paper]](https://ieeexplore.ieee.org/abstract/document/10265134) [[website]](https://sites.google.com/view/graspgpt/home) [[video]](https://www.youtube.com/watch?v=qq0DMdHRw1E)

[Chao Tang](https://mkt1412.github.io/), [Dehao Huang](http://github.com/red0orange), [Wenqi Ge](https://github.com/Laniakea77), [Weiyu Liu](http://weiyuliu.com/), [Hong Zhang](https://faculty.sustech.edu.cn/zhangh33/en/)    
Southern University of Science and Technology, Georgia Institute of Technology  

This is a Pytorch implementation of GraspGPT. If you find this work useful, please cite:
```
@article{tang2023graspgpt,
  title={GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping},
  author={Tang, Chao and Huang, Dehao and Ge, Wenqi and Liu, Weiyu and Zhang, Hong},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
  ```
We have borrowed tons of code from [GCNGrasp](https://arxiv.org/abs/2011.06431). Please also consider citing their work:
```
@inproceedings{murali2020taskgrasp,
  title={Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping},
  author={Murali, Adithyavairavan and Liu, Weiyu and Marino, Kenneth and Chernova, Sonia and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  year={2020}
}
  ```

**Note that we provide a light-weight version of GraspGPT in this repo. It is with faster inference speed and easier to train. Also, the model achieves comparable performance to the one described in our paper.**

## Installation

**Our code is tested on Ubuntu 20.04, python 3.7, cuda 11.3, pytorch 1.11.0, and pytorch lightning 0.7.1.**

1) Create a virtual env or conda environment with python 3.7:
```shell
conda create --name graspgpt python=3.7
conda activate graspgpt
```

2) Git clone the repo to your machine:
```shell
git clone https://github.com/mkt1412/GraspGPT_public.git
```

3) Install dependencies:
```shell
cd GraspGPT_public
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset
The Language Augmented TaskGrasp (LA-TaskGrasp) dataset is developed based on [TaskGrasp](https://arxiv.org/abs/2011.06431) dataset. To perform training and evaluation on the LA-TaskGrasp dataset, download the dataset [here]([https://gatech.box.com/s/821qcle6x67lvatph4fzvkc4o128g89c](https://gatech.box.com/s/et2m6scmqxjyigcivr1bcii2mepxsj2v)) and place it in the root folder as `data`:
```shell
unzip ~/Downloads/data.zip -d ./
rm ~/Downloads/data.zip
```

To visualize the collected point clouds and labeled task-oriented grasps, please refer to the [github repo](https://github.com/adithyamurali/TaskGrasp) of TaskGrasp dataset under Usage section.

To run any of the demo scripts below, download the pre-trained models [here](https://gatech.box.com/s/egzrbu1j1r20u1mrduzrrzzktf95ahuf) and put them in the `checkpoints` folder.

## Demo 
We provide two types of demos:     
* **Database version**, where GraspGPT uses the pre-generated natural language descriptions for task-oriented grasp pose prediction.
* **Interactive version**, where GraspGPT interacts with an LLM via prompts to generate language descriptions for task-oriented grasping.      

Pre-trained models for each class and task split are provided:
```
gcngrasp_split_mode_o_split_idx_0_.yml
gcngrasp_split_mode_o_split_idx_1_.yml
gcngrasp_split_mode_o_split_idx_2_.yml
gcngrasp_split_mode_o_split_idx_3_.yml

gcngrasp_split_mode_t_split_idx_0_.yml
gcngrasp_split_mode_t_split_idx_1_.yml
gcngrasp_split_mode_t_split_idx_2_.yml
gcngrasp_split_mode_t_split_idx_3_.yml
```

### Database version
Objetc class: *pan*, task: *pour*
```shell
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name pan --obj_class saucepan --task pour
```
Objetc class: *spatula*, task: *scoop*
```shell
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name spatula --obj_class spatula --task scoop
```
Objetc class: *mug*, task: *drink*
```shell
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name mug --obj_class mug --task drink
```

### Interactive version
To run interactive version on your machine, you need to set your OpenAI API key in `gcngrasp/data_specificatgion.py` line 116.
```
OPENAI_API_KEY = "xx-xxxxxx"   
```

Objetc class: *pan*, task: *pour*
```shell
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name pan --obj_class saucepan --task pour
```
Objetc class: *spatula*, task: *scoop*
```shell
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name spatula --obj_class spatula --task scoop
```
Objetc class: *mug*, task: *drink*
```shell
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name mug --obj_class mug --task drink
```

## Training
To train GraspGPT from scratch:
```shell
python gcngrasp/train.py --cfg_file cfg/train/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml
```
You can replace t_split_idx_3 with other split numbers (i.e., 0, 1, 2) or object class split (i.e., o_split_x).

## Evaluation
To evaluate a pre-trained model:
```shell
python gcngrasp/eval.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --save
```
Similarly, feel free to try out different split numbers and object class split.

# FoundationGrasp: Generalizable Task-Oriented Grasping with Foundation Models
[[preprint]](https://arxiv.org/pdf/2404.10399.pdf) [[website]](https://sites.google.com/view/foundationgrasp) [[video]](https://www.youtube.com/watch?v=0AEwXl9i77o)

[Chao Tang](https://mkt1412.github.io/), [Dehao Huang](http://github.com/red0orange), [Wenlong Dong](https://scholar.google.com/citations?user=NNafpNQAAAAJ&hl=zh-CN), [Ruinian Xu](https://scholar.google.com/citations?user=qy644T4AAAAJ&hl=en), [Hong Zhang](https://faculty.sustech.edu.cn/zhangh33/en/)  

Since FoundationGrasp is an extension of GraspGPT, we do not provide a separate GitHub repo for the code implementation and data of FoundationGrasp. If you find FoundationGrasp useful in your research, please cite:
```
@article{tang2024foundationgrasp,
  title={FoundationGrasp: Generalizable Task-Oriented Grasping with Foundation Models},
  author={Tang, Chao and Huang, Dehao and Dong, Wenlong and Xu, Ruinian and Zhang, Hong},
  journal={arXiv preprint arXiv:2404.10399},
  year={2024}
}
  ```
## Data
Step 1. Download the original LA-TaskGrasp dataset [here](https://gatech.box.com/s/et2m6scmqxjyigcivr1bcii2mepxsj2v) and place it in the root folder as `data`.   
Step 2. Download the updated language description data of the LaViA-TaskGrasp dataset [here](https://gatech.box.com/s/m1d2u3ft36flhfonygmud5tfezakmgs2) and replace the old one in the LA-TaskGrasp dataset (i.e., obj_gpt_vx, task_gpt_vx).   
Step 3. For language instruction data, please refer to `data/task_ins_v2`.  
Step 4. For image data, please refer to `data/rgb_images`.   
Step 5. Modify the dataloader in `gcngrasp/data/GCNLoader.py` so that the model can read the updated language description and image data.  

## Code Implementation
Step 1. The semantic branch of FoundationGrasp is similar to GraspGPT. Feel free to use to the code implementation of GraspGPT provided above.  
Step 2. The geometric branch of FoundationGrasp is largely inspired by [XMFNet](https://github.com/diegovalsesia/XMFnet/tree/main). Please refer to `XMFnet/model.py` to reproduce the geometric branch.



