# GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping
[[arxiv]](https://arxiv.org/pdf/2307.13204.pdf) [[website]](https://sites.google.com/view/graspgpt/home) [[video]](https://www.youtube.com/watch?v=kHhQMrmeun0)

[Chao Tang](https://mkt1412.github.io/), [Dehao Huang](http://github.com/red0orange), [Wenqi Ge](https://github.com/Laniakea77), [Weiyu Liu](http://weiyuliu.com/), [Hong Zhang](https://faculty.sustech.edu.cn/zhangh33/en/)    
[RCV Lab](https://rcvlab.eee.sustech.edu.cn/), Southern University of Science and Technology, Georgia Institute of Technology  

This is a Pytorch implementation of GraspGPT. If you find this work useful, please cite:
```
@article{tang2023graspgpt,
  title={GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping},
  author={Tang, Chao and Huang, Dehao and Ge, Wenqi and Liu, Weiyu and Zhang, Hong},
  journal={arXiv preprint arXiv:2307.13204},
  year={2023}
}
  ```
We have borrowed tons of code from [GCNGrasp](https://arxiv.org/abs/2011.06431), please also consider citing their work:
```
@inproceedings{murali2020taskgrasp,
  title={Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping},
  author={Murali, Adithyavairavan and Liu, Weiyu and Marino, Kenneth and Chernova, Sonia and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  year={2020}
}
  ```

## Demo 
We provide two types of demos: (1) database version, where GraspGPT uses the pre-generated natural language descriptions for task-oriented grasp pose prediction, (2) interactive version, where GraspGPT interacts with an LLM via prompts to generate language descriptions for task-oriented grasping. 

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
```
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name pan --obj_class saucepan --task pour
```
Objetc class: *spatula*, task: *scoop*
```
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name spatula --obj_class spatula --task scoop
```
Objetc class: *mug*, task: *drink*
```
python gcngrasp/demo_db.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name mug --obj_class mug --task drink
```

### Interactive version
To run interactive version on your machine, you need to set your OpenAI API key in `gcngrasp/data_specificatgion.py` line 116.
```
OPENAI_API_KEY = "xx-xxxxxx"   
```

Objetc class: *pan*, task: *pour*
```
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name pan --obj_class saucepan --task pour
```
Objetc class: *spatula*, task: *scoop*
```
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name spatula --obj_class spatula --task scoop
```
Objetc class: *mug*, task: *drink*
```
python gcngrasp/demo_llm.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml --obj_name mug --obj_class mug --task drink
```

## TODOs
- [ ] installation tutorial
- [ ] dataset download
- [ ] training and evaluation

