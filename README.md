

<div align="center">

<samp>

<h2> OSSAR: Towards Open-Set Surgical Activity Recognition in Robot-assisted Surgery </h1>

<h4> Long Bai*, Guankun Wang*, Jie Wang, Xiaoxiao Yang, Huxin Gao, Xin Liang, An Wang, Mobarakol Islam, Hongliang Ren </h3>

<h2> IEEE International Conference on Robotics and Automation (ICRA) 2024 </h1>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2402.06985>)]** |
    
---

</div>     



If you find our code or paper useful, please cite the paper as

```bibtex
@article{bai2024ossar,
  title={OSSAR: Towards Open-Set Surgical Activity Recognition in Robot-assisted Surgery},
  author={Bai, Long and Wang, Guankun and Wang, Jie and Yang, Xiaoxiao and Gao, Huxin and Liang, Xin and Wang, An and Islam, Mobarakol and Ren, Hongliang},
  journal={arXiv preprint arXiv:2402.06985},
  year={2024}
}
```


## Abstract
In the realm of automated robotic surgery and computer-assisted interventions, understanding robotic surgical activities stands paramount. Existing algorithms dedicated to surgical activity recognition predominantly cater to pre-defined closed-set paradigms, ignoring the challenges of real-world open-set scenarios. Such algorithms often falter in the presence of test samples originating from classes unseen during training phases. To tackle this problem, we introduce an innovative Open-Set Surgical Activity Recognition (OSSAR) framework. Our solution leverages the hyperspherical reciprocal point strategy to enhance the distinction between known and unknown classes in the feature space. Additionally, we address the issue of over-confidence in the closed-set by refining model calibration, avoiding misclassification of  unknown classes as known ones. To support our assertions, we establish an open-set surgical activity benchmark utilizing the public JIGSAWS dataset. Besides, we also collect a novel dataset on endoscopic submucosal dissection for surgical activity tasks. Extensive comparisons and ablation experiments on these datasets demonstrate the significant outperformance of our method over existing state-of-the-art approaches. Our proposed solution can effectively address the challenges of real-world surgical scenarios.


---
## Environment
    conda env create -f environment.yml -n new_env_name

---
## Dataset
JISAWS Dataset (Needle Passing, Knot Tying, Suturing)
    - [Official](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
---

### Run training
- OSR training on surgical activity recognition. For cross validation, you just need to modified the dir in the [dataset].yml. The training log shall be in './results'.
    ```bash
    bash scripts/osr/ossar/ossar_train_ossar.sh
    ```

---

## Acknowledgement
Baseline code from the [OpenOOD](https://github.com/Jingkang50/OpenOOD) library.

---
