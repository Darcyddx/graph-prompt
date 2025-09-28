<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 15px; margin: 0;">
    <a href="https://darcyddx.github.io/gcr" style="display:inline-flex; align-items:center;">
      <img src="assets/gcr_icon.png" alt="Project Page" style="width:38px; height:38px; object-fit:contain;">
    </a>
    Graph Your Own Prompt
  </h1>



[Xi Ding](https://darcyddx.github.io/), [Lei Wang](https://leiwangr.github.io/), [Piotr Koniusz](https://www.koniusz.com/), [Yongsheng Gao](https://experts.griffith.edu.au/19112-yongsheng-gao)
</div>

## 📑 Citation
```bibtex
@article{ding2025graph,
  title={Graph Your Own Prompt},
  author={Ding, Xi and Wang, Lei and Koniusz, Piotr and Gao, Yongsheng},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 📖 Overview
![framework](assets/graph_pipeline.png)
<p style="font-size: 1.8rem; line-height: 1.7;">
  GCR is a <strong>plug-and-play, parameter-free, and lightweight</strong> method that works with <span style="color: red;">any model</span>, improving feature quality and generalization without changing the architecture.
</p>


## ⚙️ Installation

```bash
git clone https://github.com/Darcyddx/graph-prompt.git
cd graph-prompt
bash setup.sh
```

## 📦 Data Preparation
Before running the experiments, please prepare the datasets as follows:

1. **Download datasets**  
   - CIFAR-10 and CIFAR-100 will be downloaded automatically if you run the training code.
   - You can download the [Tiny ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) from Kaggle.


2. **Organize the data structure**  
<!-- graph-prompt/
├── data/
│ ├── cifar100/
│ │ ├── train/
│ │ ├── val/
│ └── imagenet/
│ ├── train/
│ ├── val/ -->

## 🚀 Usage