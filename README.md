# All-Higher-Stages-In Adaptive Context Aggregation for Semantic Edge Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/document/9762721)

本仓库为论文 [All-Higher-Stages-In Adaptive Context Aggregation for Semantic Edge Detection](https://ieeexplore.ieee.org/document/9762721) （简称 AHSI）的官方 PyTorch 实现。

---

## 🚀 简介

语义边缘检测（Semantic Edge Detection, SED）旨在同时定位目标边界并识别其语义类别。现有 SED 方法多采用位置对齐的多阶段特征融合，难以满足边缘点多样化的语义需求，且常导致边缘细节模糊。为此，我们提出：

- **All-Higher-Stages-In Adaptive Context Aggregation (All-HiS-In ACA)**：通过跨阶段自注意力机制，自适应地为低层特征聚合全部高层语义，实现细致且语义丰富的边缘检测。
- **ICE（跨层互补增强）模块**：无需参数，通过相邻阶段特征互补，增强特征表达。
- **OSI（对象级语义集成）模块**：利用中间预测的对象级语义先验，进一步统一和细化边缘特征。

本方法在 SBD 和 Cityscapes 数据集上均取得新的 SOTA。

## 🏗️ 算法框架

![image](https://github.com/user-attachments/assets/c6361a07-41d8-4cda-81dc-b2dfdf3a54a3)


## 📚 论文引用
如果该项目对你有帮助，请引用我们的论文：
```
@ARTICLE{bo2022AHSI,
  author={Bo, Qihan and Ma, Wei and Lai, Yu-Kun and Zha, Hongbin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={All-Higher-Stages-In Adaptive Context Aggregation for Semantic Edge Detection}, 
  year={2022},
  volume={32},
  number={10},
  pages={6778-6791},
  keywords={Semantics;Image edge detection;Feature extraction;Open systems;Image segmentation;Horses;Aggregates;Semantic edge detection;multi-stage feature fusion;adaptive context aggregation;complementary feature enhancement;object-level semantic integration},
  doi={10.1109/TCSVT.2022.3170048}
}
```

## 📬 联系
提问/建议请提 Issues

邮箱：mawei@bjut.edu.cn | dolijunc@qq.com(代码相关的)
