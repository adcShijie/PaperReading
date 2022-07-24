# **Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer**

## Abstract

Microsatellite instability决定肠胃道癌症患者接受免疫治疗后的反应是否良好。但是在临床中不一定每个患者都接受了MSI检测（需要额外的基因检测或免疫组织化学检测）。

于是本文使用深度残差学习(deep residual learning）直接从H&E染色组织图片中预测MSI，提供较低成本的检测方式。

## 医学背景

尽管免疫治疗如今是癌症治疗重要组成部分，但是肠胃癌患者从中的获益程度就不如其他实体恶性肿瘤，如黑色素瘤或肺癌。除非该肿瘤属于microsatellite instable（MSI）肿瘤。在胃腺癌（STAD）和结直肠癌（CRC）中大概有15%的MSI，对此免疫检查点抑制剂具有相当大的临床功效，并且该结果获得了美国食品和药物管理局（FDA）的批准。

MSI可以通过免疫组织化学检测或基因检测来识别，但是并非所有患者都进行了该类型检测（贵 排队久），于是就缺失了疾病控制的机会。

如今深度学习在一些医学数据分析任务中表现由于人力，可以利用肺、前列腺和脑等肿瘤图像预测患者生存期和肿瘤突变。为了方便MSI的普遍筛选，本文研究能否直接从H&E染色的组织切片中预测MSI状态。

## Result analysis

首先比较了5中卷积网络在三类肠胃肿瘤组织上的识别，发现Resnet-18是一种有效的肿瘤检测器，在AUC曲线下样本外区域面积>0.99，代表如今先进水平。

另外一个Resnet-18在大型patient cohort中训练分类MSI和MSS







