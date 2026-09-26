# pytorch-learning-journey

入门阶段 PyTorch 学习记录

## 环境
- Python 3.9
- PyTorch 2.x
- 硬件：CPU

## 目录结构
-part1 分部学习（Dataset、DataLoader、conv-pool、activation-linear、tensorboard、loss-optimizer）
-part2 最终模型保存

### CIFAR-10 训练结果
- 模型：3层 CNN
- 准确率：**62.21%**
- 调参对比：学习率 0.01 vs 0.1，loss 曲线见 `loss_comparison.png`

### Loss 曲线
![loss曲线](loss_comparison.png)

### 准确率曲线
![acc曲线](accuracy.png)
