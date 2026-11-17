<img width="1024" height="256" alt="image" src="https://github.com/user-attachments/assets/5bada592-80b8-4e4c-ab1e-7a3b5e776eea" />

本项目为 2026 全球校园与人工智能算法精英大赛 算法创新赛 算法模型创新 的参赛作品
依照大赛要求，仅公开团队名称：WoodFormer非常棒

# 木材表面缺陷检测平台

基于 PyQt5 和 RT-DETR 的木材缺陷检测项目，包含图片/文件夹推理平台、三类数据准备与训练入口，以及 FEWD 结构改进代码。

## 项目结构

```text
main.py                          原始 UI 定义，供新平台继承
detector_app.py                  功能完整的图片推理入口
run_detector.ps1                Windows 启动脚本
train_oulu.ps1                  Windows 训练启动脚本
RT-DETR-main/rtdetr_pytorch/    保留的 PyTorch 模型、配置和训练工具
weights/best.pt                 best 权重
requirements.txt               依赖
LICENSE                        上游 Apache-2.0 许可证
```

## 安装与启动

推荐 Python 3.11。先安装与硬件匹配的 PyTorch 2.7.1 和 torchvision 0.22.1，再在同一环境安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python detector_app.py
# 或在已激活的 Python 环境中运行
.\run_detector.ps1
```

启动器默认使用当前环境的 `python`，也可指定 `-Python "你的环境路径/python.exe"`。不再自动跳转到作者电脑的 Conda 路径。代码兼容 Python 3.11 下本地 `.runtime` 中的 Qt 依赖；该目录只用于本机运行，已被 .gitignore 排除，无需上传。

## 使用

1. 选择 `weights/best.pt`，等待权重加载完成。
2. 打开图片并点击“开始检测”；或打开文件夹，在批量检测页开始检测。
3. 批量页左侧显示缩略图，右侧显示已完成检测的带框图像。当前图片完成后一次性切换，避免提前刷新无框原图。
4. 检测中表格只显示表头；检测结束和点击缩略图时显示真实类别、置信度、位置、大小、ID。
5. “总检测数量”是本次运行累计完成的图片检测次数，不是缺陷框数。日志上方分隔条可拖动调整高度。
6. “保存”导出带框图片，“导出检测”导出当前结果 CSV。批量结果默认写入 `detection_results/`。

模型为端到端检测，不使用 NMS，因此 IOU 滑块不影响当前推理。置信度、类别勾选和框颜色等控件实际生效。视频检测暂未接入。

## 保留的权重

`best.pt` 是 RT-DETR-R18 基线在 OULU 三类子集上微调后的 EMA 参数，不是 FEWD 权重，也不是 YOLO 模型。对应 `configs/inference/baseline.yml`。保存格式为包含 `model` 和 `metadata` 的 PyTorch 字典。

- 源检查点：零基 epoch 17（第 18 轮），按验证 mAP@0.5:0.95 选出。
- 验证 mAP@0.5:0.95：39.81%；mAP@0.5：59.44%；小目标 AP：28.18%。
- 数据划分：训练 255、验证 85、测试 85。上述是验证集结果，未作为独立测试集结果报告。
- 文件约 77 MiB，只保留 EMA 参数与必要元数据。已逐张量核对与源检查点相同。
- 不含优化器、调度器和训练进度，不能用于严格断点续训；可用于推理或作为微调初始化。

图像数据集不随代码分发。训练数据路径通过 `--data` 或 `-Data` 指定，原始数据路径通过 `--root` 指定，自行下载并准备数据。

## 数据集下载

- [作者团队发布页](https://www.triple-olab.com/oulu-wood/)
- [官方百度网盘下载](https://pan.baidu.com/s/11Ft2fnZdsqmZWP6d6aRNnw?pwd=9h3l)
- 提取码：`9h3l`
- 网盘文件：`I2GF-Net-Data/Oulu-Wood.zip`

官方数据说明为 839 张图像，包含 JPG 图片和 PASCAL VOC XML 标注。项目不附带训练图片，也不自动下载数据。将解压后的原始数据放在任意外部目录即可，按下面的命令准备子集。

## 数据准备与训练

### 环境与数据

使用项目根目录 requirements.txt 中的依赖，已验证 Python 3.11、PyTorch 2.7.1、torchvision 0.22.1。当前数据转换接口使用 tv_tensors，不要使用旧版 torchvision 0.15 的安装说明。

原始数据目录须包含 JPEGImages/*.jpg 与 Annotations/*.xml：

```powershell
python RT-DETR-main/rtdetr_pytorch/tools/prepare_oulu.py --root "你的原始数据目录"
```

脚本在原始目录下生成 DK_SK_EK_COCO，保留三类：dry_knot、sound_knot、edge_knot。实际筛选得到 425 张而非参考材料中的 478 张；按种子 2026 固定划分为 train 255、val 85、test 85。只保留三类框，其它类别在这个任务中视为背景。原始数据不修改；输出目录存在时拒绝覆盖。COCO 类别 ID 为 1/2/3，经仓库映射转为 0/1/2。manifest.json 记录划分列表；字节相同图像会阻止划分，但不同视角的同木板图像仍需按来源分组检查。

### 训练与微调

```powershell
# 检查数据、模型与损失，不更新参数
.\train_oulu.ps1 -Data "你的数据目录/DK_SK_EK_COCO" -Check
# 默认基线：从保留的已微调 best.pt 继续微调，重新开始优化器与轮数
.\train_oulu.ps1 -Data "你的数据目录/DK_SK_EK_COCO"
# FEWD 结构无对应预训练参数，从随机初始化开始
.\train_oulu.ps1 -Data "你的数据目录/DK_SK_EK_COCO" -Model fewd
# 指定基线官方 COCO 预训练参数，复现原训练的初始化方式
.\train_oulu.ps1 -Data "你的数据目录/DK_SK_EK_COCO" -Pretrained "你的COCO权重.pth"
```

原训练使用官方 [RT-DETR-R18 COCO 参数](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)。该文件已从交付目录移除，可自行下载并通过 -Pretrained 指定。默认 best.pt 已针对本子集训练过，用它再次训练不等同于复现首次实验。

默认 80 epochs、batch 4、workers 0、640×640、种子 2026。可用 -Epochs、-BatchSize、-Workers 修改，-Amp 开启混合精度。基线 backbone 学习率 1e-5，其余参数 1e-4，AdamW；在总轮数 80% 处衰减。FEWD 统一 1e-4。基线三层解码器、FEWD 六层，不能当作严格单变量消融对比。

仅使用四种在线增强：随机光照 p=0.5（亮度系数 0.8–1.2）、对比度 p=0.5（0.8–1.2）、高斯噪声 p=0.2（std=0.02）、椒盐噪声 p=0.2（0.5% 像素）。不旋转、不翻转、不改变色相或饱和度。上述强度为工程默认值，不是原文给出的精确参数。val/test 不做随机增强。

### 输出与断点续训

输出位于 RT-DETR-main/rtdetr_pytorch/output/oulu_baseline_pretrained 或 oulu_fewd。最新完整训练检查点为 checkpoint.pth，逐轮文件 checkpoint0000.pth 等；指标记录在 log.txt。训练时仅用 train 和 val，test 留出。

```powershell
.\train_oulu.ps1 -Data "你的数据目录/DK_SK_EK_COCO" -Resume "新训练输出/checkpoint.pth"
```

-Resume 需要完整训练检查点，不能传 best.pt；Epochs 表示总轮数。已有输出目录含 checkpoint.pth 时，脚本拒绝直接覆盖。需要新输出路径时可直接调用 tools/train_oulu.py 并传 --output。

保留权重的验证指标见上文“保留的权重”。该模型的效果仍需在实际生产图像及独立测试集上验证。

## 来源与许可

模型代码基于 [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)，保留 Apache-2.0 许可证及源码版权注释。项目扩展了 FEWD 模块、数据增强、训练入口和桌面推理功能。

OULU-DET 来源与下载见[作者团队发布页](https://www.triple-olab.com/oulu-wood/)。参考论文为 Luo Q 等，I²GF-Net，IEEE TIM，2024，DOI：10.1109/TIM.2024.3413144。数据集及上游预训练参数的使用条件需分别遵循各自发布者的约定。

## 仓库内容

说明统一保留在本 README 中；旧参考截图、Word 材料和大体积逐轮权重已清理。`.gitignore` 排除环境、缓存、数据集、运行日志和后续训练输出，仅允许 `weights/best.pt` 作为保留权重。请使用 Git 提交待跟踪文件，不要将整个工作目录（含 `.runtime`）直接打包上传。
