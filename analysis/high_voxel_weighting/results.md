# 高稳定样本逐体素置信度加权消融

## 实验约束与实现

- 数据集、划分、44 个评估病例、推理滑窗步长 0.5、阈值 0.5 和指标定义沿用仓库原协议。训练中关闭验证与早停；每个固定训练终点只评估一次，未用测试集 GT 选择权重或检查点。
- 逐体素 `joint_voxel_confidence` 来自已有 `ablation_consensus_cache`，未重新计算。对 101 个训练病例逐一核对缓存几何与原图空间尺寸；高稳定伪标签和图像 patch 使用同一个 `case_slicer`。confidence 在加载时核验 case、图像大小和 mtime，提取后再核验 patch 张量尺寸。
- 加权仅用于高稳定分支 BCE 和 Dice；BCE 原有前景/背景分别归一化策略保留。低稳定更新、路由、损失系数、增强、优化器及其余配置不变。未使用 prototype。
- 默认权重为 `w=0.3+0.7C`；配置项可设 `enabled/function/floor/scale/power`，同时实现线性和幂函数以支持后续单变量消融。低置信度体素仍有正权重，不会被丢弃。

## 正确性验证

`PYTHONPATH=src python -m pytest tests/test_losses.py -q`：7 项通过。包含置信度映射范围、低置信体素保留、权重全 1 时 BCE/平衡 BCE/Dice 与原实现一致、加权损失梯度有限且非零等检查。训练日志中的图像/目标/confidence patch 形状检查对实际切片再次执行。

## 逐步消融

| 实验 | 配置 | 训练范围 | Dice | mIoU | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| 高稳定等权 | [high_only_equal YAML](../../configs/high_voxel_weighting_high_only_equal.yaml) | 仅高稳定，1000 步 | 0.3273270305 | 0.2510412623 | 0.3377105661 | 0.4919110542 |
| 高稳定线性加权 | [high_only YAML](../../configs/high_voxel_weighting_high_only.yaml) | 仅高稳定，1000 步；`w=0.3+0.7C` | 0.3271719234 | 0.2503797377 | 0.3295978954 | 0.5040907737 |
| 完整高低稳定线性加权 | [full YAML](../../configs/high_voxel_weighting_full_high_low.yaml) | 原动态 hard 高低路由，固定 1250 步；线性加权只作用于高稳定分支 | 0.2799542809 | 0.2055626658 | 0.2142559213 | 0.6675833432 |
| 完整高低稳定幂加权 | [power2 YAML](../../configs/high_voxel_weighting_full_high_low_power2.yaml) | 同完整流程，1250 步；`w=0.3+0.7C²` | 0.2597602816 | 0.1875766366 | 0.1941242958 | 0.6942897884 |
| 完整高低稳定等权 | [equal YAML](../../configs/high_voxel_weighting_full_high_low_equal.yaml) | 与完整流程相同；关闭高稳定 voxel weighting，1250 步 | 0.3231063570 | 0.2459611690 | 0.2640275239 | 0.6070583397 |

前两次运行使用相同 seed 和相同采样序列，仅启用加权与否不同。病例级 Dice 的 weighted-minus-equal 平均差为 -0.0001551，中位数约 0；15 例提高、27 例下降、2 例不变，标准差 0.01509，范围 [-0.04364, 0.06616]。配对差异见 [CSV](high_only_case_dice_differences.csv)。

| 高稳定仅训练总体差异（加权 - 等权） | 差值 |
|---|---:|
| Dice | -0.0001551 |
| mIoU | -0.0006615 |
| Precision | -0.0081127 |
| Recall | +0.0121797 |

线性加权使预测体积增加（总量 +9,592 个体素；病例平均 +218），与 Recall 上升、Precision 下降一致。高稳定仅训练的 Dice 几乎持平但略降，因此目前没有证据表明该线性加权有效；待完整流程结果后再决定是否进行预先规定的幂函数单变量试验。

完整流程线性加权相对历史第 1250 步：Dice -0.0644613、mIoU -0.0583349、Precision -0.0856155、Recall +0.0927829。44 例中 10 例 Dice 提高、34 例降低；病例差异均值 -0.06446、中位数 -0.03318、标准差 0.11622，范围 [-0.55750, 0.12158]。这属于明显性能退化，不是有效方案。病例级差异见 [CSV](full_high_low_linear_comparison/case_dice_differences.csv)。

平方权重相对历史第 1250 步：Dice -0.0846553、mIoU -0.0763209、Precision -0.1057471、Recall +0.1194894。44 例中 9 例 Dice 提高、35 例降低；病例差异均值 -0.08466、中位数 -0.03975、标准差 0.14191，范围 [-0.57411, 0.10596]。平方映射进一步降低了 Dice 和 Precision，说明单纯强化中低置信体素降权也没有纠正预测扩张。病例级差异见 [CSV](full_high_low_power2_comparison/case_dice_differences.csv)。

完整流程等权的历史对照 Dice 为 0.3231064、mIoU 0.2459612、Precision 0.2640275、Recall 0.6070583；相对历史最佳 Dice 差 -0.0213093，9 例提高、35 例降低，病例均值差 -0.02131、中位数 -0.00177、标准差 0.07383。完整线性加权相对这次同 seed 等权：Dice -0.0431521、mIoU -0.0403985、Precision -0.0497716、Recall +0.0605250，病例 Dice 12 例提高、32 例降低。平方加权相对等权：Dice -0.0633461、mIoU -0.0583845、Precision -0.0699032、Recall +0.0872314，病例 10 例提高、34 例降低。对应病例差异见 [linear vs equal](full_high_low_linear_vs_equal/case_dice_differences.csv) 与 [power² vs equal](full_high_low_power2_vs_equal/case_dice_differences.csv)。两种加权都比直接等权差，且 Recall 增、Precision 降，支持预测区域扩张导致 Dice 退化的判断。

## 置信度与监督有效性

59 个高稳定训练病例的缓存空间共 802,462,080 个体素（包含训练 patch 内的 padding），总体置信度均值 0.999812，线性权重均值 0.999869。分布如下：

| 置信度区间 | 体素数 | 占比 | 对应权重区间 |
|---|---:|---:|---:|
| [0, 0.2) | 2,640 | 0.000329% | [0.30, 0.44) |
| [0.2, 0.4) | 7,562 | 0.000942% | [0.44, 0.58) |
| [0.4, 0.6) | 19,992 | 0.002491% | [0.58, 0.72) |
| [0.6, 0.8) | 144,354 | 0.017989% | [0.72, 0.86) |
| [0.8, 1.0] | 802,287,532 | 99.978248% | [0.86, 1.00] |

相同置信区间用于平方权重时，权重区间依次为 `[0.30,0.328)`、`[0.328,0.412)`、`[0.412,0.552)`、`[0.552,0.748)`、`[0.748,1.00]`；总体平均权重为 0.999764。完整统计和病例文件见 [power² 分布](power2_confidence_weights/confidence_weight_distribution.json)。

高稳定-only frozen pseudo cache 的逐 patch 分析（重叠 patch 按训练暴露次数计；排除前景软标签不足阈值的跳过 patch）有 164 个有效 patch、126 个跳过 patch。有效 patch 中前景/背景暴露分别为 1,202,901 / 1,159,570,731；前景置信度均值 0.869928（权重均值 0.908950），背景均值 0.999794（权重均值 0.999856）。低权重体素主要落在前景附近，线性权重对已极高置信的背景几乎不起作用。逐区间 exposure 数、跳过 patch 和详细记录见 [分析 JSON](effective_supervision_source_cache.json) 与 [置信度分布 JSON](confidence_weight_distribution.json)。完整线性 run 的动态 hard 缓存于第 701 步刷新一次，覆盖 101 例；新增前景 334,937 体素、移除 227 体素，平均置信变化 -0.001076。训练权重仍固定复用原始 source `joint_voxel_confidence`；动态 hard 伪标签刷新但置信图不重算。刷新记录已归档于 [JSON](run_logs/full_high_low_linear_refresh_history.json)。

该缓存统计是训练初始源伪标签；完整动态流程运行后还需汇报 refresh 后置信度分布与有效监督统计。

## 历史最佳参照

仓库历史记录 `G_dynamic_hard_final` 在第 1250 步评估的 44 病例均值为 Dice 0.3444156224、mIoU 0.2638975594、Precision 0.2998713834、Recall 0.5748004052。对应记录和配置见 [历史结果目录](historical_dynamic_hard/)。此次完整流程固定训练到相同步数后仅评估一次，训练中关闭验证/早停，不据测试集 GT 选择检查点或阈值。

历史上的 `E1_eval_1000` 文件夹实际日志加载的是第 500 步检查点；因此本次高稳定等权和线性加权比较均重新以相同 seed 训练到 1000 步，配对比较不依赖该误标产物。

## 结果与建议

`w=0.3+0.7C` 在高稳定-only 和完整流程都没有提升；`w=0.3+0.7C²` 的完整流程结果更差。两种权重都令 Recall 上升、Precision 下降，平方映射没有缓解预测扩张，说明现有置信图与目标误差的相关性不足，且 99.98% 体素置信度落在最高区间。GPU 1 同 seed 的 full 等权对照已完成，Dice 0.3231064，高于线性和平方权重，但仍未超过历史最佳 0.3444156224。基于这些结果不建议继续调整 confidence 函数；应先改进或校准可复用置信度来源，或保持等权监督，再用独立验证方案评估。
