# High 分支样本更新接纳实验

四个实验都以 `configs/ablation_m1_dynamic_hard.yaml` 的 M1 Dynamic-hard 训练设置为基础，仅替换 High 分支更新是否接纳。训练样本路由仍从原有 `routing_uncertainty_csv` / `routing_reference_partition` 读取；Low 分支仍使用同一 Dynamic-hard 缓存、损失、采样、增强与刷新代码。拒绝 High 更新只跳过当前 High optimizer step，不重抽病例，也不把此步替换为 Low 更新。所有分支仍更新同一个 Student。

High 基础目标仍是 affinity random-walk 伪标签上的 `high_bce_weight * BCE + high_dice_weight * Dice`。接纳指标只代表无标签代理证据，不保证真实伪标签正确或目标 Dice 提升。

## VoxTell 语义特征

VoxTell 的 `return_features=True` 真实接口返回 `d5_pre_head`，这是最终 D5 mask einsum 之前的 3D 视觉张量。VoxTell 的 decoder 在较早阶段已融合文本 mask embedding，因此将其作为文本条件化视觉表示的间接代理；不会把该张量与 Qwen embedding 做余弦相似度。特征只在冻结 LoRA 的 source teacher 下提取，并在伪标签前景位置采样、逐病例限制 token 数，特征保存在 CPU。它仍可能包含系统性错误结构，不能当成正确性证明。

## 四种模式

- **GA (`gradient_agreement`)**：当前 High 全病例梯度与固定 High 参考病例梯度逐一算归一化内积，均值严格大于 0 接纳。零、非有限或不足参考梯度拒绝。只检查方向，不做显著性检验或语义判断。
- **CUV (`cross_case_validation`)**：用 `Delta_ij = -learning_rate * dot(g_j, g_i)` 估计候选步对参考病例伪标签损失的一阶变化。对参考病例变化计算单侧 t 上界，只有上界小于 0 接纳。该代理检验依赖随机参考抽样、病例间近似独立和变化量近似正态等假设；小样本或数值异常拒绝。实际 Student 使用 AdamW，因此该 SGD 形式的一阶更新只是方向代理，不精确模拟 AdamW 参数步长；它不等于真实 Dice 检验。
- **CSS (`semantic_subspace`)**：对除候选病例外的固定参考病例前景 token 做归一化 SVD，取累计谱能量达到 `spectral_energy` 的最小子空间。候选相对重建残差与参考病例 leave-one-case-out 残差的 median + `robust_mad_multiplier * 1.4826*MAD` 比较。候选不会参与自身子空间或阈值构造。有效病例/分解不足时拒绝；该残差不是伪标签错误率。
- **SUV (`semantic_update_validation`)**：独立计算 CSS 和 CUV 判定，只在两者同时通过时接纳（AND）；没有手工加权分数，也不额外运行 GA 判定。

所有模式 `none`（默认）会绕过样本接纳代码，沿用原有 scaler backward、梯度裁剪、optimizer/scaler step 顺序。开启接纳后，High 梯度使用当前 GradScaler scale 手动缩放，判断前再反缩放；High 被拒绝时不调用 optimizer 或 scaler step/update。Low 仍原样调用原始 scaler 路径。

## 缓存与计算

参考病例从固定 High 路由中用 `experiment.seed` 派生的 RNG 无放回抽样，候选病例排除在外。每次刷新按配置顺序逐例生成数据，不并存计算图；梯度向量和语义 token 留在 CPU。缓存元数据写到实验输出 `high_selection_cache_metadata.json`，包括病例 ID、生成迭代、源权重文件签名、LoRA 参数 SHA-256、刷新间隔和配置。刷新间隔内的参考证据是过期快照近似，不会假装与当前 Student 同步；低接纳率不会触发无条件训练退化。

额外工作量按每次缓存刷新估计（reference_count 为 R；单病例 P 个 patch）：

| 模式 | 额外参考工作 | 缓存 |
|---|---|---|
| GA | 刷新时约 `2RP` 次 High 前向、`RP` 次反向（每 patch 一次 source teacher 和一次 Student）；当前候选的反向属于原训练 | R 个 CPU 梯度向量 |
| CUV | 与 GA 相同；统计计算仅向量点积 | R 个 CPU 梯度向量 |
| CSS | 刷新时 `RP` 次 source-teacher 特征前向；每个 High 候选还需当前病例 `P` 次特征前向 | R×tokens_per_case×channels 的 CPU 特征 |
| SUV | 梯度参考的 `2RP` 次前向、`RP` 次反向，加语义参考的 `RP` 次前向；每个 High 候选另有 `P` 次特征前向 | CPU 梯度与特征 |

单张 3D patch 图驻留 GPU；参考病例数、token 数和刷新间隔分别控制计算与缓存，不保证 11GB GPU 下的实际速度。每个 High 候选接纳结果逐次追加到 `high_selection_training_log.csv`，含拒绝原因和模式对应统计。

## 参数

四份 YAML 都设置：

- `mode`: 各自固定为 `gradient_agreement`、`cross_case_validation`、`semantic_subspace` 或 `semantic_update_validation`。
- `reference_count: 8`: 每次刷新参考 High 病例数（至少 2；候选不计入）。
- `reference_refresh_interval: 100`: 以训练迭代计的快照刷新间隔。
- `significance_level: 0.05`: CUV 单侧显著性水平。
- `spectral_energy: 0.9`: CSS 累计谱能量保留率。
- `robust_mad_multiplier: 3.0`: CSS 阈值稳健离散倍数。
- `tokens_per_case: 256`: 每例前景 token CPU 缓存上限。

统计检验和子空间阈值不使用训练 GT 调参。项目已有基线用验证 GT 选 checkpoint 的流程仍保留，但这是 oracle checkpoint selection 记录，不能视为部署期无标签选择方法。

## 启动

以下命令只供后续实验使用，本次实现过程没有执行：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH}"
python scripts/run_cue_sfda.py train --config configs/high_selection_GA.yaml --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
python scripts/run_cue_sfda.py train --config configs/high_selection_CUV.yaml --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
python scripts/run_cue_sfda.py train --config configs/high_selection_CSS.yaml --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
python scripts/run_cue_sfda.py train --config configs/high_selection_SUV.yaml --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels
```

四份配置写入互相独立的 `outputs/high_selection/{GA,CUV,CSS,SUV}` 目录。检查配置文件中的 M1 固定路由文件和数据路径在目标机器上存在后再启动。

## 理论和数值边界

GA 可能拒绝有益但病例特异的梯度，也可能接受共享系统错误。CUV 的梯度只是一阶局部近似，有限学习率、共享参考偏差、病例相关性和梯度快照陈旧都会使检验偏离实际变化；参考数 8 也令 t 检验不稳定。CSS 的参考病例可能共享错误，MAD 为零或特征秩不足会导致保守拒绝，随机 token 采样可能遗漏小病灶。SUV 取两个代理的交集，通常会进一步降低接纳率，且继承两者全部失败模式。所有路径遇到零梯度、NaN/Inf、空前景、参考不足、SVD 失败时都拒绝，不自动切换回无条件自训练。
