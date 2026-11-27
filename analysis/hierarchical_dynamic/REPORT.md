# CUE-SFDA 双层动态不确定性实验进度

## 执行状态

实验分支：`hierarchical-dynamic-sfda`。

M1（全病例均匀采样、同一病例同时执行 High 与 Low 策略）和 M0 对齐的历史 Dynamic-hard 路由对照，分别运行在物理 GPU 0 和 GPU 1，使用独立输出、缓存和进程。均从同一 VoxTell 源权重和 seed 42 独立初始化；训练预算为 2,000 iterations，patch 为 192³，学习率 1e-5，Dynamic-hard 最多刷新一次。训练集 GT 不参与伪标签、采样或权重构造，只供既有验证流程读取。

两项均已完成 101 例的 version-0 冻结 Source teacher 缓存。M1 在 iteration 250 检查点续跑并启用分支诊断；M0 的 iteration 450 验证曾因表面距离计算未去除单例通道轴而中断，已修复并从 iteration 400 检查点续跑。两进程分别使用 GPU0/1，缓存和 checkpoint 独立。

## M1 实现

新增 `experiment.hierarchical_mode: M1`：病例从全部 101 例均匀抽取；每个 patch 依次构造并反传 High 与 Low 损失，随后统一进行一次病例级 optimizer step；原 High 有效前景检查保留，Low 继续使用 Dynamic-hard consensus BCE。两个 3D 计算图按顺序释放，降低峰值显存。

M1 的最初运行段没有逐分支梯度范数；现已在 iteration 250 检查点续跑，新段按分支逐个求梯度、记录 High/Low 损失及梯度范数，并保留已有 optimizer/checkpoint 状态。

## M2 准备

从 Stage-1 已有 101 例预测缓存离线生成固定像素可靠性图，没有再次推理。按归一化二分类熵（log2）计算 `W=1-[H(mean(image views))+H(mean(prompt views))]/2`，并按训练 ROI 几何插入 padding 坐标。检测到原始预测为空的 20 例已标记，M2 会保留 KL consistency 并跳过额外共识伪标签 BCE。配置和训练代码已准备，M2 的 version-0 Dynamic-hard 缓存已复用 M1 的冻结教师版本；正式训练等待 M1 结果审核。

## 结果

训练仍在进行，Zero-shot 初始验证均值为 Dice=0.283467（44例；分层数量 22/10/12）。M1 iteration 250 的 Dice=0.310338，空预测 6 例；相对初始结果提升超过0.02的12例，下降超过0.02的9例。初始 Dice<0.1、0.1–0.5、>0.5 三组的 M1 均值分别为0.0508、0.3375、0.7635。M0 iteration 400 Dice=0.294836，空预测9例；提升超过0.02的9例，下降超过0.02的8例，三组均值为0.0275、0.3242、0.7604。这些是早期验证，不能作为最终结论。M0 iteration 450 Dice=0.293868、空预测9例。HD95/ASD 已写入逐病例 CSV；9例空预测对应无穷表面距离，因此 44 例聚合均值为 inf，35例有限值的 HD95 均值为56.47 mm，ASD 均值为33.93 mm。暂不据早期验证决定是否进入 M3–M6。逐步状态、GPU、恢复命令和配置哈希见 `experiment_queue.json`。

病例级置信度表已另行生成：`image_confidence.csv`，按原 Pairwise Hard Dice 置信度 0.4/0.6 融合，并将原始空预测病例置零；101 行中有 20 行为零。Dynamic-hard 缓存构建器也增加了完整单病例缓存复用逻辑，便于缓存阶段中断后接续。

## 更新：iteration 300/500

M1 iteration 300 的验证 Dice=0.3090（iteration 250 的最高值仍为0.3103），空预测降至3例；相对初始 source model，12例提升超过0.02，10例下降超过0.02。初始 Dice 三组均值为0.0613、0.3301、0.7455。41例表面距离有限，有限 HD95/ASD 均值为68.44/45.64 mm，3例空预测对应 inf。

M0 对照 iteration 500 的 Dice=0.2977，空预测9例；11例提升超过0.02、10例下降超过0.02。三组均值为0.0326、0.3148、0.7692。35例表面距离有限，有限 HD95/ASD 均值为56.23/33.84 mm。以上是训练中期节点，不能据此定胜负。机器可读逐例及汇总输出已更新为 `validation_per_case.csv` 和 `validation_summary.csv`。

At the latest checkpoint review M0 reached iteration 550 with mean Dice=0.301292 (its current best); M1 was at iteration 330, current best still iteration 250 Dice=0.310338. Both are nearing their first full-case coverage and configured Dynamic-hard refresh.

## 更新：iteration 350/600（进行中）

M1 已完成 iteration 350 的训练步，iteration 350 验证正在进行；截至上一完整验证 iteration 300，最高 Dice 仍为 0.310338（iteration 250）。iteration 340 的分支损失/梯度范数为 High 1.2196/77.03、Low 2.1734/357.12，表明单步梯度贡献有明显波动，需要结合完整累计统计判断是否存在持续主导；不会据单步值单独下结论。

M0 对齐对照已完成 iteration 600 验证，mean Dice=0.303031、mean IoU=0.228243，保存了 iteration 600 当前 best checkpoint。M1/M0 的 101 例初始缓存均完成；双卡继续各运行一个独立进程，当前 GPU 利用率约 83%/93%。这些仍是训练中间节点。

M1 iteration 350 验证已完成：mean Dice=0.297499、mean IoU=0.223146，44例预测均非空；当前 best 仍为 iteration 250（Dice=0.310338）。这说明空预测减少与总体 Dice 改善并不等价，后续按固定预算及 best checkpoint 双口径评审。M0 iteration 600 的逐例文件显示空预测8例。M1 检查点中分支梯度累计统计含 NaN，尽管当前日志中的单步 High/Low 梯度范数可见有限值；该统计需要进一步溯源，不能用作已通过的梯度平衡证据。

M0 iteration 650 新 best mean Dice=0.305689、mean IoU=0.230138；M1 仍在 iteration 360。两项训练已覆盖 101 个病例中的 100 个，动态刷新尚未触发，预计最后一个未覆盖病例到达后开始。M1 当前 patience=2（best iteration 250）；M0 在 iteration 650 刷新了 best。M1 250–350 的逐 patch 梯度累计里 Low gradient norm 汇总为 NaN，但 iteration 260–360 的抽样日志均有限；后续核查需区分早期个别非有限梯度与统计累计问题，并检查 GradScaler 是否跳过对应更新。

## 更新：首次 Dynamic-hard 动态刷新

M1 在 iteration 374 完成首次 101/101 病例覆盖，正在用 Student snapshot 生成 version 1 缓存；23:25 已完成 6/101 例。M0 对照在 iteration 701 完成首次 101/101 覆盖，也开始刷新；M0 iteration 700 验证 mean Dice=0.311436、mean IoU=0.235442，为当前 best。两个刷新并行占用两张 GPU；M1/M0 version-0 缓存不变，version-1 按实验分别写入独立目录。

补充 optimizer step 审计：M1 iteration 350 检查点中所有 Adam 参数状态的 step 均为343，M0 iteration 700 均为694；对应检查点前累计有7次和6次未执行有效 optimizer step。旧进程的 trainer state 没有保存这个计数，因此从一致的 Adam 参数 step 还原；已修正 `finalize_hierarchical_training_summary.py`，在 trainer 计数缺失时使用 Adam 状态，而不再把训练 iteration 当作 optimizer step 数。M1/M0 version-1 缓存构建仍在继续。

## 更新：两项中期验证的统一汇总

机器可读汇总已由 `scripts/summarize_hierarchical_dynamic.py` 生成，分别保留 latest validated 和 best checkpoint。以同一 Zero-shot 验证集（44例，Dice=0.283467、平均预测体积2226.7 voxels）作离线参照：M1 iteration 350 Dice=0.297499、IoU=0.223146、平均预测体积5895.0，13例提升>0.02、15例下降>0.02、空预测0；其 best iteration 250 Dice=0.310338、平均预测体积2843.9，12例提升、9例下降、空预测6。M0 对齐对照 iteration 700 Dice=0.311436、IoU=0.235442、平均预测体积3747.0，14例提升、12例下降、空预测8；由于空预测导致本次44例 HD95/ASD 聚合均为 inf。以上验证训练预算不同且 M0/M1 仍运行中，暂不作方法胜负结论。逐病例体积变化与零预测变化已纳入统一 CSV。

按初始 Zero-shot Dice 分层的中期结果：M1 best iteration 250 三组均值为0.0508/0.3375/0.7635（低/中/高，n=22/10/12）；M1 iteration 350 为0.0721/0.3086/0.7014。M0 iteration 700 为0.0564/0.3278/0.7654。M1 的 iteration 350 虽不再有空预测，但高 Dice 组均值下降，说明需同时看病例分层、体积及距离指标。

截至 23:36，M1/M0 的动态 version-1 缓存分别完成38/101和30/101例；两个训练 PID 均存活，缓存继续写入，首轮刷新尚未结束。

截至 23:39，M1/M0 version-1 动态缓存分别完成47/101和39/101例；所有病例均为 Student snapshot 刷新，尚未切换到新版本继续训练。

截至 23:42，M1/M0 version-1 缓存完成55/101和46/101例，磁盘剩余175 GiB。两项刷新仍在独立目录中继续写入，没有检查点或配置覆盖。

M5 已补齐刷新专用的 benchmark Gaussian 协议实现：以0.5滑窗步长对 logits 做 Gaussian 融合后 sigmoid，基于 image/prompt 多视图 mask 的 Pairwise Hard Dice 更新 `C_case`，原始标准预测为空时置0。静态版本仍复用既有初始 PE 与病例置信度；M5 的 version-1 Student refresh 才使用 Gaussian 协议更新共识、PE 和病例置信度。质量协议写入缓存 metadata，并以版本化 JSON 提供病例置信度，避免训练时反复解压完整概率图。配置哈希已更新。M6 只列为条件门控阶段，待 M1–M5 结果确认确实有效后再决定是否执行。

M5 初始 version-0 缓存也已准备离线构建脚本：从 Stage1 已缓存的 image/prompt 概率图复算 benchmark Gaussian 一致的共识、PE、Pairwise Hard Dice 置信度及原始空预测状态，不调用模型。刷新后用独立 GT 诊断脚本比较所有101例的 Dice、体积和置信度/不确定性变化。该缓存会在 M5 依赖阶段启动前生成，避免与当前两项 GPU 刷新争用 CPU 写盘。

截至 23:54，M1/M0 version-1 缓存完成85/101和76/101例。两项进程未退出，最新日志均记录新病例写入。

## 更新：M1 动态刷新完成

M1 version-1 刷新完成101/101例，checkpoint `latest.pt` 已绑定 refresh_cycle=1、teacher_iteration=374，缓存 metadata 的病例清单与实际文件数均为101。M1 已恢复训练，等待 iteration 400 验证。M0 的 version-1 刷新达到95/101例，仍在 GPU1 构建。

M0 version-1 刷新现已完成101/101例，metadata 与病例覆盖完整；随后已恢复到 iteration 710。M1 iteration 400 的验证正在运行中，双卡当前 GPU 利用率约100%/94%，显存约8.6/8.9 GiB。

## 更新：M1 iteration 400 的刷新后退化

M1 iteration 400 mean Dice 降至0.276633、IoU=0.207724，仍未产生空预测，但平均前景体素升至19585.7（iteration 350 为5895.0；Zero-shot 为2226.7），Precision=0.2374、Recall=0.6420；初始低/中/高 Dice 三组均值为0.0981/0.2913/0.5917，best 仍在 iteration 250（Dice=0.310338）。在刷新后50个病例更新中出现明显过分割迹象。M1 patience=3，当前作为退化门控调查项；后续机制不应直接叠加，需进一步核实刷新前后共识目标体积和训练 GT 离线 Dice，判断 Dynamic-hard refresh 是否放大错误伪标签。

M0 对齐对照 iteration 750 mean Dice=0.316863、IoU=0.240749，更新为当前 best；当前 M1/M0 仍在训练。`validation_summary.csv` 和 `validation_per_case.csv` 已刷新为 latest/best 双口径。

M1 因 version-1 刷新后持续过分割，在 iteration 450 验证 Dice=0.093026（iteration 400 已为0.276633）后手动停止后续更新；iteration 450 latest 与 iteration 250 best checkpoint 均保留。M1 best 值仍高于 Zero-shot，但后续崩塌使固定预算稳定性不成立。

对 M1 Dynamic-hard version-0/1 的训练集缓存作离线 GT 诊断（仅分析，不用于训练）：version-0→version-1 共识阈值前景均值由2213升至6794 voxels，空共识预测37例降至1例；平均伪标签 Dice由0.3287微降至0.3234，平均 voxel confidence 仍接近饱和（0.9982→0.9892），置信度变化与 Dice 变化相关性近零（Spearman=-0.044）。结合 validation 400 前景均值5895、validation 400/450 Dice继续下降，当前证据支持刷新后伪标签体积扩张未带来质量提升，原 Dynamic-hard confidence 也没有识别这类错误扩张。此为算法机制退化证据，尚未证明存在几何或代码错误。

M2 已在 GPU0 从相同 VoxTell source 权重独立启动，完整复用预生成的101例 Dynamic-hard version-0 与固定 PE 缓存；iteration 0 验证 Dice=0.283467、IoU=0.217583，与共享 Zero-shot 完全一致。iteration 1 的空预测病例保留 KL loss，额外 Dynamic-hard BCE 为0，符合保护条件。M0 iteration 950 新 best Dice=0.320854、IoU=0.242098，仍在 GPU1 训练。

## 更新：M2 梯度告警核查中

核实当前进程绑定：M2 在物理 GPU0（`CUDA_VISIBLE_DEVICES=0`），M0 在物理 GPU1（`CUDA_VISIBLE_DEVICES=1`）；两张卡均有一个训练进程，没有显存填充进程。M0 已到 iteration 980，iteration 950 是当前 best（Dice=0.320854）。

M2 到 iteration 20 时累计记录多次 High 分支非有限梯度告警，训练 loss 本身仍有限，日志显示有限的梯度范数（如 iteration 20 High=5.009、Low=0.00557）。检查确认固定 PE 权重是有限值并处于[0,1]；分支梯度由 AMP GradScaler 缩放后通过 `autograd.grad` 获取，非有限值会进入累积梯度并由后续 scaler unscale/step 检查，理论上应触发跳过该 optimizer step 和降低 scale。当前尚无 iteration 50 检查点，因此无法从训练状态确认发生次数及实际 optimizer step 数；保留 M2 运行以取得首个检查点后再判断是否为可恢复的 AMP overflow 或持续性数值异常。没有改写运行中的模型或训练配置。

截至 00:27，M2 到 iteration 30，High 分支非有限梯度告警累计7次；日志仍显示有限的标量 loss 与梯度范数，最新为 iteration 30 loss=2.1193、High grad norm=0.0902、Low grad norm=0.0759。iteration 50 前尚无训练状态快照，暂不判定告警对应的 optimizer skip 数。M0 到 iteration 1010；iteration 1000 验证 Dice=0.316503、IoU=0.238457，best 仍为 iteration 950 Dice=0.320854。

M2 iteration 50 checkpoint 已审计：验证 mean Dice=0.285859、IoU=0.219544，为当时 best，略高于 Zero-shot；50个病例更新中 Adam 实际 step=44，6次被 GradScaler 跳过。scaler scale 从初始65536回退至1024（growth tracker=15）；High 梯度非有限计数7、Low=0。由此确认告警与 AMP 溢出/跳步相符，而非 loss 非有限或固定 PE 权重异常；M2 继续运行，观察 scaler 是否稳定。该检查点的 high_gradient_norm 累计量为 NaN，是因为非有限梯度范数被直接累加；将为后续汇总增加 finite-only 统计，原始告警计数和 optimizer step 作为准确审计依据。

截至 00:36，M2 已到 iteration 100，iteration 50 之后未新增非有限梯度告警（累计仍为 High=7、Low=0）；M2 进程持续运行，等待 iteration 100 验证/检查点落盘。M0 iteration 1100 验证 Dice=0.315849、IoU=0.237563，best 仍为 iteration 950；M0 当前 iteration 1130。双卡利用率约89%/98%，显存约8.7/8.9 GiB。统一 validation summary 已重生成。

M2 iteration 100 验证 mean Dice=0.293449、IoU=0.224753，新 best，较 Zero-shot 提高0.009982。累计 optimizer steps=94/100，AMP scale 保持1024且 growth tracker=65；非有限 High 梯度仍为7（没有新增），Low=0，说明早期溢出经过 scaler 回退后已稳定。M2 固定 PE 目前无坍塌迹象，继续完整预算；统一 summary 已更新。

M0 iteration 1150 验证更新为新 best：mean Dice=0.325198、IoU=0.246429。当前运行至 iteration 1170。M2 最新训练日志到 iteration 110，iteration 100 best=0.293449 未变，训练继续。统一汇总已同步到当前验证结果。

M0 iteration 1200 验证 Dice=0.330970、IoU=0.252063，新 best；M2 到 iteration 130，iteration 100 best 仍为 Dice=0.293449，继续训练。M2 AMP overflow 计数仍为早期7次，后续暂无新告警。summary/queue 已刷新。

M2 iteration 150 验证 Dice=0.305852、IoU=0.232214，新 best，已较 Zero-shot 提高0.022385；iteration 50 后未出现新非有限梯度告警。M0 iteration 1250 验证 Dice=0.325172、IoU=0.246311，较 iteration 1200 略低，best 保持 iteration 1200 Dice=0.330970。统一 validation summary 与队列状态已更新。

M2 iteration 200 验证 Dice=0.305604、IoU=0.231880，较 iteration 150 best（0.305852）低0.000248；训练趋势目前稳定，AMP 告警无新增。M0 iteration 1350 验证 Dice=0.315433，best 保持 iteration 1200 Dice=0.330970。两项运行正常，统一 summary 更新。

M2 iteration 250 checkpoint 的验证 Dice=0.309401、IoU=0.234208，新 best；较 Zero-shot 提高0.025934。检查点 scaler scale=512，High/Low 非有限梯度计数为7/1，optimizer step=243/250（7次 AMP 跳步），Adam 参数状态与计数一致。早期 overflow 已由 GradScaler 降低 scale 后控制；近期仍有偶发单次 overflow，但训练在继续且验证持续改善，暂不修改训练超参数。M0 已到 iteration 1500，待验证。

M0 iteration 1500 验证 Dice=0.338918、IoU=0.259374，新 best。M2 已到 iteration 280，best 保持 iteration 250 Dice=0.309401。M0 与 M2 均未触发 early stopping，继续完成固定预算；统一汇总已刷新。

M2 iteration 300 验证 Dice=0.307273、IoU=0.232047，低于当前 best iteration 250 的0.309401；仍高于 Zero-shot，需观察后续是否稳定。M0 iteration 1550 验证 Dice=0.331123，best 保持 iteration 1500 的0.338918。两个训练进程正常，summary/queue 已更新。

M2 iteration 350 验证 Dice 降至0.286789、IoU=0.211989，低于 Zero-shot 约0.003；best 仍为 iteration 250 的0.309401。该回撤幅度已值得关注，但还不足以判定坍塌（训练集无 GT 参与），继续监控至至少下一验证点，并与病例分层/体积变化核对。M0 iteration 1650 取得新 best Dice=0.340715、IoU=0.259986。两进程仍在训练，summary 更新。

## M2 实现缺陷修复与重跑

对 iteration 350 的退化作实现审计后发现：`original_empty` 已从 PE 缓存读取，但 Low 分支 Dynamic-hard BCE 原先只检查共识是否含前景，没有检查病例原始预测是否为空。因此原始空预测病例可能被非空共识目标额外监督，违反 `1.md` 的明确门控要求。M2_pre_empty_gate 作为正式但实现无效的运行完整保留到 iteration 350：best iteration 250 Dice=0.309401；iteration 350 Dice=0.286789，平均预测体积9536 voxels，空预测0例。该结果不作为有效 M2 结论。

已在 Low 分支加入 `original_empty` 门控：空预测病例的 Dynamic-hard BCE 置为零，仅保留多视图 KL consistency；非空病例仍使用固定 PE 权重。训练逻辑编译检查通过。旧目录及其 Dynamic-hard 101例缓存被保留并复用，不重新推理；修正后的 M2 从相同 VoxTell source 权重独立重跑，输出写入干净目录 `outputs/hierarchical_dynamic/M2`，位于物理 GPU0。M0 在 GPU1 继续运行。新配置 SHA256=17805371ee1e85ba9ead163a2b47b3bc1da5a337e4285ab711b29f912a32ff69。

修正后 M2 iteration 0 验证 Dice=0.283467、IoU=0.217583，与共享 Zero-shot 完全一致；iteration 1 空预测病例仍保留 KL loss，Dynamic-hard BCE=0，确认新增门控按预期生效。M0 iteration 1700 Dice=0.338944、IoU=0.258215，best 仍为 iteration 1650 Dice=0.340715。summary 和队列状态已更新。

修正后 M2 iteration 50 验证 Dice=0.285859、IoU=0.219544，为当前 best；早期梯度告警4次，AMP scaler 正在回退（待 checkpoint 状态核对）。M0 iteration 1800 Dice=0.336440，当前 best 仍是 iteration 1650 的0.340715。修正后 M2 没有复现前一版本到350后明显体积膨胀的前段行为，继续按固定预算监控。

更正 M2 iteration 50 梯度审计：checkpoint 的累计计数为 High 非有限梯度7次、Low 0次，optimizer step=44/50，scaler scale=1024；之前按短日志片段估计的告警次数不完整，以 checkpoint 状态为准。M0 iteration 1800 验证 Dice=0.336440、IoU=0.255771。

M0 aligned control 已完成固定预算 iteration 2000：latest Dice=0.331105、IoU=0.250840；best iteration 1650 Dice=0.340715、IoU=0.259986；optimizer steps=1994（6次 AMP skip）。M2 修正版到 iteration 150，best Dice=0.305852，继续训练。

GPU1 释放后已启动 M3-control（PID 2858231，独立 source 初始化），配对配置与 M3-discount 的 seed、数据、损失、预算和 B=2 梯度累积设置一致；两者 cache 均确认覆盖101例。M3-control 在 GPU1、M2 在 GPU0 并行运行。M3-discount 待下一 GPU 空闲后启动。

M3-control iteration 50 已完成验证，Dice=0.285071、IoU=0.218885。保存 best checkpoint 后，训练摘要写 CSV 因字段列表遗漏 `discount_sum` 退出。已在 `case_gradient_contributions.csv` schema 加入该字段并通过编译检查；将该错误保留在 `outputs/hierarchical_dynamic/M3-control/iteration_50_resume_error.txt`，并从 iteration-50 best checkpoint 恢复成功。新 PID=2887341，在 GPU1 继续。M2 修正版已到 iteration 250，best Dice=0.309401，继续训练。

修正后 M2 iteration 300 验证 Dice=0.307273、IoU=0.232047，best 保持 iteration 250 Dice=0.309401；从 iteration 50 到300的前景体积趋势待持续审查。M3-control 从 iteration50恢复后已运行至iteration90，梯度贡献日志恢复正常写入；仍未到下一验证点。

M3-control 恢复后 iteration 100 验证 Dice=0.290453、IoU=0.223314，新 best；高于 iteration 50（0.285071）和 Zero-shot。M2 当前到 iteration 310，best 仍为 iteration 250（0.309401）。summary/queue 已刷新。

M2 iteration 374 启动 version-1 Dynamic-hard 缓存刷新；刷新使用 Student snapshot 生成共识，当前约完成13/101例，训练等待缓存完成。固定 PE 未刷新。M3-control 已完成 iteration 200 验证，待读取均值并更新队列。

M3-control iteration 200 验证 mean Dice=0.302903、IoU=0.230885，新 best；相较 Zero-shot 提高0.019436。M2 refresh version-1 缓存约完成18/101例，GPU0 仍在 Student snapshot 推理刷新；M3-control 在 GPU1 继续至iteration210。summary已更新。

截至 02:30，M2 的 Student snapshot refresh version-1 缓存完成33/101例；M3-control 已到 iteration250（当前 best Dice=0.302903 @200）。两项均正常运行，双卡各一个进程。

M3-control iteration250 验证 Dice=0.306095、IoU=0.232839，新 best；相较 Zero-shot +0.022628。M2 version-1 refresh 缓存完成36/101例，刷新进程继续。Summary/queue 更新。

M3-control iteration300 验证 Dice=0.310622、IoU=0.235728，新 best。M2 version-1 refresh 进度58/101；等待刷新后恢复训练并观察iteration400。summary/queue已更新。

M3-control iteration350 验证 Dice=0.319616、IoU=0.241754，新 best。M2 version-1 动态缓存完成74/101例。两进程保持活跃，summary/queue更新。

## M2 修正版退化门控结论

修正空预测门控后，M2 在 iteration 400 验证 mean Dice=0.179595、IoU=0.131829；best 仍为 iteration250 Dice=0.309401。均值预测体积从 Zero-shot 2227 voxels 增至 iteration350 的9536，再在动态共识刷新后 iteration400 增至78664 voxels；空预测从13降至0，Precision降至0.1438、Recall升至0.7940。M2 的 iteration 350 已在刷新前低于 Zero-shot，iteration400 刷新后进一步过分割，符合明显退化判据。iteration400 latest checkpoint 保留，当前停止此 run，训练状态 optimizer steps=390；best/latest、逐病例指标与训练日志完整保留。此结果显示固定 PE 加权和现有 Dynamic-hard 训练组合在本实现/设置下不稳定，后续需依据 M3-control 的独立结果决定是否继续 image discount 对照；暂不启动 M4/M5/M6。

说明：M2 iteration400 optimizer steps 经最终 checkpoint 审计为392（训练 iteration 计数与有效 step 不同），已更新机器记录。M3-control 在刷新前 best Dice=0.319616（iteration350），高于M2 best/Zero-shot；为形成同 seed、相同 B=2 累积方式的配对比较，M3-discount 已从源权重独立启动在物理 GPU0（PID2937715），M3-control 在 GPU1继续。该对照用于检查病例级 discount 是否能抵消基础 PE/Dynamic-hard 设置的体积扩张风险；其结果出炉前不继续 M4/M5/M6。

M3-discount iteration50 验证 Dice=0.285705、IoU=0.219443，略高于Zero-shot 0.283467和M3-control iteration50 0.285071；样本级 discount 在日志中有实际缩放（空预测病例 d=0）。仍属早期，继续成对观察。M3-control version-1 refresh 当前70/101例。


M3 成对运行进展（2026-11-27 03:22 Asia/Shanghai）：M3-control 已在 iteration 374 触发 Student-snapshot Dynamic-hard version-1 刷新，缓存完成 85/101 例；其刷新前 best 仍是 iteration 350 Dice=0.319616。M3-discount 到 iteration 110，iteration 100 验证 Dice=0.291327、mIoU=0.223780，较 Zero-shot 0.283467 高 0.007860；iteration 50 Dice=0.285705。此时 control 和 discount iteration 100 验证差异很小（control=0.290453，discount=0.291327），尚不足以支持 discount 改善，须等待匹配的后续验证和刷新后表现。两个进程分别占用物理 GPU1（control）与 GPU0（discount），各约 8.5–8.7 GiB 显存；没有为占显存而申请额外张量。


更新 M3 运行进度（2026-11-27 03:29 Asia/Shanghai）：M3-control 完成 iteration 374 的 Student-snapshot version-1 缓存刷新（101/101例），当前恢复训练，刷新后验证尚未完成。M3-discount iteration 150 验证 Dice=0.301541、mIoU=0.230222，为新 best；从 Zero-shot 提升0.018074，且 iteration50/100/150 连续改善。control 的刷新前 best 是 iteration350 Dice=0.319616，但这不是同 iteration checkpoint 对照；继续等 control iteration400 验证后，再决定 M4 是否有依据启动。


M3-control 首次刷新后验证（2026-11-27 03:33 Asia/Shanghai）：Student-snapshot version-1 刷新后 iteration400 mean Dice=0.319280、mIoU=0.242722；刷新前 iteration350 Dice=0.319616、mIoU=0.241754。Dice 仅低0.000337，未观察到类似 M2 的即时坍塌。44例验证平均预测体素由2939增至3293（+12.0%），空预测由6降至3；mean precision=0.3576、recall=0.4226。control best 保持 iteration350。该结果说明相同 M3-control 设置的单轮刷新即时影响有限，但仍需继续确认后续稳定性。M3-discount 最新验证仍为 iteration150 Dice=0.301541，训练在 iteration180。


M3-discount iteration200 验证 Dice=0.308105、mIoU=0.234410，为新 best；相较 iteration150 的0.301541继续改善，较 Zero-shot +0.024638。当前 M3-control 已在刷新后运行到 iteration430，最近验证 iteration400 Dice=0.319280；M3-discount 仍未触发 refresh（默认 trigger iteration374），需等待动态缓存刷新后进一步比较。


M3-discount iteration250 验证 mean Dice=0.312851、mIoU=0.237436，新 best，较 iteration200 再升0.004746，较 Zero-shot +0.029384。mean predicted voxels=2671.8，空预测9例，mean precision=0.3696、recall=0.3804；iteration200 时分别为2588.9 voxels、11空预测。M3-control 最近 iteration450 Dice=0.315013、best仍为iteration350 Dice=0.319616，iteration500验证尚待完成。discount 当前 best 已接近 control 的 best，但需等同 iteration 和触发刷新后验证再判断。


M3-control iteration500 验证 mean Dice=0.312801、mIoU=0.237665；best 仍为 iteration350 的0.319616。平均预测体积=5191.1 voxels，空预测1例，mean precision=0.3222、recall=0.4393；相较 iteration400 的3293.3 voxels/3空预测，继续扩大且召回上升。M3-discount iteration250 的 Dice=0.312851、预测体积=2671.8、空预测9例，平均精度更高、召回较低。该趋势符合折扣正在压低较低置信病例梯度的可能机制，但迭代点未匹配且 discount 尚未刷新，不能作因果结论。


M3-discount iteration300 mean Dice=0.315807、mIoU=0.239161，新 best；mean predicted voxels=2765.7、空预测8例，precision=0.3657、recall=0.3863。已有 M3-control iteration300 Dice=0.310622，discount 暂高0.005185；但 control 是 iteration50 崩溃后由 best@50 恢复，可能存在 RNG/数据序列恢复差异，因此这是探索性趋势而非完全严格的 paired seed 结果。M3-control 当前 iteration540，最新已验证 iteration500 Dice=0.312801。M3-discount 到 iteration300，动态刷新触发点尚未到。


M3-control iteration550 mean Dice=0.321396、mIoU=0.245934，新 best。平均预测体积8935.8 voxels，空预测1例，precision=0.3018、recall=0.5260；相较 iteration500 Dice+0.008595但体积扩大约72%，属过分割风险信号，尚未出现指标骤降。M3-discount 在 iteration340，best iteration300 Dice=0.315807，即将触发 iteration374 version-1 刷新。


M3-discount iteration350 mean Dice=0.321696、mIoU=0.242812，新 best；mean predicted voxels=2975.8，空预测6例，precision=0.3542、recall=0.4039。与 M3-control 同轮 iteration350 Dice=0.319616，相差+0.002080；两者初始预测体积区间也近似，但后续 control volume 上升至 iteration550 的8935.8 voxels。discount 当前达到刷新触发前 validation checkpoint，iteration374 后将以 Student snapshot 刷新。


M3-control iteration600 出现显著过分割退化信号：mean Dice=0.290121（相对 iteration550 best 下降0.031275），mIoU=0.218427；平均预测体积17161.9 voxels（iteration550 8935.8），空预测由1降到0，mean precision=0.2546、recall=0.6103。latest checkpoint 已保存到 iteration600，当前继续到下一验证点以确认趋势；若持续快速恶化，将按 1.md 原则停止该分支并保留 best/latest 及指标。与此同时 M3-discount 于 iteration374 开始 Student-snapshot Dynamic-hard version-1 缓存刷新，缓存生成中0/101，检查点 best@350 Dice=0.321696。


M3-control iteration650 确认严重过分割坍塌：Dice=0.167391，mIoU=0.122340，平均预测体积66465.4 voxels，空预测0例，precision=0.1367、recall=0.7663。相比 best@550 Dice=0.321396，下降0.1540；预测体积由8935.8扩大到66465.4。iteration650 latest checkpoint 与 best@550 已保留，iteration660 处于刷新缓存读取中时经 Ctrl-C 停止（无新的参数更新），不再继续 M3-control，以遵守失败机制先定位、不叠加的原则。此次过分割在 M3-control 的固定 image-level 权重 d=1 下复现；M3-discount 当前 Student-snapshot version-1 刷新约30/101例，接下来验证 discount 能否抑制该风险。


截至 04:16，M3-discount version-1 Student-snapshot 缓存已完成41/101例；GPU0 正运行该刷新，GPU1 在 M3-control 坍塌确认后释放。未额外运行不相关训练以避免干扰 M3-discount 的刷新后评价。


M3-control version-1 刷新 GT 离线审计完成（101例，`analysis/hierarchical_dynamic/M3-control_refresh_audit/`）：共识 mean Dice 从0.328676升至0.346803（+0.018127），平均共识预测体素从2213.2升到3088.7（+39.6%），空共识从37降至16；平均 voxel confidence 0.99817→0.99668。置信度变化与 Dice 变化 Pearson r=-0.146、Spearman r=-0.069，均无明显相关。刷新缓存本身质量未下降，但仍有温和前景体积扩张，随后 control Student 体积于 iteration550–650持续扩大并坍塌，说明错误/偏大的伪标签监督可能被训练放大，而 confidence 不足以辨别可靠度。此处 GT 仅离线审计，未参与训练或缓存构造。M3-discount version-1 缓存构建至49/101例。


M3-discount 动态缓存刷新截至 04:26 完成72/101例，仍在 GPU0 执行；刷新后 checkpoint iteration374 后续验证待缓存完成。M3-control 已终止于 iteration660（latest保存iteration650），离线刷新审计报告与逐例CSV已落盘。


截至 04:32，M3-discount version-1 缓存刷新87/101例，预计剩余14例；GPU0 运行稳定，control 已在 iteration650 终止并保留失败正式结果。


M3-discount version-1 刷新完成，Student-snapshot 缓存101/101例。刷新后 iteration400 Dice=0.319633、mIoU=0.242756；刷新前 iteration350 best Dice=0.321696。iteration400 平均预测体积3247.3 voxels、空预测4、precision=0.3596、recall=0.4212。M3-control 同轮 iteration400 Dice=0.319280、体积3293.3；discount 略高0.000353且体积/空预测相近。M3-control 后续 iteration650 Dice坍至0.167391、预测体积66465.4；M3-discount 刷新后目前 iteration410，继续观察是否抑制后续体积扩张。


M3-discount version-1 缓存 GT 离线审计完成（101例，`analysis/hierarchical_dynamic/M3-discount_refresh_audit/`）：共识 mean Dice 从0.328676升至0.347661（+0.018985），平均共识预测体素2213.2→3059.7（+38.3%），空共识37→17例；平均体素置信度0.99817→0.99673。置信度变化与 Dice 变化 Pearson r=-0.158、Spearman r=-0.057，均无明显关系。刷新增幅与 control 很接近，说明 discount 并未改变共识刷新质量；若其后续 Student 体积稳定，改善应来自病例梯度抑制与更新贡献重分配。M3-discount 目前 iteration420，刷新后 iteration400 Dice=0.319633，持续训练。


M3-discount iteration450 mean Dice=0.313785（best@350低0.007911），平均预测体积3685.0 voxels、空预测3例、precision=0.3457、recall=0.4215；相对刷新后 iteration400 体积+13.5%，Dice-0.00585。尚未复现 M3-control iteration550–650 的快速膨胀幅度，继续运行至至少 iteration600 观察稳定性。


M3-discount iteration500 mean Dice=0.312164，best@350低0.009532；平均预测体积4768.6 voxels（iteration400 3247.3，+46.8%），空预测1例，precision=0.3241、recall=0.4341。出现缓慢体积扩张及指标回落，但尚未达到 M3-control 迭代400→600 的快速失控；继续到 iteration600 后再决定是否推进 M4。


M3 逐病例梯度贡献审计快照（control 累计至 iteration650；discount 累计至 iteration500，见 `analysis/hierarchical_dynamic/M3_gradient_contribution_snapshot.json` 和原始逐例CSV）：discount 组每次病例机会的梯度范数 after/before 汇总比为0.560，样本加权平均 d=0.537；500次病例机会中 d=0 的记录89次，零梯度记录102次。Control 全程 d=1，梯度范数缩放比1.0。两组累计训练长度不同，不直接比较总贡献绝对值；该结果验证 discount 确实削减了病例梯度，但其对最小化体积漂移的效果仍需结合后续 Dice/volume 曲线判断。


M3-discount iteration550 mean Dice=0.323672、mIoU=0.248132，新 best；相较 M3-control 同轮 Dice=0.321396 高0.002276。discount 平均预测体积7322.5 voxels，低于 control 的8935.8（约18%），空预测均1例；但相较 discount iteration400 体积已增126%，并有Precision下降/Recall上升，故仅能认为部分缓和而非消除过分割。继续观察 iteration600 后再决定 M4。


M3-discount iteration600 mean Dice=0.309432、mIoU=0.235059，best仍为 iteration550 Dice=0.323672；mean predicted voxels=13232.8、空预测0、precision=0.2759、recall=0.5822。相较 M3-control 同轮 Dice=0.290121、预测体积17161.9，discount 高0.01931且预测体积低约23%，说明 gradient discount 部分缓和但没有消除过分割趋势。基于 M3-discount iteration350/550 的正向指标、与 control 的同轮差异及未复现 66k voxels 级坍塌，启动 M4 探索性实验：在物理 GPU1 由原始 source 独立初始化，开启刷新后的 Dynamic-hard consensus + PE 权重；不继承 M3 checkpoint。M3-discount 在 GPU0 继续，以完成其后续轨迹。M4 PID/session 尚待启动日志确认，命令与 session 68835 已记录。


M4 已于 2026-11-27 05:13 在物理 GPU1 启动，PID 3034530，配置哈希保持 queue 中记录值。复用既有 frozen-source version-0 Dynamic-hard cache（101例，完整配置/来源签名校验通过），未重复推理；Student 从原始 VoxTell source 独立初始化，iteration0 Dice=0.283467 与共享 Zero-shot 一致；iteration1 已见 d=0 空预测病例的梯度为0。M4 到 iteration1，出现1次 High 非有限梯度警告，继续监控。M3-discount 在 GPU0 到 iteration620，latest validation@600 Dice=0.309432、best@550=0.323672。


M5 准备检查：工作区已有 M5 version-0 cache（101例），但 metadata 是旧协议（refresh tile step=1.0 且无 quality_protocol），与当前 M5 配置要求的 Gaussian benchmark 协议（tile step=0.5）不兼容。尚未启动 M5；须在到达 M5 阶段时用 `scripts/prepare_hierarchical_m5_cache.py` 从现有 benchmark probabilities 和固定 PE cache 重新生成/验证，旧缓存需保留作审计。


M3-discount 于 iteration680 停止（latest iteration650）：iteration650 mean Dice=0.235956，mIoU=0.172153，平均预测体积28040.9 voxels，空预测0，precision=0.1924、recall=0.6931；相对 best@550 Dice=0.323672 下降0.0877，低于 Zero-shot。M3-discount 的 best@550 和 latest@650 已保留，训练进程因确认持续过分割后 Ctrl-C 停止（未执行后续参数更新）。梯度 discount 将坍塌从 control 的 iteration600 Dice0.290/volume17k 延后到 iteration650 才跌至0.236/volume28k，属于部分缓和，不是有效解决。

M4 iteration50 验证 Dice=0.285705、mIoU=0.219443；iteration0 为0.283467。iteration50 checkpoint 统计 High 非有限梯度11次、Low 0次，Adam optimizer steps=19/50，gradient batches skipped=6，AMP scale=1024。原始 validation 逐例CSV已保存，继续监控 scaler/验证趋势再决定是否停止。


M4 iteration100 mean Dice=0.291327、mIoU=0.223780，新 best；相较 Zero-shot +0.007860，预测体积2390.8 voxels、空预测12例，mean precision=0.3593、recall=0.3533。checkpoint审计：High/Low 非有限梯度12/0，Adam steps=42/100，8个梯度累积批次因AMP跳过，scale从1024降至512；数值不稳定较明显但 scaler 已回退，训练仍继续。此时 M4 和 M3-discount 在 iteration100 指标相同，因动态PE要到refresh周期后才生效，符合预期。


M4 iteration150 mean Dice=0.301541、mIoU=0.230222，新 best，较 iteration100 +0.010214；High/Low 非有限梯度13/0，较iteration100只新增1次，Adam steps=66/150，9批次跳过，AMP scale=512。数值告警经 GradScaler 回退后趋稳，继续运行。M4 尚未到 Dynamic-hard/PE 刷新触发点（iteration374）。


M4 iteration200 mean Dice=0.308105、mIoU=0.234410，新 best；mean predicted voxels=2588.9、空预测=11，precision=0.3734、recall=0.3744。相较 Zero-shot +0.024638，训练上升趋势延续，动态 PE 尚未刷新（迭代触发374）。


M5 缓存准备首次运行发现固定 PE 图存储为 padded geometry，概率图则是 ROI geometry；已修正 `scripts/prepare_hierarchical_m5_cache.py` 按 metadata 的 `roi_slicer` 裁剪 padded PE 权重，并增加几何不匹配诊断。未改任何原始输入或训练代码。旧 M5 version-0 缓存移至 `outputs/hierarchical_dynamic/M5/dynamic_hard_cache_legacy_20261127` 保留；修正后从 cached probabilities 重新准备 Gaussian benchmark version-0，当前写入14/101例，CPU进程3061823。M4 仍在 GPU1，latest validation@200 Dice=0.308105；GPU0保持空闲，尚无下一阶段运行依据。


M5 初始 cache 准备完成，101/101例逐病例 NPZ 存在且抽查权重/掩膜几何与 metadata padded geometry 一致；version metadata 为 `quality_protocol=benchmark_gaussian`、`refresh_tile_step_size=0.5`、`teacher_type=frozen_source`。由脚本使用预存 Stage1 benchmark probabilities、固定 PE 权重生成；无训练或新推理。M5 仍等待 M4 阶段审查后再启动。


M4 iteration250 mean Dice=0.312851、mIoU=0.237436，新 best；预测体积2671.8 voxels、空预测9例、precision=0.3696、recall=0.3804；较 Zero-shot +0.029384。此时仍未到刷新触发点，数值应与 M3-discount 刷新前轨迹相同；动态 PE 的效果要从 iteration400 后检验。


M4 iteration300 mean Dice=0.315807、mIoU=0.239161，新 best；mean predicted voxels=2765.7、空预测=8例。距 iteration374 动态共识+PE 刷新约74次迭代。到此刷新前曲线和 M3-discount 一致；评估 dynamic PE 需等刷新后的 iteration400+。


M4 iteration350 mean Dice=0.321696、mIoU=0.242812，新 best，与 M3-discount iteration350 相同；动态 PE 尚未生效。距固定刷新触发 iteration374 约24步。


M4 于 iteration374 完成 101例覆盖并开始 Student-snapshot version-1 刷新。当前 refresh cache 2/101，Student checkpoint 为 best@350 Dice=0.321696。刷新会一并生成 benchmark 匹配协议下的 PE 权重，完成后评估 iteration400。


M4 Student-snapshot version-1 刷新至12/101例，GPU1 进程持续运行；M5 Gaussian benchmark version-0 缓存已准备并通过 metadata/geometry 校验，M5训练暂未启动，等待 M4 刷新后审查。


截至 06:24，M4 Student-snapshot version-1 refresh 31/101例；训练保持等待缓存完成。M5 正式 Gaussian cache 101例已备好；M4 刷新后验证前不启动 M5。


M4 version-1 Student refresh 截至06:28为41/101例。best@350 checkpoint 保留，训练仍等待缓存完成；M5 新 benchmark_gaussian version-0 cache 与 legacy backup 均保留。


M4 Student-snapshot version-1 cache 59/101例；仍在 GPU1 进行。当前阶段无其他训练任务可并行，不人为占用 GPU0。


M4 version-1 cache refresh 截至06:40达75/101例；cache generation仍在GPU1运行，GPU0空闲。此等待阶段未额外启动无依赖不明确的训练。


M4 version-1 刷新离线审计（101例，`analysis/hierarchical_dynamic/M4_refresh_audit/`）：初始→刷新后 mean case confidence 0.5338→0.5566（+0.0228）；GT-only mean Dice 0.3382→0.3422（+0.00403）；预测体积均值2877.5→3628.4（+26.1%），空预测20→10（3例新空、13例恢复）。confidence change 与 GT Dice change Pearson r=0.407（p=2.4e-5）、Spearman rho=0.282（p=0.0043），为中等正相关；contrast M3-control/discount 的 refresh audit 相关性接近0。平均 pixel weight 0.99834→0.99491（差=-0.00343）。GT 仅用于离线诊断。刷新已完成101/101，M4 已恢复训练，将观察 Student iteration400 验证。


M4 iteration400（version-1 动态PE生效后）mean Dice=0.319150、mIoU=0.242271；best仍为iteration350 Dice=0.321696（下降0.002546）。预测体积3263.2 voxels vs iteration350 2975.8（+9.7%），空预测4 vs6；precision=0.3589、recall=0.4210。和 M3-discount iteration400 Dice0.319633 接近，尚未显示动态PE额外收益，但刷新后没有快速恶化。checkpoint审计High/Low非有限梯度13/0、optimizer steps188/400、AMP跳步12、scale256。继续观察至550/600。


M4 iteration450 mean Dice=0.312216（best@350低0.009480），mIoU=0.237089；预测体积3817.7 voxels、空预测3例、precision=0.3431、recall=0.4226。相较 iteration400，体积+17.0%，Dice-0.00693；属温和漂移，未达到 M3 过分割坍塌强度，继续观测。


M4 iteration500 mean Dice=0.307948（best@350低0.013748）、mIoU=0.233478；mean predicted voxels=5203.3、空预测1、precision=0.3197、recall=0.4332。相较 post-refresh iteration400 体积+59.5%，Dice-0.01120，当前趋势与 M3-discount 类似，暂未看到动态PE带来改善。继续到iteration600确认风险；在确认 M4 无有效增益前不启动 M5训练。


M4 iteration550 mean Dice=0.319126、mIoU=0.243693，较 iteration500 回升0.011179，best仍@350=0.321696；预测体积8334.9 voxels、空预测0、precision=0.3016、recall=0.5151。体积继续扩大但 Dice 回升，且其体积低于 M3-control@550（8935.8），与 M3-discount@550（7322.5）接近；需以 iteration600 检查是否稳定。


M4 iteration600 mean Dice=0.296832、mIoU=0.224107（best@350低0.024864），平均预测体积15703.8 voxels、空预测0，precision=0.2608、recall=0.6038。相较 iteration550 体积+88.3%、Dice-0.022294；与 M3-discount 同轮 iteration600 Dice0.309432、体积13232.8相比更差。动态PE未抑制体积漂移，先继续至650确认是否坍塌，再决定终止和是否不进入M5。


M4 iteration650 确认过分割坍塌：mean Dice=0.208088、mIoU=0.150699，平均预测体积36885.0 voxels、空预测0、precision=0.1669、recall=0.7252；best@350 Dice=0.321696下降0.11361，低于 Zero-shot。latest@650保留，Adam steps=308/650，17个累积批次AMP跳过，High/Low非有限梯度13/0，scale256。M4 已停止，不继续将无效动态PE机制带入下一轮。

虽然 M4 的 dynamic PE 未改善训练后期稳定性，但其刷新离线审计显示 Student refresh 的 case-confidence change 与真实 Dice change 中度相关（r=0.407，rho=0.282），为 M5 动态 image confidence 提供独立的可检验依据。故启动 M5 探索性对照：从源权重独立初始化，在 M4 机制上同时刷新 image-level confidence，并使用已验证的 Gaussian benchmark cache；不继承 M4 checkpoint。初始 M5 cache 101例、quality_protocol=benchmark_gaussian、tile step0.5，legacy cache 保留。


M5 已在 GPU0 推进到 iteration30；0–30 期间 High 非有限梯度警告8次、Low无告警（暂待 iteration50 checkpoint 读取 scaler/optimizer 计数），最近十余步未新增 warning，继续观察。


M5 iteration50 mean Dice=0.285588、mIoU=0.219341，为新 best；相较 Zero-shot +0.002121。预测体积2270.6 voxels、空预测13、precision=0.3669、recall=0.3411。checkpoint High/Low 非有限梯度11/0，Adam steps19/50，6个累积批次AMP跳过，scale1024；早期梯度数值问题仍在，继续观察后续是否稳定。


M5 iteration100 mean Dice=0.291277、mIoU=0.223734，新 best；较 Zero-shot+0.007810，与 M3-discount/M4 iteration100几乎相同，符合动态image confidence仅在刷新后生效。预测体积2390.0 voxels、空预测12、precision0.3593、recall0.3533。checkpoint High/Low nonfinite12/0，optimizer steps42/100，AMP跳步8，scale512（相比@50的11次非finite/19步/6跳步有所趋稳）。


M5 iteration150 mean Dice=0.301495、mIoU=0.230206，新 best，较 Zero-shot +0.018028；mean predicted voxels=2513.2、空预测11、precision=0.3739、recall=0.3662。梯度 checkpoint High/Low nonfinite13/0，optimizer steps66/150，9次累积批次跳过，scale256；仍有早期数值警告但训练继续。


M5 iteration200 mean Dice=0.307931、mIoU=0.234298，新 best；mean predicted voxels=2587.4、空预测11，precision=0.3742、recall=0.3740。checkpoint High/Low 非有限梯度13/0，optimizer steps89/200、跳过11，scale256；自 iteration150 后非有限梯度计数未增加，AMP状态稳定。此时尚未触发刷新。


M5 iteration250 mean Dice=0.312894、mIoU=0.237475，新 best；较Zero-shot +0.029427。预测体积均值2669.9 voxels、空预测9例，mean precision=0.3697、recall=0.3804；HD95有限样本均值 56.634 mm（35/44），ASD有限样本均值 34.071 mm（35/44）。optimizer steps=114/250，AMP累积批次跳过11，High/Low非有限梯度13/0，scale256。与M4同轮Dice0.312851、体积2671.8、空预测9例基本一致；刷新尚未发生，M5动态image confidence的作用需在iteration374刷新后评估。


M5 iteration300 mean Dice=0.315802、mIoU=0.239178，新 best；较Zero-shot +0.032335。预测体积均值2763.0 voxels、空预测8例。与M4 iteration300 Dice0.315807基本完全一致（差-0.000005），动态image confidence尚未刷新生效。checkpoint optimizer steps=139/300、AMP跳过11批、High/Low非有限梯度13/0。继续至iteration350/374刷新。


M5 iteration350 mean Dice=0.321703、mIoU=0.242815，新 best；较Zero-shot +0.038236。预测体积均值2974.0 voxels、空预测6例。与M4刷新前 iteration350 Dice0.321696几乎完全相同（+0.000007），image confidence刷新尚未生效。optimizer steps=164/350、AMP跳过11批、High/Low非有限梯度13/0。下一步越过iteration374触发同步refresh。


M5 iteration374 完成101/101病例覆盖，触发 version-1 Student snapshot refresh（源 checkpoint 为best@350 Dice0.321703）。按 M5 统一 Gaussian benchmark 推理协议刷新 consensus、PE、case confidence及空预测状态；刷新阶段没有训练参数更新。缓存完成后单独审计其离线质量与置信度变化。


M5 version-1 refresh progress: 10/101 cases written; training remains held at the iteration374 refresh boundary pending complete cache construction.


M5 version-1 refresh progress: 20/101 cases written; process remains active and training waits at the refresh boundary.


M5 version-1 refresh progress: 30/101 cases written; GPU process and cache construction remain active.


M5 version-1 refresh progress: 43/101 cases written; full cache remains pending.


M5 version-1 refresh progress: 51/101 cases written; cache generation continues normally.


M5 version-1 refresh progress: 64/101 cases written; the training checkpoint remains held at iteration374.


M5 version-1 refresh progress: 79/101 cases written; final cache validation and offline refresh audit are pending.


M5 version-1 cache完成101/101且metadata核验通过（benchmark_gaussian、tile step0.5、student_snapshot iteration374）。GT-only离线审计 `M5_refresh_audit/refresh_summary.json`：mean case confidence 0.53381→0.62722（+0.09341）；GT mean Dice 0.33816→0.34432（+0.00616）；空预测20→2，18例由空恢复、无新增空预测；confidence change与GT Dice change Pearson r=0.51475（p=3.65e-8）、Spearman rho=0.52272（p=2.06e-8）。较M4 refresh相关性更强，但尚不能推断学生训练稳定性；M5已恢复训练，继续观察post-refresh验证。GT只用于离线审计。


M5 iteration400（version-1动态image confidence + 动态PE生效）mean Dice=0.319538、mIoU=0.242536；best仍为iteration350 Dice0.321703（差-0.002165）。平均预测体积3261.2 voxels、空预测4例、precision=0.3585、recall=0.4220；HD95有限均值67.313mm（40/44），ASD有限均值43.508mm（40/44）。相较M4同轮Dice0.319150、体积3263.2、空预测4例几乎相同；M5尚无明显学生层面收益。optimizer steps=188/400，累积AMP跳过12次；继续观察iteration450–600。


M5 iteration450 mean Dice=0.313997、mIoU=0.238483；best仍@350=0.321703（差-0.007706）。预测体积均值3868.4 voxels，较@400增长18.6%，空预测3例，precision=0.3422、recall=0.4267。与M4同轮Dice0.312216、体积3817.7接近；未见动态image confidence消除刷新后的体积增长，继续观察至iteration600以评估是否进入过分割坍塌。


M5 iteration500 mean Dice=0.307936、mIoU=0.233382；best仍@350=0.321703，低0.013767。预测体积均值5470.9 voxels、空预测0例、precision=0.3187、recall=0.4357。M4同轮Dice0.307948几乎完全相同，体积5203 voxels；M5体积更大约5.1%，无明显稳定性收益。optimizer steps=238/500，AMP跳过12批，High/Low非有限梯度13/0。继续至iteration600确认趋势。


M5 iteration550 mean Dice=0.314558、mIoU=0.238941；best仍@350=0.321703。预测体积均值9509.0 voxels、空预测0、precision=0.2939、recall=0.5268；体积继续增长且precision下降。相较M4@550 Dice0.319126、体积8334.9，M5此时Dice更低、体积更大。optimizer steps=263/550、AMP跳过12批、High/Low非有限梯度13/0。继续至iteration600检查坍塌。


M5 iteration600 mean Dice=0.279047、mIoU=0.208430，较best@350 Dice0.321703下降0.042656，并低于Zero-shot 0.283467。平均预测体积19195.0 voxels、空预测0、precision=0.2398、recall=0.6224。相比M4@600 Dice0.296832、体积15703.8，M5更差且过分割更重。iteration600 checkpoint已保存（optimizer steps=288/600，AMP跳过12，High/Low非有限梯度13/0），随后停止参数更新。M5动态image confidence刷新显著提升了离线置信度变化与GT Dice变化相关性（r=.515），并改善缓存共识Dice，但没有带来Student层面的稳定性/性能收益。M6的先决条件“前序实验显示有效改善”不成立，因此按1.md条件不启动，不将CSS叠加到已失败机制。

## Final cross-stage review

Formal M5 stopped after the saved iteration600 checkpoint because validation Dice fell below zero-shot while predicted foreground volume expanded to 19,195 voxels. Best/latest checkpoints and per-case validation outputs are retained. No effective stable improvement across M1–M5 supports the conditional M6 gate, so CSS was not layered onto the failed dynamic mechanisms.

| Experiment | Best validation Dice (iteration) | Latest/final validated Dice | Latest mean predicted voxels | Latest empty cases | Outcome |
|---|---:|---:|---:|---:|---|
| Zero-shot | 0.283467 (0) | 0.283467 | 2,226.7 | 13 | Shared source reference |
| M0 aligned Dynamic-hard | 0.340715 (1650) | 0.331105 (2000) | 8,479.8 | 0 | Only full 2000-iteration aligned run |
| Historical E joint | 0.338528 (1800) | 0.338528 | 5,395.5 | 3 | Historical protocol reference |
| Historical CSS | 0.311889 (2000) | 0.311889 | 12,292.6 | 0 | Historical protocol reference |
| M1 | 0.310338 (250) | 0.093026 (450) | 229,456.7 | 0 | Post-refresh severe oversegmentation |
| M2 | 0.309401 (250) | 0.179595 (400) | 78,664.3 | 0 | Post-refresh severe oversegmentation |
| M3-control | 0.321396 (550) | 0.167391 (650) | 66,465.4 | 0 | Post-refresh collapse |
| M3-discount | 0.323672 (550) | 0.235956 (650) | 28,040.9 | 0 | Discount delayed, did not prevent collapse |
| M4 | 0.321696 (350) | 0.208088 (650) | 36,885.0 | 0 | Dynamic PE did not prevent collapse |
| M5 | 0.321703 (350) | 0.279047 (600) | 19,195.0 | 0 | Dynamic image confidence did not prevent collapse |

Best scores for the exploratory mechanisms are all below both M0's aligned best (0.340715) and the historical E result (0.338528). M5's best exceeds M4 by only 0.000007 and occurs before the image-confidence refresh. At matched post-refresh checkpoints, M5 does not show stable gains: it is +0.000388 over M4 at 400, effectively tied at 500, then -0.004568 at 550 and -0.017785 at 600. M3 case-discount reduced per-case gradient contribution (sample-weighted mean discount 0.537; gradient norm after/before ratio 0.560) and partially delayed the control collapse, but none of the explored pixel/image uncertainty additions stabilized the student.

The M5 refresh audit is a positive cache-level finding: the refreshed confidence change aligns with offline GT Dice change (Pearson r=0.515, Spearman rho=0.523), and mean consensus Dice improves by 0.00616 while empty refresh predictions decline 20→2. This did not translate into improved student training: after refresh, its predicted volume grew from 3,261 at iteration400 to 19,195 at iteration600 while Dice fell from 0.319538 to 0.279047. Thus image/prompt information appears useful for offline cache quality ranking, but the current gradient-discount and refresh implementation does not make that signal sufficient to prevent oversegmentation.

No experiment achieved the stated mean Dice target above 0.8; the strongest aligned result is M0 best=0.340715. The 2000-iteration fixed-budget result is available only for M0 aligned (Dice=0.331105, 1,994 optimizer steps); the other mechanisms were stopped early according to the predeclared collapse review and therefore have no fixed-budget result. Best checkpoints are reported alongside latest validated checkpoints so selection does not hide post-refresh collapse.

The final machine-readable outputs are `validation_summary.csv` and `validation_per_case.csv`. They include Dice, mIoU, Precision, Recall, HD95, ASD, foreground volume, zero-shot Dice strata, per-case deltas, new empty predictions, and counts improved/decreased by more than 0.02. Refresh audits under `M*_refresh_audit/` use GT strictly offline; GT was not used for training, sampling, confidence calculation, or refresh construction. M6 remains intentionally unrun because its prerequisite—effective improvement in earlier stages—was not met.

GPU scheduling note: M5 was the only eligible dependent experiment after the M4 review; M6 was gated on an effective earlier improvement and therefore was not eligible to run in parallel. M5 used physical GPU0 for training/refresh. GPU1 was not filled with dummy allocations or a duplicate training run, consistent with preserving the experiment protocol; it remained idle when no independent valid experiment was available.
