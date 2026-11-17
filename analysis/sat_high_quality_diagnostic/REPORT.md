# C_joint High 组 SAT 互补性离线诊断

## Executive summary

- 固定使用 `outputs/response_routing/C_joint/routing.csv` 的 59 个 High；核对 split 后 59/59 均属于 101-case train split。分组为 Empty High 10、Non-empty Hard High 11、Good High 38。GT Dice 只用于分组后的评估，没有参与 routing。
- SAT 使用官方 general SAT-Nano `UNET` checkpoint 和配套 text encoder，prompt 为 `liver cancer`。复用旧 `sat_no_response` 的 10 个 Empty High 预测，只对其余 49 个病例做了 SAT 离线推理。无 VoxTell 适配或 SFDA 训练。
- 按已存冻结 Dice 比较，SAT 在 All High 的 mean Dice 为 0.196，对比 frozen VoxTell 的 0.524；SAT 高于 VoxTell 的病例 14/59，低于 VoxTell 的病例 38/59。SAT 总体不优于 VoxTell。
- SAT 的主要可见价值在 VoxTell-empty 病例：10/10 SAT 非空，mean Dice 0.109、mean precision 0.311、mean recall 0.145；但中位 Dice 仅 0.003，只有 3/10 Dice >0.1、3/10 >0.3、0/10 >0.5，说明多数新增前景较差。
- 对 11 个 Non-empty Hard High，SAT Dice 比 VoxTell 高的有 6 例、低的 3 例、相同的 2 例；`ΔDice > .02/.05` 分别 5/4 例，`ΔDice < -.02/-.05` 均为 3 例。SAT full-mask mean precision 只有 0.110。由于这 11 例没有现成 frozen VoxTell mask，不能测 SAT-only 区域的精确度，也不能断言 SAT 新增区域恢复了多少 FN。
- 在 38 个 Good High 中，SAT Dice 低于 VoxTell 的有 35 例；34 例下降超过 0.05。不能用 SAT 替换 Good High 的 VoxTell prediction。
- Case-level best-of-two oracle 的 All High mean uplift 为 **+0.0342**、median uplift 为 **0**，14/59 例有提升。oracle 只说明病例级选择的上限，不是可执行的 pseudo-label 方法。
- 目前不能判断两模型交集“显著更可靠”：完整 VoxTell masks 只在 10 个 Empty High 可用，这些病例交集为空，precision=0；其余 49 例缺少 frozen VoxTell mask，intersection/union 与 SAT-only 分析无法完成。当前证据更支持把 SAT 看作**候选 proposal 来源**，不支持直接作为 pseudo-label teacher，也不足以评估 agreement verifier。

## 固定数据和分组

`routing.csv` 有 101 个 train IDs；其中 High 共 59 个，所有 ID 都与 `split.json` 的 train split 一致。冻结 VoxTell 的逐例 `baseline_dice` 来自 `VoxTell_from_disk/diagnostic/dwi_uncertainty_train/per_case_uncertainty.csv`，prompt 是标准 `liver cancer`，在适配前生成。routing 不使用 GT Dice。

Empty High 由固定 C_joint routing 的 `response_state=No-response` 标识，并与已有的 frozen VoxTell 概率图逐例核验：这 10 个 raw mask 在阈值 0.5 下均为空。其余 49 例是 response-present，因此 frozen mask 非空；其中 frozen Dice <0.5 的 11 例为 Non-empty Hard High，Dice ≥0.5 的 38 例为 Good High。Good/Hard 切分是离线分析标签，不回写 routing。

## SAT 推理与预测复用

- 使用 `SAT/weights/Nano/nano.pth`、`SAT/weights/Nano/nano_text_encoder.pth`，本地兼容版入口 `SAT/runtime_compat/inference.py`，`vision_backbone=UNET`、`modality=mri`、`orientation_code=RAS`、batch size 1、prompt `liver cancer`。
- 已有 SAT 预测复用 10 例；新推理只覆盖其余 49 例，结果保存在本目录的 `sat_predictions/`，未写入或覆盖 `outputs/`。
- 全部 59 例 SAT mask 均非空。

## 分组指标

VoxTell 的 mean/median Dice 可从冻结逐例 CSV 覆盖全部 59 例。冻结 VoxTell precision、recall、前景体素数以及跨模型逐体素比较只有 10 个已保存 frozen masks，且这 10 例都为空。其余位置留空，不能把缺失误读成 0。SAT/GT 指标在 SAT 推理的 RAS、1×1×3、foreground-cropped 网格计算。

| Group | N | SAT non-empty | VoxTell Dice mean / median | SAT Dice mean / median | SAT P / R mean | ΔDice mean / median | SAT better / worse | Δ>.02 / >.05 | Δ<-.02 / <-.05 | best-of-two mean uplift / median | SAT-only precision | intersection P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Empty High | 10 | 10/10 | .000 / .000 | .109 / .003 | .311 / .145 | +.109 / +.003 | 5 / 0 | 4 / 3 | 0 / 0 | +.109 / +.003 | .311 | 0.000* |
| Non-empty Hard High | 11 | 11/11 | .148 / .0003 | .129 / .044 | .110 / .349 | −.018 / +.002 | 6 / 3 | 5 / 4 | 3 / 3 | +.046 / +.002 | unavailable | unavailable |
| Good High | 38 | 38/38 | .771 / .776 | .239 / .092 | .397 / .222 | −.532 / −.620 | 3 / 35 | 2 / 2 | 34 / 34 | +.011 / .000 | unavailable | unavailable |
| All High | 59 | 59/59 | .524 / .700 | .196 / .055 | .329 / .233 | −.327 / −.338 | 14 / 38 | 11 / 9 | 37 / 37 | +.034 / .000 | .311† | 0.000† |

`SAT better/worse` counts use strict ΔDice >0 / <0; ties are 7/59 overall. `Δ` thresholds are strict. `*` Empty High intersection is empty for all 10 cases, so its precision is zero by convention and does not support an agreement-quality claim. `†` All High values are calculated only over the 10 complete mask pairs, all of which are Empty High; they are not All High estimates. For Hard and Good, SAT-only/intersection values are unavailable because frozen VoxTell masks were not saved.

On the only complete-pair subset (n=10 Empty High), aggregate intersection Dice/Precision/Recall are **0/0/0**; aggregate union Dice/Precision/Recall are **0.1089/0.3115/0.1447**. Since VoxTell is empty in every one of these cases, the union is exactly SAT. These are empty-mask overlap facts, not evidence that agreement improves reliability for non-empty predictions.

## Empty High 逐例 SAT Dice

| Case | SAT Dice | SAT Precision | SAT Recall | SAT foreground voxels |
|---|---:|---:|---:|---:|
| 0001679853.nii.gz | .0000 | .0000 | .0000 | 58 |
| 0017354579.nii.gz | .3256 | .8951 | .1990 | 15,256 |
| 0017452826.nii.gz | .0350 | 1.0000 | .0178 | 1,083 |
| 0018806748.nii.gz | .3001 | .9492 | .1783 | 12,432 |
| 0019873347.nii.gz | .4215 | .2672 | .9984 | 4,675 |
| 0032182222.nii.gz | .0000 | .0000 | .0000 | 2,964 |
| 0032229283.nii.gz | .0063 | .0033 | .0538 | 3,003 |
| 0033235247.nii.gz | .0000 | .0000 | .0000 | 613 |
| 0033498429.nii.gz | .0000 | .0000 | .0000 | 524 |
| 0033923781.nii.gz | .0000 | .0000 | .0000 | 891 |

SAT-only equals SAT in Empty High because VoxTell is empty. Its mean precision of 0.311 and mean recall of 0.145 show both some recovered GT foreground and substantial false positive volume. This is a proposal signal, not evidence that the entire SAT mask should be accepted.

## 诊断问题结论

1. **SAT 是否总体优于 VoxTell？** 否。All High mean Dice 0.196 vs 0.524；14 例 SAT 较好、38 例较差、7 例相同。跨模型 Dice 使用现有冻结 Dice 与 SAT 重采样网格指标，受下方 preprocessing caveat 限制。
2. **价值在 Empty/Hard 还是 Good？** Empty High 的 SAT 可提供非空前景，且 5/10 Dice 高于空的 VoxTell；Hard 有一定病例级互补（6/11 SAT Dice 较高），但整体 mean 仍低 0.018，full-mask precision 低；Good High 中 35/38 被 SAT 明显拉低。因此潜在增量集中在 Empty/Hard，而不是 Good。
3. **SAT-only 是否补 VoxTell FN？** Empty High 可直接确认：SAT-only 就是 SAT mask，平均 recall 0.145，说明确有一部分 GT 前景被找回，同时 precision 0.311。Hard/Good 因 frozen masks 缺失，不能直接统计 SAT-only TP/FP，不能进一步量化 FN 修复。
4. **Intersection 是否更可靠？** 不能据此判断。唯一完整配对的 10 例都是 VoxTell 空预测，交集为空；49 例有 VoxTell prediction 但文件缺失。没有可用的非空交集样本来评估一致区域 precision。
5. **best-of-two oracle 有多大？** All High mean +0.0342 Dice、median +0、14/59 有提升；Hard +0.0463、Empty +0.1089、Good +0.0111。它逐病例预知哪一个模型更好，属于 oracle 上限。
6. **SAT 更适合什么角色？** 目前只支持把它作为待进一步筛选的 candidate proposal model。直接 teacher 风险高，Good High replacement 明显有害；agreement verifier 因缺失非空 VoxTell masks 尚无证据。

## 数据与 protocol 限制

- **Train vs test：** 本分析严格限定 C_joint 固定训练 High 59 例，没有混入 44 个 held-out/test cases。
- **Frozen vs adapted：** 所有 VoxTell 数值来自未适配 VoxTell；没有读入 adapted checkpoint prediction 定义分组。
- **分组 vs GT：** High/Empty/Hard/Good 标签来自既有 routing、frozen Dice 和 response 状态。GT 只用于之后的分割统计。
- **VoxTell mask 缺失：** 仓库可复用的 frozen VoxTell probability/mask 仅有 Empty High 的 10 例。其余 49 例没有 raw frozen mask；没有补跑 VoxTell。故这些病例的 VoxTell precision/recall/volume、union/intersection、SAT-only/VoxTell-only precision 均缺失。
- **SAT grid：** SAT 按官方 preprocessing 输出 RAS、1×1×3 spacing、foreground crop 的网格。新推理 mask 与当前 MONAI 转换的 GT 网格逐例匹配。旧 `sat_no_response` 的 10 个 SAT mask 是在旧环境生成的；在当前 MONAI 1.6.1 的 GT transform 下有个别一体素 shape 差异。为了不改旧结果，这 10 例直接复用了已保存的 post-hoc SAT Dice/P/R/voxel 数；由于对应 VoxTell mask 全空，可准确确定交集为空、并集和 SAT-only 等于 SAT。SAT 与冻结 VoxTell mean Dice 的横向比较因此有预处理网格差异 caveat。
- **Protocol：** SAT 是外部 general model，不作真值或默认更可信；本次没有训练、SFDA adaptation、Stage 2、loss/routing 更新或新方法设计。
- **显著性：** 报告为描述性统计，没有做显著性检验。

## SAT 目录整理记录

以下目录从 `/home/ubuntu/zy/pythoncode/sfda/` 归入明确的 SAT 根目录，未删除数据或已有实验结果：

- `SAT_repo/` → `SAT/runtime_compat/`：本地兼容版 checkout，保留了移动前已有的 5 个未提交源码修改与 `.git` 元数据。
- `SAT_weights/` → `SAT/weights/`：Nano 权重及原有 Hugging Face cache 一并保留。
- 原 `SAT/` 保持为干净 upstream checkout。

路径引用检查未发现 CUE-SFDA 代码中的旧绝对路径或 `SAT_repo`/`SAT_weights` 运行时依赖。更新了两处几何说明注释：`scripts/sat_no_response_diagnostic.py` 和 `scripts/analyze_sat_voxtell_response.py` 现在指向 `../../SAT/runtime_compat/data/inference_dataset.py`。新增 `SAT/LOCAL_LAYOUT.md` 记录目录布局。既有 `outputs/` 没有移动或覆盖。

## 输出文件

- `per_case.csv`：59 个 High 的冻结 Dice、SAT Dice/P/R/volume、GT volume、可用的 intersection/union/SAT-only/VoxTell-only 指标及 oracle ΔDice。不可计算项为空值。
- `subgroup_summary.csv`：Empty High、Non-empty Hard High、Good High、All High 汇总；所有可用样本数单独列出。
- `sat_high_missing.jsonl`：本轮 SAT 实际推理的 49 个病例。
- `sat_predictions/`：新 SAT 输出；旧复用结果仍保留在原 `outputs/response_routing/sat_no_response/`。
- `input_manifest.json`：固定分区与复用/推理病例数记录。
- `analyze_sat_high.py`：离线聚合脚本；复用现有 SAT No-response GT transform 与 metric 实现。
