# MSTT_RUL v0.3.0：HUST 冻结校准与一次性外部确认

这个代码包用于论文的下一轮实质性补强：在研究内尚未查看容量轨迹和模型结果的 HUST 数据上，先做结构门禁，再冻结评估器，只打开 20 个校准电芯，最后一次性打开 57 个确认电芯。

它不是“多跑一个数据集”的普通脚本。程序会强制执行证据顺序，保留所有排除、失败、右删失和不利结果；如果确认队列不足 40 个可评估电芯，H1/H2 自动标记为不可估计，不会事后改 landmark。

## 1. 这个包已经包含什么

- 原 `v0.2.0-SOH` 的 MSTT full K5 与 single-scale K5 源码；
- 完整训练协议、三种子（42、2024、3407）和参数量门禁；
- XJTU Batch 1+3 的冻结开发输入；
- HUST 结构清点、20/57 SHA-256 固定分组、解析和容量积分；
- 动态 SOH landmark、局部线性强基线、RMSE/timing/删失评价；
- 80%/90% simultaneous conformal 校准；
- 电芯级 Wilcoxon、配对 bootstrap 和固定顺序 H1/H2；
- 评估器冻结、校准冻结、外部登记回执、一次性运行状态和最终结果打包；
- 不依赖真实 HUST 结果的合成测试。

不包含 HUST 原始数据，也不包含预先跑好的 HUST 权重或结果。

`src/mstt_soh/pipeline.py` 是为了保持 `v0.2.0-SOH` 模型与训练实现逐字可追溯而保留的原文件，因此其中仍可见旧的 HNEI/BIT 兼容命令和旧协议标签。本包的 HUST 正式流程只允许通过 `manager.sh` 调用其中的 XJTU `prepare-development`、`train` 和冻结权重加载函数；不要运行该旧文件里的 HNEI/BIT 命令，也不要据此把 HNEI 重新包装为新确认集。

## 2. 只下载这一个外部数据包

从 HUST 的官方 Mendeley Data 版本 2 下载原始归档：

- 数据页：https://data.mendeley.com/datasets/nsc7hnsg4s/2
- DOI：`10.17632/nsc7hnsg4s.2`
- 文件名必须为：`hust_data.zip`

不要先解压，不要打开其中的 pickle，不要画容量曲线，也不要用 BatteryLife 预处理。默认将原包放在：

```text
/root/shared-storage/hust_data.zip
```

如果放在别处，运行前设置：

```bash
export HUST_ARCHIVE=/你的绝对路径/hust_data.zip
```

警告：pickle 可能执行代码。程序只在原始 ZIP 已通过结构清点并绑定 SHA-256 后允许解析，而且仍要求显式的 `--trust-official-pickle` 确认。只使用上述官方来源下载的原包。

## 3. 环境

建议 Python 3.10–3.12；正式训练和确认建议使用 CUDA。CPU 也能运行，但耗时取决于硬件。

```bash
unzip MSTT_RUL_v0.3.0_HUST_confirmatory_code_20260801.zip
cd MSTT_RUL_v0.3.0_HUST_confirmatory_20260801
chmod +x manager.sh
python scripts/verify_code_package.py
./manager.sh install
./manager.sh check
```

如已有满足 `requirements.txt` 的环境，可以跳过 `install`。默认所有运行文件写到代码包内的 `run_v030/`；也可在第一次运行前设置新的绝对路径：

```bash
export HUST_RUN_ROOT=/你的绝对路径/MSTT_RUL_HUST_v030_run
export HUST_DEVICE=cuda
```

同一次正式研究必须始终使用同一个 `HUST_RUN_ROOT`，不能在看到结果后换目录重新开始。

## 4. 严格运行顺序

### 阶段 A：不读取 HUST 容量值

```bash
./manager.sh inventory
./manager.sh prepare_development
./manager.sh smoke
```

`inventory` 只检查 ZIP 结构、CRC、77 个 pickle 文件名和元数据，并按固定盐值对电芯 ID 做 SHA-256 排序：前 20 个为校准集，后 57 个为确认集。此时不 unpickle，不计算容量、SOH、EOL或模型输出。

`smoke` 只做 2 epoch、1模型、1种子的程序连通测试，输出目录明确标为 `smoke_only_not_formal`，绝不能作为论文权重。

### 阶段 B：正式训练并冻结评估器

前台训练：

```bash
./manager.sh train
```

或后台训练：

```bash
./manager.sh train_bg
./manager.sh status
```

正式训练固定为 2 个神经模型 × 3 个种子。程序可按已完成且哈希一致的 job 安全续跑；不得修改配置或复用 smoke 权重。

训练完成后：

```bash
./manager.sh freeze_evaluator
```

该命令输出：

- `MSTT_RUL_v0.3.0_HUST_evaluator_pre_calibration_freeze.zip`
- 对应 `.sha256`
- 对应 `.manifest.json`

把确切代码提交到你的论文代码仓库，并把冻结 ZIP、manifest 和 SHA-256 发布到不可变版本归档（推荐 Zenodo specific-version DOI；也可使用能明确绑定该 SHA-256 的固定 release）。然后登记：

```bash
./manager.sh register_evaluator 40位Git提交哈希 不可变URL或DOI
```

示例格式仅说明参数形状，不是可用回执：

```bash
./manager.sh register_evaluator 0123456789abcdef0123456789abcdef01234567 https://doi.org/10.xxxx/example
```

没有真实 commit 和真实不可变定位符，不要继续。

### 阶段 C：只打开 20 个校准电芯

```bash
./manager.sh open_calibration
./manager.sh calibrate
./manager.sh freeze_calibration
```

校准阶段只允许：

- 验证固定解析器；
- 计算 full K5 的电芯级最大绝对 SOH 误差；
- 冻结 80% 和 90% simultaneous band 宽度。

禁止根据校准结果更换模型、权重、特征、landmark、排除规则、对手或主指标。如果合格校准电芯不足 15 个，不确定性标为不可估计，但仍按冻结方案继续点预测确认。

发布校准冻结包并登记第二份回执：

```bash
./manager.sh register_calibration 40位Git提交哈希 不可变URL或DOI
```

注意：第二个 commit 应真实包含校准冻结的 manifest/哈希或能唯一追溯它；不可伪造哈希或使用示例值。

### 阶段 D：一次性打开 57 个确认电芯

前台运行：

```bash
./manager.sh confirm
```

或后台运行：

```bash
./manager.sh confirm_bg
```

确认命令会在读取任何确认 pickle 之前先写 `confirmation/one_shot_state.json`。一旦状态文件出现，这个确认集就已经视为揭盲：

- 成功后禁止重跑；
- 若因断电或程序中断失败，只能使用完全相同的数据、代码、权重和冻结包执行：

```bash
./manager.sh confirm_resume
```

不得删除状态文件、换目录或修改代码后声称仍是同一次确认。

### 阶段 E：汇总和打包

```bash
./manager.sh aggregate
./manager.sh pack
```

最终 ZIP 位于：

```text
run_v030/MSTT_RUL_v0.3.0_HUST_confirmatory_one_shot_results.zip
```

同时生成 SHA-256 sidecar。把这个 ZIP、sidecar、两个冻结包及其外部回执原样发回，不要只发截图或手工抄写均值。

## 5. 冻结的科学定义

### SOH 与 EOL

对每个物理电芯：

```text
SOH(t) = C(t) / median(first 5 valid capacity observations)
EOL = first genuinely observed raw SOH <= 0.80
```

模型目标是物理周期网格上因果 LOCF 后的 7 周期单侧移动平均 SOH；EOL 和 RMSE 真值仍使用真实观测的 raw SOH，不使用平滑值伪造事件。

### 动态 landmark

选择第一个同时满足以下条件的真实观测周期：

1. `0.80 < raw SOH <= 0.90`；
2. 此前没有真实观测达到 `raw SOH <= 0.80`；
3. 截点前已有至少 20 个真实观测容量点；
4. 截点后在 EOL/右删失边界内至少有 5 个真实观测点用于 RMSE。

这个 landmark 只用当前与过去数据，且在查看 HUST 容量前写死。没有合格 landmark 的电芯保留在 flow 和 exclusion 表中，不能另换 cycle。

### 主假设

- H1：`mstt_full_K5` 对比 `local_linear_trend`；
- H2：`mstt_full_K5` 对比 `mstt_single_scale_K5`，只有 H1 的双侧 p < 0.05 才打开确认性 p 值门；
- 主指标：每个物理电芯一个未来 raw-SOH RMSE；
- 种子先在轨迹层逐点平均，种子不计作独立样本；
- 至少 40 个完整配对确认电芯才进行 H1/H2 推断；
- 实质成功还要求 H1 配对 bootstrap 95% CI 下界 > 0，且平均改善至少 0.005 SOH。

## 6. 关键输出怎么看

最终结果包最重要的是：

- `aggregate/hust_confirmation_flow.json`：57 个确认电芯的纳入、排除和可估计性；
- `aggregate/hust_fixed_sequence_inference.json`：H1/H2 结果和门控状态；
- `aggregate/hust_confirmatory_model_summary.csv`：三个模型的描述性汇总；
- `aggregate/hust_fixed_sequence_effects.csv`：配对效应、bootstrap 区间和 p 值；
- `aggregate/hust_calibration_coverage_summary.csv`：80%/90% 区间表现；
- `confirmation/prepared/confirmation_exclusions.csv`：所有排除原因；
- `confirmation/ensemble_cell_records.csv`：逐电芯、逐模型核心结果；
- `RESULT_MANIFEST.json` 和 ZIP sidecar：完整性核验。

结果出现以下任何一种情况都必须如实保留：

- 少于 40 个可评估确认电芯：确认假设不可估计；
- H1 不通过：H2 不给确认性 p 值；
- full K5 输给局部线性：不能调整 landmark 或排除电芯再跑；
- 区间覆盖率不足：不能把校准写成部署优势；
- 右删失或未穿越：使用删失感知结果，不删除这些电芯。

## 7. 数据解析依据和表述边界

HUST 解析遵循 Microsoft BatteryML 的公开 HUST 预处理思路：在每个周期内对负电流按时间积分得到放电容量，并对 `7-5` 电芯跳过最初两个异常周期。参考实现：

https://github.com/microsoft/BatteryML/blob/main/batteryml/preprocess/preprocess_HUST.py

论文中允许写：

> The HUST evaluation was prospectively specified and frozen before study-specific access to HUST capacity trajectories and model outputs.

不允许写：

> HUST was a prospectively collected cohort.

HUST 是公开历史数据，前瞻性只指本研究内的分析方案和一次性揭盲顺序。

## 8. 故障处理

- `Expected 77 ... found ...`：文件不是注册的 HUST v2 原包，停止并重新从官方记录下载。
- `pickle location` 或列名错误：先保留完整错误和哈希，不要自行改解析器后继续确认；在校准阶段可作为新版本修订，但必须重新冻结并重新登记，且确认集仍未打开。
- `formal model manifest gate failed`：误用了 smoke 权重、训练未完成或文件被修改。
- `receipt gate failed`：冻结 ZIP、manifest、模型或回执不一致。
- `confirmation already completed`：这是正常保护，不能重跑。
- `FAILED_EXACT_RESUME_ONLY`：保留所有文件，用 `confirm_resume`；如果修改任何输入，就只能作为新的探索性版本，不能替代 v0.3.0。

运行路径可随时查看：

```bash
./manager.sh paths
```
