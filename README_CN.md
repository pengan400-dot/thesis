# MSTT-RUL v0.2.0：SOH 开发、HNEI 探索与 BIT 评估前冻结

本包执行以下顺序：

1. 仅用 XJTU Batch-1 与 Batch-3 开发 SOH 模型；
2. 冻结并归档 `v0.2.0-SOH-development`；
3. 用 HNEI 做探索性跨机构评估；
4. 明确记录 HNEI 是否导致任何模型修改；
5. 仅对 BIT 做结构预检并冻结 `v0.2.0-BIT-SOH-freeze`。

本包故意不含 BIT 模型评估命令。BIT 的物理周期字段和容量字段必须在
真实结构报告出来后解析并冻结，不能提前猜。

## 已冻结的关键规则

\[
SOH_{i,t}=
\frac{C_{i,t}}
{\operatorname{median}(C_{i,\text{first 5 valid}})}.
\]

- 分母只取按时间排序的前 5 个有效 Ah 容量观测。
- 作为 BOL 分母候选的单个有效值必须有限且处于标称容量的
  0.5–1.5 倍；真实晚期容量低于 0.5 倍时仍保留在轨迹中。
- 前五个有效值的中位数还必须处于标称容量的 0.8–1.2 倍；这会拒绝
  被误标为 Ah 的既有归一化值。
- 不因查看完整寿命后认为早期值“异常”而改取后面的周期。
- 少于 5 个有效值直接触发数据质量门禁。
- EOL 为 `SOH <= 0.8`。

绝对 Ah 超参数按预先指定的 XJTU 2.0 Ah 基准做量纲换算，不根据 HNEI
或 BIT 成绩选择：

| 项目 | 冻结 SOH 值 |
|---|---:|
| 残差边界 | 0.04 |
| 物理裁剪 | [0.2, 1.2] |
| Huber delta | 0.01 |
| EOL 权重带宽 | 0.04 |
| 向上跳变容差 | 0.001 |

七个输入通道统一使用 XJTU-only StandardScaler。完整模型参数量为
74,405；single-scale K5 为 69,789。旧绝对 Ah 权重不加载。

模型输入窗口为 16 步，滚动特征还需要 4 个物理周期预热，因此 cutoff
前最少历史固定为 20 个 physical cycles。主要 RMSE 至少需要 cutoff 后
5 个真实 SOH 观测点。

Conformal 使用 XJTU B1+B3 的四折、批次分层、物理电芯级 outer
cross-fitting。每个电芯和 cutoff 的 nonconformity score 是种子轨迹先
逐点平均后得到的最大绝对 SOH 误差，因此输出的是 simultaneous
trajectory band，而不是把训练残差当校准集。

## 运行前准备

### 1. 机器

- Linux 服务器；
- Python 3.10–3.12；
- 推荐 NVIDIA GPU，显存至少 16 GB；CPU 可跑冒烟测试，但正式 18 个
  训练作业会明显更慢；
- 推荐至少 32 GB 内存和 50 GB 可用磁盘。

正式开发包括 2 个神经模型 × 3 个种子，以及 conformal 的 4 折 ×
3 个种子，共 18 个训练作业。程序会保存 epoch 选择、最终训练历史、
权重、scaler、环境版本和 SHA-256；一致的已完成作业可安全续跑。

### 2. XJTU B1+B3

准备 16 个逐电芯 CSV：B1 八个、B3 八个，不需要 B2。

推荐直接使用论文现有的 `development_prepared` 目录。每个文件必须
包含：

```text
cycle, raw_capacity, capacity, battery_id, batch
```

其中 `raw_capacity` 必须仍是绝对 Ah，`batch` 只能是 1 或 3。已有
`capacity` 列会被审计记录但不参与新开发；程序统一从 `raw_capacity`
重新计算冻结的 7 周期单侧滚动均值，以消除既有平滑规则差异。

也支持简单格式：

```text
cycle, capacity_Ah
```

简单格式的文件名必须明确含 `_B1_` 或 `_B3_`。此时程序自行做 7 周期
单侧滚动平滑。

所有 cycle 必须是从 1 开始的连续 physical aging-cycle number。不要先
把容量除以初始值；代码会执行一次且仅一次 SOH 转换。
16 个电芯 ID、批次和周期数还必须与包内
`manifests/xjtu_expected_reference.csv` 一致；输入文件和归一化曲线的
实际 SHA-256 会写入冻结清单。

默认目录：

```text
/root/workspace/MSTT_RUL_Q2_Batch456_confirmatory_20260725/development_prepared
```

若位置不同：

```bash
export XJTU_PREPARED_DIR=/实际路径/development_prepared
```

### 3. HNEI

准备原始文件：

```text
/root/shared-storage/BatteryLife.zip
```

冻结的预期 SHA-256：

```text
a6c33eaee5edbf6288dea4bf8ce7513b0fb1c27579c8ae78882dd1279f54a901
```

若位置不同：

```bash
export HNEI_ARCHIVE=/实际路径/BatteryLife.zip
```

### 4. BIT Version 3

从官方页面下载 Version 3、DOI `10.17632/kw34hhw7xg.3` 的完整原始归档：

```text
https://data.mendeley.com/datasets/kw34hhw7xg/3
```

先不要解读容量曲线，也不要运行任何模型。文件可放为：

```text
/root/shared-storage/BIT_V3_archive.zip
```

若名称或位置不同：

```bash
export BIT_SOURCE=/实际路径/BIT原始归档
```

结构预检会记录文件、字段、shape、缺失、候选物理周期字段、队列提示与
cycle 190 可表示性；除物理周期字段外，不输出数值型容量或电流范围，
也不计算 SOH、EOL、RMSE 或预测。

### 5. 归档账号

准备：

- 一个 GitHub 仓库；
- Zenodo 与 GitHub 的归档关联；
- BIT 正式冻结时使用的 OSF Registration。

Zenodo 必须填写 specific-version DOI，不能只填概念 DOI。OSF 必须是
真正提交后的 registration URL，不是可继续编辑的普通项目草稿。

## 第一段：XJTU-only SOH 开发与冻结

```bash
cd /root/workspace
unzip MSTT_RUL_v0.2.0_SOH_BIT_freeze_code.zip
cd MSTT_RUL_v0.2.0_SOH_BIT_freeze_20260727
chmod +x manager.sh

bash manager.sh setup
bash manager.sh verify
bash manager.sh prepare_dev
bash manager.sh smoke
bash manager.sh train_soh
bash manager.sh calibrate_soh
bash manager.sh freeze_development
```

主要产物：

```text
/root/shared-storage/MSTT_RUL_v0.2.0_SOH_development_freeze.zip
/root/shared-storage/MSTT_RUL_v0.2.0_SOH_development_freeze.zip.sha256
/root/shared-storage/MSTT_RUL_v0.2.0_SOH_development_freeze.manifest.json
```

把该冻结状态提交到 Git，创建 GitHub release，并在 Zenodo 取得该版本
DOI。然后登记收据：

```bash
bash manager.sh register_development \
  <40位Git提交哈希> \
  <GitHub-release-tag-URL> \
  <Zenodo特定版本DOI>
```

这一登记完成前不要打开 HNEI 容量内容。

## 第二段：HNEI 探索性评估

```bash
bash manager.sh hnei_inventory
bash manager.sh hnei_open
bash manager.sh hnei_evaluate
bash manager.sh hnei_aggregate
bash manager.sh hnei_pack_results
```

HNEI 结果包：

```text
/root/shared-storage/MSTT_RUL_v0.2.0_SOH_HNEI_exploratory_results.zip
/root/shared-storage/MSTT_RUL_v0.2.0_SOH_HNEI_exploratory_results.zip.sha256
```

查看结果后必须明确二选一。

若没有根据 HNEI 修改权重、架构、scaler、数值超参数、cutoff、EOL、
删失或统计方案：

```bash
bash manager.sh hnei_decision unchanged \
  'HNEI仅作探索性评估，v0.2.0未作任何性能驱动修改'
```

若做了任何修改：

```bash
bash manager.sh hnei_decision modified \
  '说明具体修改及其原因'
```

`modified` 会阻止当前版本接触 BIT。此时必须建立 v0.2.1，从
XJTU-only 开发和冻结重新开始；HNEI 对 v0.2.1 只能称开发期探索数据。

## 第三段：BIT 仅做结构预检

只有 HNEI decision 为 `unchanged` 时运行：

```bash
bash manager.sh bit_preflight
```

运行后立即停止。请发送以下三个文件进行结构解析：

```text
11_bit_structure_preflight/bit_structural_preflight.json
11_bit_structure_preflight/bit_file_inventory.csv
11_bit_structure_preflight/bit_field_inventory.csv
```

它们位于默认运行目录：

```text
/root/workspace/MSTT_RUL_v020_SOH_BIT_run/
```

不要自行猜 CSV 行号、参考测试序号或某个含有 `cycle` 字样的字段就是
physical aging-cycle。真实 schema 解析后，再从模板生成：

```text
bit_schema_mapping.yaml
bit_structure_cell_manifest.csv
frozen_bit_parser.py
```

解析器只根据冻结字段读取数据；冻结前只写代码和计算源码哈希，不在 BIT
容量值上执行，也不导入任何模型。

## 第四段：BIT 评估前不可变冻结

只有 schema 和结构电芯清单经核对后运行：

```bash
export BIT_SCHEMA_MAPPING=/实际路径/bit_schema_mapping.yaml
bash manager.sh bit_freeze
```

随后完成 OSF Registration、GitHub Release 与 Zenodo 特定版本归档，
再登记：

```bash
bash manager.sh bit_register \
  <40位Git提交哈希> \
  <GitHub-release-tag-URL> \
  <OSF-registration-URL> \
  <Zenodo特定版本DOI> \
  CONFIRM
```

本包到此停止，没有 `bit_evaluate` 命令。这可以保证当前交付不会在
schema、文件清单和不可变注册完成前意外生成 BIT 模型输出。注册收据
和 BIT 冻结包核验后，再生成只读取冻结内容的 BIT 评估器。

## 状态检查和注意事项

```bash
bash manager.sh status
```

- 不要在原运行目录中手工覆盖权重或 scaler。
- 若必须重新开始，设置一个新的 `MSTT_RUN_ROOT`，保留旧目录作审计。
- 不要删除失败电芯。数据质量排除、190 前 EOL、190 前右删失、190 时
  风险集、RMSE 集和 timing 集必须分别计数。
- 不要把 future cycle 当独立样本；每个物理电芯只有一个主要 RMSE。
- BIT H1 与 H2 使用固定顺序 gatekeeping，不对两者做 Holm。
- 不提供 fixed/arbitrary 普通 pooled paired p 值。
- 主要任务禁止 cutoff 后实际电流计划、未来长度、终止位置、寿命、
  EOL、缺失模式和完整轨迹统计量。
