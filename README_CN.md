# MSTT_RUL：XJTU Batch‑4/5/6 前瞻冻结与外部验证

本包把“冻结前”和“打开外部数据后”强制分成两个阶段。默认路径适配你此前的 Linux 服务器：

```text
/root/shared-storage/   持久存储与压缩包
/root/workspace/        训练工作目录
```

可以通过环境变量 `MSTT_SHARED_ROOT` 和 `MSTT_WORK_ROOT` 修改。

## 数据角色

- **Batch‑4**：主要跨工况外部确认。官方说明表明每个循环后均测量参考容量，因此可以按物理循环执行固定截止点递归预测。
- **Batch‑5**：次要随机游走、稀疏容量观测迁移。
- **Batch‑6**：次要 GEO 卫星工况、稀疏容量观测迁移。
- Batch‑5/6 绝不与 Batch‑4 合并成一个“24 电芯同任务检验”。

稀疏观测只使用最近一次已经发生的容量测试作为模型输入；不使用线性、三次或双向插值。误差仅在真实参考容量测试点计算，EOL 作为区间删失事件处理。

## 服务器需要的文件

把本代码包解压到 `/root/workspace/`，并确保 `/root/shared-storage/` 中有：

```text
Batch-4(1).7z
Batch-5(1).7z
Sim_satellite_battery-1.7z
...
Sim_satellite_battery-8.7z
xjtu_q2_prepared_csv.zip
```

`xjtu_q2_prepared_csv.zip` 是此前 Batch‑1/2/3 的 31 条逐电芯 CSV。训练只使用 B1+B3；B2 已经被多次查看，只保留为既往开发证据。

如果服务器上没有这个 ZIP，但还保留那 31 个源 CSV 所在目录，可先运行：

```bash
bash manager.sh setup
bash manager.sh make_dev_zip /精确路径/31个源CSV所在目录
```

脚本会先验证 Batch‑1/2/3 数量必须为 8/15/8，再创建
`/root/shared-storage/xjtu_q2_prepared_csv.zip`；不会从既有结果表反推或伪造训练曲线。

## 现在运行：冻结前阶段

进入代码目录：

```bash
cd /root/workspace/MSTT_RUL_Q2_Batch456_confirmatory_20260725
chmod +x manager.sh
```

依次运行：

```bash
bash manager.sh setup
bash manager.sh verify
bash manager.sh inventory
bash manager.sh prepare_dev
bash manager.sh smoke
bash manager.sh train_freeze
```

作用：

1. `verify` 检查代码语法、K=1/K=5 支持一致性、有限样本分位数和右删失逻辑；
2. `inventory` 只校验压缩包、文件名和散列，不解析 `.mat`；
3. `prepare_dev` 只读取已经暴露的 B1/B2/B3 CSV；
4. `train_freeze` 在 B1 训练、B3 选轮数，再以 B1+B3 固定轮数重训；
5. K=1 与 K=5 都要求未来 5 步真实目标存在，训练样本支持完全一致。

训练期间查看：

```bash
bash manager.sh status
tail -f /root/workspace/mstt_b456_freeze_train.log
```

`train_freeze` 是后台任务。等待 `bash manager.sh status` 显示
`frozen_model_manifest.json: PASS` 后，再运行：

```bash
bash manager.sh calibrate
```

`calibrate` 也是后台任务。它对 B1+B3 做四折电芯级交叉拟合，并在每个外折内重新完成 B1 训练/B3 轮数选择，留出的校准电芯不参与该折选择。等待状态显示
`calibration_quantiles.json: PASS` 后，先完成下面三项元数据：

1. 填写 `preregistration/xjtu_b456_preregistration_v0.1.md` 末尾的姓名、ORCID 和日期；
2. 按 `LICENSE_DECISION_REQUIRED.md` 核对代码来源并决定许可证；
3. 填写 `CITATION.cff.template` 中的作者、仓库、日期和许可证；删除尚未生成的 DOI 字段，另存为 `CITATION.cff`。

确认这些内容不再需要修改后，才运行：

```bash
bash manager.sh freeze_pack
```

该命令生成待发布的 `v0.1.0-freeze` 包及 SHA‑256，并把冻结模型、校准结果和审计文件复制到仓库内的 `freeze_artifacts/`。若服务器中断，可重跑相应命令；已经通过哈希和审计门禁的训练作业会跳过。

## 冻结发布：必须由你完成

只有在元数据填写完成且 `freeze_pack` 成功后，才能提交 Git：

```bash
git init
git add .
git commit -m "Freeze Batch-4/5/6 confirmatory protocol before outcome access"
git rev-parse HEAD
```

把代码推到公开 GitHub 仓库，创建 `v0.1.0-freeze` release，并让 Zenodo 为该 release 生成**版本 DOI**。完整命令见 `templates/RELEASE_COMMANDS.md`。

得到 40 位 commit 和 Zenodo 版本 DOI 后运行：

```bash
bash manager.sh register_receipt \
  0123456789abcdef0123456789abcdef01234567 \
  10.5281/zenodo.12345678
```

脚本会把 commit、DOI、冻结包 SHA‑256 和时间写入解锁凭据。没有有效凭据时，外部 `.mat` 提取程序会拒绝运行。

## DOI 生成之后运行：只打开一次

```bash
bash manager.sh open_external
bash manager.sh evaluate
```

`evaluate` 是后台任务。等待 `evaluation_audit.json: PASS` 后再运行：

```bash
bash manager.sh aggregate
bash manager.sh pack_results
```

最终结果：

```text
/root/shared-storage/MSTT_RUL_Q2_Batch456_confirmatory_results.zip
/root/shared-storage/MSTT_RUL_Q2_Batch456_confirmatory_results.zip.sha256
```

把这两个文件交回审计，不要先挑选或删除不理想的电芯。

没有版本 DOI 之前，`open_external` 会被门禁拒绝。请不要绕过，也不要手工解压或查看 Batch‑4/5/6 的 `.mat`。

## 冻结模型

| 模型 | 用途 |
|---|---|
| `mstt_single_scale_K5` | 主要预选模型 |
| `mstt_full_K5` | 原完整拓扑的次要比较 |
| `mstt_full_K1` | 同目标支持的单步损失控制 |
| `local_linear_trend` | 透明强基线 |
| `persistence` | 透明基线 |

三个 MSTT 模型均使用种子 `42/2024/3407`。主指标在种子平均轨迹上以物理电芯为单位计算。

## 证据边界

这是一套新的、可公开复现的确认性实现，batch size 固定为 64。它不会把此前失败的 batch-size=128/CUDA 锚点伪装成原论文训练源码复现。旧主表仍是归档结果；本包产生的是新的外部验证证据。

Batch‑4 只有 8 个电芯，显著性检验能力有限。正文应优先报告配对效应和电芯 bootstrap 区间，Wilcoxon p 值作为补充，不应把“不显著”写成“等价”。

若某个电芯在最后一次真实参考容量测量时仍未低于 1.6 Ah，代码会保留该电芯并按右删失处理；不会为了让门禁通过而删除它。删失感知时间分数只惩罚已经被观测事实否定的过早预警。
