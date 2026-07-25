# 先从这里开始

## 当前状态

- Batch‑4/5/6 的 10 个压缩包和 24 个预期 MATLAB 文件已经通过只读完整性核验。
- 尚未解析任何 Batch‑4/5/6 容量、寿命、`summary` 或实验结果。
- 当前存储文件中缺少开发集 `xjtu_q2_prepared_csv.zip`。它必须来自此前
  Batch‑1/2/3 的 31 条逐电芯 CSV，不能从结果表反推。
- 在取得 `v0.1.0-freeze` 的 Zenodo **版本 DOI** 前，禁止运行
  `bash manager.sh open_external`，也不要手工解压 Batch‑4/5/6。

## 第一步：补齐开发集

如果服务器已经有 `xjtu_q2_prepared_csv.zip`，放到：

```text
/root/shared-storage/xjtu_q2_prepared_csv.zip
```

如果只有原来的 31 个 CSV，先解压代码包，然后运行：

```bash
cd /root/workspace/MSTT_RUL_Q2_Batch456_confirmatory_20260725
bash manager.sh setup
bash manager.sh make_dev_zip /精确路径/31个源CSV所在目录
```

该命令要求 Batch‑1/2/3 数量严格为 8/15/8。

## 第二步：冻结前运行

确认 `/root/shared-storage/` 中已有 10 个外部压缩包和开发集 ZIP 后：

```bash
cd /root/workspace/MSTT_RUL_Q2_Batch456_confirmatory_20260725
chmod +x manager.sh
bash manager.sh setup
bash manager.sh verify
bash manager.sh inventory
bash manager.sh prepare_dev
bash manager.sh smoke
bash manager.sh train_freeze
```

训练是后台任务。反复运行：

```bash
bash manager.sh status
```

当 `frozen_model_manifest.json: PASS` 后：

```bash
bash manager.sh calibrate
```

当 `calibration_quantiles.json: PASS` 后，填写预声明签名、许可证和
`CITATION.cff`，再运行：

```bash
bash manager.sh freeze_pack
```

## 第三步：发布冻结版本

把冻结后的完整目录提交到新的公开 GitHub 仓库，创建
`v0.1.0-freeze` release，并等待 Zenodo 生成该 release 的版本 DOI。
发布命令见 `templates/RELEASE_COMMANDS.md`。

## 第四步：DOI 生成后才打开外部数据

```bash
bash manager.sh register_receipt <40位完整commit> <Zenodo版本DOI>
bash manager.sh open_external
bash manager.sh evaluate
```

再次用 `bash manager.sh status` 等待 `evaluation_audit.json: PASS`，然后：

```bash
bash manager.sh aggregate
bash manager.sh pack_results
```

最终交回这两个文件，不要挑选或删除不理想结果：

```text
/root/shared-storage/MSTT_RUL_Q2_Batch456_confirmatory_results.zip
/root/shared-storage/MSTT_RUL_Q2_Batch456_confirmatory_results.zip.sha256
```
