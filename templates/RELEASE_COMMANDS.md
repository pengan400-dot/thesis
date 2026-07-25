# v0.1.0-freeze 发布命令

在运行 `freeze_pack` 之前，先完成预声明签名、许可证决定和
`CITATION.cff`。冻结后不要再修改配置、代码、模型、校准文件或这些元数据。

先在 GitHub 创建一个**空仓库**，不要添加 README 或许可证，然后在代码根目录执行：

```bash
git init
git add .
git commit -m "Freeze Batch-4/5/6 confirmatory protocol before outcome access"
git branch -M main
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
git rev-parse HEAD
```

在 Zenodo 的 GitHub 设置中启用该仓库，然后创建并推送冻结标签：

```bash
git tag -a v0.1.0-freeze -m "Preregistered freeze before Batch-4/5/6 outcome access"
git push origin v0.1.0-freeze
```

在 GitHub 创建同名 release。等待 Zenodo 完成归档，取得**该版本 DOI**。不要只记录“所有版本通用 DOI”。

收到 DOI 后：

```bash
bash manager.sh register_receipt \
  "$(git rev-parse HEAD)" \
  "10.5281/zenodo.<VERSION_ID>"
```

生成的冻结收据会自动进入最终结果包；不要修改已经归档的 `v0.1.0-freeze` 标签或 release。可在实验完成后的 `v1.0.0` 中同时记录冻结 DOI 和最终结果。

最终实验完成后再创建 `v1.0.0`，论文同时引用：

1. `v1.0.0` GitHub release URL；
2. `v1.0.0` 完整 commit；
3. `v1.0.0` Zenodo 版本 DOI；
4. 补充材料中的 `v0.1.0-freeze` DOI。
