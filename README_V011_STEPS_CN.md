# v0.1.1 揭盲后、模型评估前修订包

这个包不允许篡改原始 `v0.1.0-freeze` 失败结果。

## 正确流程

1. 保留原始 FAIL 证据。
2. 将本包中的两个 Python 文件复制到项目 `src/`。
3. 将修订说明复制到项目 `preregistration/`。
4. 提交新 commit，创建新标签：
   `v0.1.1-postunblinding-pre-evaluation-amendment`
5. 创建 GitHub Release，并让 Zenodo 生成新的版本 DOI。
6. 在任何外部模型评估之前运行
   `register_postunblinding_amendment.py`。
7. 使用 `prepare_external_v011_postunblinding.py` 写入全新的运行目录。
8. 只有新 amended preflight 为 PASS，才运行模型评估。

## 绝对禁止

- 不得把原始 `external_preflight.json` 的 FAIL 改成 PASS。
- 不得删除原始失败输出。
- 不得声称 v0.1.1 是未揭盲预注册确认实验。
- 不得修改冻结模型、超参数、截止点或统计方案。
