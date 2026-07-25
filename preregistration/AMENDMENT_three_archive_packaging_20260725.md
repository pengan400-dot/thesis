# 预声明修订：外层压缩包封装形式

修订时间（UTC）：`2026-07-25T15:03:47.039965+00:00`

## 修订原因

外部数据实际保存为 3 个原始外层压缩包，而不是早期清单中的 10 个外层压缩包。本修订只调整外层封装清单，不改变 24 个预期 MATLAB 文件、批次角色、任务、模型、超参数、截止点、统计方案或排除规则。原始文件名保持不变。

## 实际外层压缩包

- `Batch-4.7z`: `e5fb9fffb2c77c62d8c8bcd59ccb4677a23effcb3fad02194c14fe7136d26e5e`，8 个 `.mat`
- `Batch-5.7z`: `9a2be3de953c5aa81c335dfda3656551674acbbab6139b8cd60121cac78ac2a1`，8 个 `.mat`
- `Batch-6.7z`: `8aa15fffbc0b22e6b7e0bc26ef855169747e01195a7356e8e06a9fdb8cc1a29b`，8 个 `.mat`

## 未揭盲声明

本次只计算外层 SHA-256、执行 7z 完整性测试，并读取压缩包目录中的文件名及未压缩字节数。未解压或解析 MATLAB payload，未读取 `summary`、`data`、容量、寿命、描述或任何实验结果。

原 10 压缩包清单保存在 `manifests/library_archive_inventory_20260725.original_10_archives.csv`；当前有效清单为 `manifests/library_archive_inventory_20260725.csv`。
