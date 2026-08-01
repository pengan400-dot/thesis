# BIT 结构解析审计

## 结论

**当前 v0.2.0 的物理周期 190 主 landmark 不能冻结。**

上传的结构预检包 SHA-256 已核验为：

`15c62dceb4f9c9123b787ea481992a55ae0bfbcd5d92fc3de7be2aba86256622`

原始 BIT 压缩包在预检记录中的 SHA-256 为：

`9701ff850b8b738f9b15b5e02d1ea6fe2e3630ef704336f19acc918f133b18d7`

## 结构事实

- 官方元数据：77 个电芯，55 个 arbitrary-use，22 个 fixed-profile。
- 当前归档：73 个物理电芯目录，55 个 arbitrary-use，18 个 fixed-profile。
- 缺失目录：#10, #13, #16, #19。
- 不完整目录：#2，其中 #2 只有 first20 工作簿。
- 每个完整电芯包含一个 first20 工作簿和一个 later 工作簿。
- first20 的本地 `Cycle_Index` 范围为 1–21。
- later 的本地 `Cycle_Index` 范围为 1–101。
- 即使采用最宽松的相加上界，完整电芯也只有 122，仍低于 190。
- 145 个 XLSX 的 `physical_cycle_190_representable` 均为 false。

## 已安全解析的字段

- 工作表：`记录表`
- 本地周期：`Cycle_Index`
- 容量：`Capacity(Ah)`
- 电流：`Current(A)`
- 电压：`Voltage(V)`
- 温度：`Temperature(℃)`
- 时间：`Date_Time`
- 电芯 ID：目录中的 `#<整数>`
- 队列：父目录映射为 `arbitrary_use` 或 `fixed_profile`
- BIT 标称容量：2.4 Ah

## 现在不能做什么

1. 不能在不改协议的情况下执行 `bash manager.sh bit_freeze`。
2. 不能执行 `bash manager.sh bit_register`。
3. 不能运行任何 BIT 模型。
4. 不能把 190 硬解释成行号、局部记录号或别的方便数字。数字不会因为论文着急就自动变得诚实。

## 合规修复路径

建立 **post-BIT-structure-only, pre-model-evaluation amendment**：

- 建议版本：`v0.2.1-BIT-structural-feasibility-amendment`
- 主 landmark：从已预设的 70/130/190 中改为 70
- 130 和 190 保留为“结构上不可评估”，不产生结果
- 任意工况仍为主队列
- 固定工况仍为关键次队列
- #2 结构性排除
- #10、#13、#16、#19 记录为归档缺失，不伪造补入
- 所有模型、权重、scaler、SOH 公式、EOL=0.8、H1→H2 顺序和统计规则保持不变

本包是**审计与修订草案**，不是 BIT freeze，也不会生成模型输出。
