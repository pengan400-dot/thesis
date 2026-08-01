# 使用说明

这个压缩包不是最终 BIT freeze，而是结构性阻断审计和 v0.2.1 修订草案。

## 服务器上先做

```bash
cd /root/workspace/MSTT_RUL_v0.2.0_SOH_BIT_freeze_20260727

python3 /path/to/verify_bit_structure_blocker.py   /root/workspace/MSTT_RUL_v020_SOH_BIT_run/11_bit_structure_preflight/bit_structural_preflight.json
```

脚本以退出码 3 结束是预期行为，表示旧的 cycle-190 freeze 被阻断。

## 不要做

```bash
bash manager.sh bit_freeze
bash manager.sh bit_register
```

下一步应在新分支中建立结构可行性修订，主 landmark 改为 cycle 70，并重新生成与注册 pre-model-evaluation freeze。
