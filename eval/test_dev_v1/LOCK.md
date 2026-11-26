# MuSiQue dev 最终测试集锁定记录

no strategy has been run on this set; first use requires a Claude-written task

原始 dev 共 2417 题。历史排除 612 题，包含任务01验收 12 题、任务02/03可行性 600 题，两者交集 12 题。其他历史产物额外排除 12 题，逐文件来源见 exclusion_audit.json。

按原始 dev 顺序取剩余 ID，用 random.Random(20261122).shuffle 后连续每四题分组。锁定 451 个任务（n=1804 题），丢弃尾部 1 题；预算 1876 / 2814 / 5628。

先复现任务02的 150 个任务，完整任务文件逐字节相同。历史 steps.jsonl 中 30286 行（n=1594 个检索步骤，每步 19 个配置）均与复现的有序返回列表及金标、题序和跳序相同。

结构量、证据 token P90、训练集逐字重合率及分母见 structural_stats.json。

## SHA256

- /root/userfolder/ctx-compressor/data/test_dev_v1/test_tasks.jsonl  f2e1856acb54991badddab06ebea99bba942e1c51aaa429d69756751d0ffc71a
- /root/userfolder/ctx-compressor/data/test_dev_v1/steps.jsonl  e285d284d4baed60b09bfc7cae73b469b4b1e0e0d966468116faa2315f41293b
- /root/userfolder/ctx-compressor/data/test_dev_v1/excluded_ids.json  1d0e27ddb8f88c4d53374f3a1257fb265836fe79135c21629afeccdaa9915ad9
- /root/workplace/ctx-compressor/eval/test_dev_v1/exclusion_audit.json  aa114c1e677293bc9e30a4947b4c2ea3efb9c4074e6865bf473228a578edcc19
- /root/workplace/ctx-compressor/eval/test_dev_v1/structural_stats.json  70d0d86e2dd3412334248e1f456fee2f45f59ce1ac163da500fd2704463fc80a
