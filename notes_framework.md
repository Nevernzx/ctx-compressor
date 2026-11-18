# 框架实现相对 PLAN v2 的逐项说明（并入 NOTES.md）

1. 按任务书指定，将片段渲染为 `标题 + "\n" + 原正文`；`text` 与 `ntok` 使用完全相同的渲染串。`paragraph_text` 另存原正文，去重键仍是原始 `(title, paragraph_text)`。BM25 对渲染串分词，标题仅出现一次。预算只约束被选片段的 token 数之和，不包括输入模板、编号标签和任务问题；与 PLAN 的片段预算口径一致。
2. 随机操作只使用显式种子的 `random.Random(seed)`。`simulate_task` 加可选 `seed=0`；`sample_tasks(examples, n_tasks, seed, questions_per_task=4)` 一次无放回抽样后切成任务，任务内与任务间都不重复，保留抽样顺序。重复数据集 ID 或数量不足时报错，不偷偷复用 dev 题目。
3. 原 `#0` 会错误引用最后一个答案，前向引用会原样漏入查询。现在 `hop_queries` 只接受 `1 <= k <= 已完成跳数`；缺失/空/无效 `paragraph_support_idx`、无效答案和证据不在语料库中均报 `ValueError`，调用方可整题跳过并记录原因。`实体 >> 关系` 保留原字符串作为检索查询，不擅自改写。其他引用格式由真实数据核对层识别并保守跳过。
4. `simulate_task` 的候选数取 `min(top_n, 语料库段数)`，修复合成小语料检索 top-20 越界；空语料或非正 top_n 显式拒绝。每步增加原始/替换后查询、题目 ID、零起始题号和跳号等审计字段，原有 new/gold/natural_hit 保留。
5. 原贪心排序在 token 数和已保留状态相同时依赖集合迭代顺序；最后再按稳定编号排序，使结果可复现。首要顺序仍是 token 从少到多，重选模式同长度仍优先已有证据。
6. `apply_action` 与奖励共用严格格式语义：add 必须存在，add/drop 必须为列表，编号必须为 Python int（拒绝 bool/float）；drop 缺省仍视为空列表。重复编号仍按集合处理，与 PLAN 相同。原 apply_action 对非字典、不可哈希元素等可能抛异常，现统一返回 None，让 rollout 按 PLAN 保持原状态。非法模式、非正整数预算也拒绝；零预算原会在奖励里除零。
7. `make_sample` 对 kept/new 各自去重并去掉 new 中已保留项，补齐直接调用时 kept 的稳定编号；正常 PLAN 轨迹行为不变。增加 pid_by_lid 供验收追溯，不渲染给模型。
8. `make_oracle_noise_policy(seed)`：这是 Claude 的理解。只在 reselect 的理想动作上从本步新返回中随机额外加入 1–3 个非证据段，预算不足时少加或不加；旧噪声仍由下一步纯 oracle 目标丢弃。append 返回纯 oracle。rollout 的 sft_target 始终独立使用 oracle_action(s)，不含噪声。
9. 实现 render_input(s) 与 render_target(s)：输入依次列全部问题、模式、预算、已用 token、已保留/新返回两组片段与编号/token 数；不泄漏 gold、oracle、recall、pid_by_lid。目标 JSON 紧凑输出且键固定 add、drop。
10. rollout 增加 action/action_valid/used_after/recall 审计字段，不改变 PLAN 预算、可获得金标准或非法动作保持原状的规则。预算必须正整数；append 已有状态若超预算则显式拒绝，因为在不能 drop 的规则下不可能修复。
11. reward.py 使用包内相对导入 `from .build_traj import apply_action`，以便从仓库根目录执行 `python -m pytest` 或导入 `src.reward`；奖励数值公式不变。eval/cost.py 与 PLAN 公式一致，倍率仍是参数；未增加供应商价格查询、GPU 或模型调用。
