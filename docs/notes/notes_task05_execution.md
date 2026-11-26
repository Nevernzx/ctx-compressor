# 任务05执行决策与限制记录

本文件只记录任务05，不修改任务01–04的证据。最终三份主文档只追加。

- 所有实际运算均经 `ssh 5880`；3080只写mirror。旧ctxc/reader/train环境安装内容不变，没有安装或下载包/模型。
- 本任务root Used基线为931440451584 bytes（2026-11-17 14:49:56 UTC），最终限额增量300000000 bytes。物理GPU1初始375 MiB，UUID沿用已核对的GPU-4318dc90-4126-038a-9402-1323533db94f。
- §2“每种跑30步”保守解释为三配置各做新的30-step实测；任务04的100-step结果仅作历史参考。固定480条训练样本，两模式各240，三配置使用同一组原始样本；无需重训任务04基准。
- 40 GB按40,000,000,000 bytes解释。筛选显存取torch reserved峰值加启动前背景占用、以及nvidia-smi整卡采样峰值两者较大值；不把allocated与reserved混写。最长样本探测在30步后，Adam状态驻留，只forward/backward，不增加optimizer step。
- A1（1×16）和A4（4×4）未使用长度分组；DataLoader仍按固定seed洗牌，并非逐原文件行序训练。A4实测显示约45.78 GiB reserved峰值，超过40 GB，且较A1慢。真实ms-swift3.12.6的`prepare_logits_to_keep`对batch>1保留从最早监督位置到batch末尾的整个suffix，变长padding因此增加logits内存。任务书明确允许group_by_length，故在A8首次启动之前为8×2启用该项；不重跑A1/A4、不改权重/超参。三者是三个完整配置，不把差异全归因于batch size；正式训练继承胜出trial的实际flag，长度分组会改变顺序。
- 已启动的比较orchestrator原元信息写group_by_length=False。正式训练入口保留该值为`orchestrator_initial_group_by_length`，并按各trial的实际worker_config填入逐trial flag、`group_by_length_by_batch`和selected flag。保留原记录、解释修正，不修改吞吐/显存实测值。每次worker实际代码均外置保存并核对SHA；A1启动时的源文本已还原且与启动时记录SHA完全相符。
- A1最初未显式关闭ms-swift默认check_model=True。独立禁网拦截核查显示，本地非ModelScope缓存目录被识别为unknown，默认检查路径会尝试GET `models/unknown/revisions` 和 `models/unknown/repo/files`；框架吞掉异常并压制日志，现有A1日志不能确认是否送达服务端。后续A4/A8/正式训练显式`check_model=False`，所有缓存/模型均本地。没有调用外部模型推理API。详见`eval/sft_run/network_guard_audit.json`，不把此元数据检查路径隐瞒成从未可能联网。
- NaN/Inf每个microbatch和optimizer更新前检查，禁用loss日志过滤。任何新worker异常写`training_stop.json`，后续trial及正式训练拒绝继续；不自动重试、改超参或恢复训练。A1旧worker在补强跨进程门禁前已启动，但它随后完整30步与最长样本检查通过，无失败窗口遗留。
- 后台pipeline使用nohup+新POSIX session，PID714854；父进程已变为PID1，不依赖SSH/Codex连接。先三trial后立即正式训练。训练监控线程每30秒记录进程、step、非有限标记、GPU和root磁盘，远严于30分钟上限。所有进程管理仅使用已记录确切PID。
- 训练后另有独立等待器，只在该PID退出且正式结果确认为2500步/40000样本/状态ok后启动留出模型评测。等待器首次PID721529仅因补强zombie退出识别而被确切PID终止并替换为722344；没有终止或重启任何GPU训练。等待器、推理也以nohup和独立session运行，均记录PID和日志。
- 训练抽样使用单一Random(20261118)，固定模式reselect→append、task数值顺序、task内step顺序。每任务先随机取2，剩余530/模式从其余行均匀补齐，最终按源行序逐字节复制。两模式各10000，共20000，每模式覆盖4735任务；训练循环另由固定seed sampler决定顺序。heldout loss子集用独立Random同seed均匀抽1000，只用于loss曲线，不参与选checkpoint。
- 评测只读train corpus/index和已保存heldout任务steps；没有重新检索，没有在dev测试任务上运行模型或评测。旧dev文件仅作为四代SHA完整性保护的一部分读取字节。CPU四策略沿用任务02语义；no_compress为不受预算约束参考线，requested/effective预算分列。
- 模型输出不修复、不抽取markdown JSON，只解析整个原始字符串。schema/类型错误归invalid_json，其后候选ID、append-drop、over-budget依次互斥分类；失败动作no-op，下一步使用自己的保留结果。现有apply_action的额外字段忽略、重复ID集合语义保持。NaN/Infinity、重复JSON对象key不是合法标准JSON，按invalid_json处理。
- 六模型使用同一原生Qwen3-1.7B，vLLM原生LoRA加载，base加5个checkpoint；不合并/复制大权重。greedy/max_tokens256，按步跨所有活动轨迹并行。训练/推理实际prefix token IDs已在两个既有环境核对，1810 masked+4空think=1814；全历史union最大34489，加256仍低于原生40960。每批实际prompt继续硬检查，绝不裁切或丢轨迹。
- checkpoint按6格共1494条轨迹的末端证据丢失率升序、合并末召回降序、步数升序选择；eval loss不参与排序。不做GRPO、阅读F1或dev评测，不对训练质量下结论。

最终状态、精确耗时、显存和选择结果以`eval/sft_run/sft_run_report.md`及JSON实测为准；本记录不提前宣称未完成的训练或评测完成。

## A8零步初始化修复

首次A8在2026-11-17 15:14 UTC构造分组DataLoader时遇到`KeyError: Column length not in the dataset`。真实ms-swift自定义get_train_dataloader要求Dataset有length列；该进程optimizer_steps=0、consumed_samples=0、NaN/Inf=False、OOM=False，没有做任何参数更新。只增加由已核验完整编码长度得到的length元信息，input_ids/labels、样本、seed与超参不变；CPU实测验证模板collator忽略该元信息，未初始化CUDA。

完整失败输出保留在`runs/batch_bench_20261117/initialization_attempts/batch_8_missing_length/`，失败标记归档到`eval/sft_run/batch8_initialization_failure.json`。新增显式`resume-initialization`仅允许这一特定零步错误，并验证A1/A4已有30步结果、原样本SHA及实际worker源SHA；复用结果，不重复优化步。原pipeline714854和等待器722344已退出；15:18:18 UTC重启控制流程PID724147、等待器724148，仍为nohup/独立session。A8若出现NaN或OOM仍停止，不调整超参再试。

## A完成、B后台启动

三配置实际各30步、480条、1,555,774个非padding token，均无NaN/OOM；A1/A4/A8吞吐分别5695.282197、2852.472606、3432.511442 token/s，筛选显存分别7,492,075,520、49,548,361,728、41,059,090,432 bytes。三者最长batch探测均通过且没有额外optimizer更新。选定A1：batch1、累积16、不按长度分组，预计完整129,734,216 token训练循环22,779.243秒（6.3276小时）；模型启动、编码、五次验证和保存另计。

正式进程沿用PID724147，2026-11-17 15:26:37 UTC开始初始化。进程父PID1、独立session724147；实测环境CUDA_VISIBLE_DEVICES=1、三种offline标志均1、代理变量为空，唯一GPU计算进程绑定既定物理GPU1 UUID。证据见formal_process_proof.json。每30秒监测训练资源，但训练资源线程本身不自动按根盘阈值停训；根盘由执行者持续核查，最终按任务初始df计增量。

独立复核澄清`dataset_shuffle=False`只控制数据集构造，ms-swift的`train_dataloader_shuffle=True`仍生效。比较结果中旧`original sample order`描述保留为历史元说明，再修正为“不按长度分组、仍用固定seed的DataLoader洗牌”；不修改执行源归档或实测值。

## 磁盘监测口径与调查

只读挂载、实际进程环境和可写fd核查确认，模型、训练数据、缓存及日志位于独立挂载`/root/userfolder`。已定位的root增量主要包括本任务两轮定向测试的临时目录，合计17,133,568 bytes分配空间，以及小代码/报告；调查时未移动或删除它们。来源复核纠正先前“全套测试”的口头描述：pytest-7与pytest-8分别来自25项与70项定向测试，而非357/419项全套测试。`df /`反映共享宿主nvme文件系统统计，另有约每5分钟4 MB的阶跃尚未定位，不能据此认定是其他任务，也不从最终增量中扣除。调查证据见`eval/sft_run/root_disk_investigation.json`。

任务书要求的是“最终增量≤300 MB”。C原先把任一瞬时超限写成永久监测错误，超出了这一要求，现已在C启动前修正：启动及运行中持续记录原始df、基线增量、峰值、超限/恢复事件；超限本身不中断推理，读写监测失败仍报错。最终验收仍按原始基线931440451584 bytes与最终原始df之差，硬性要求≤300000000 bytes，未扣除任何未归因量。没有修改训练worker、模型参数、GPU门禁或finalizer；修改通过12项定向CPU测试，测试临时目录已显式外置。

独立复核进一步补强了C的退出清理：监测线程退出超时明确失败，最终采样/诊断/写入异常也必须执行引擎关闭，关闭后才发布终态，并保留原始推理异常。最终推理runner SHA256为`649ecd4685a484d976a0f0015b7eb0780989f05809c157757176643097c7d7d5`。22项定向CPU测试和独立只读复核通过；代码稳定后全套436项测试通过（含原264项），该轮pytest临时目录显式放在userfolder，结果见`eval/sft_run/tests.xml`。

2026-11-17 18:18 UTC前后，在确认两处目录属于本任务且无活跃测试/打开句柄后，将`/tmp/pytest-of-root/pytest-7`和`pytest-8`完整归档到`/root/userfolder/ctx-compressor/cache/tmp/task05_pytest_root_archive/`，原路径保留目录符号链接。2537个普通文件（805167逻辑bytes）、1563个目录、39个内部符号链接的相对路径、文件SHA256及链接目标均验证一致；没有丢弃调试内容，没有触碰pytest-0至6或其他目录。实测df在迁移时下降17272832 bytes，计入逐项证据JSON后净下降15859712 bytes；共享文件系统波动限制仍适用。完整证据为`eval/sft_run/pytest_tmp_migration.json`，原调查记录保留。

验证设置为`prediction_loss_only=True`、`predict_with_generate=False`，没有生成问答答案或计算阅读F1。ms-swift原生日志仍附带同一次前向统计的`token_acc`/`eval_token_acc`字段，原始日志完整保留；任务报告和loss图只取loss，checkpoint选优不使用这些附带字段或eval loss。该框架附带日志行为没有引发额外模型调用或训练超参变化。

## 正式权重与推理配置的CPU核验

checkpoint-500/1000的实际safetensors均在CPU上检查：392个FP32张量、196组A/B、28层×7类线性层，形状与Qwen3-1.7B一致，转BF16后仍全部有限；vLLM的PEFTHelper接受rank16/alpha32和现有配置，检查前后文件SHA相同。没有创建LLM或初始化CUDA，实际GPU加载仍由C验证。详见actual_lora_cpu_preflight.json。

实际SamplingParams经vLLM配置合并后仍为greedy、temperature=0、max_tokens=256、seed=20261118，仅补充模型EOS。原模型磁盘generation_config没有max_new_tokens；训练日志中的64来自训练进程内存配置。V1源码将LoRA ID纳入prefix-cache key，base无LoRA、五个adapter用互异ID1–5；源码检查不代替真实GPU backend与执行检查。详见sampling_config_cpu_check.json。

2026-11-17 19:08 UTC实测整卡显存10063MiB，唯一可见本任务训练进程占7146MiB，两种NVML口径存在未归因差额；未杀进程、未调整训练配置，也未从实测峰值扣除差额。保留gpu_accounting_observation.json，C仍须等待本任务训练退出并重新检查整卡空闲门禁。

## 根盘原始增量首次超限记录

2026-11-17T20:42:47.372658+00:00 的训练资源采样首次记录 root_delta_bytes=302219264，超过300000000 bytes。此前元数据调查未找到持续增长来源；未扣除未知增长、未调整最终门槛、未清理其他任务文件。任务书规定最终增量，因此仍继续既定训练及评测；若最终原始df差值仍超限，则最终status必须为fail，即使训练和模型指标已全部完成。原始采样和资源日志前缀SHA见eval/sft_run/root_limit_first_exceeded.json。

## 根盘超限时的失败证据交付

为避免最终root超限时无法交付已完成工作的校验记录，新增独立scripts/export_sft_resource_failure.py，原严格finalizer不变。仅在B/C完成、watcher和所有记录PID均退出、历史只用GPU1、GPU1身份正确且整卡占用<2GiB、exact Task05 compute PID清零等所有非root门槛满足，并在inventory前后都确认原始root增量>300000000时导出；根盘回落则拒绝失败导出并指示使用严格finalizer。final_resource_check明确status=failed/root_limit_passed=false；manifest.complete仅指文件穷举，并明确task_acceptance_status=failed。成功导出仍exit1，不能视为任务通过。报告继续提供完整指标与checkpoint选择，整体维持failed。26项CPU fixture及独立只读复核通过，随后全套462项测试通过，测试临时目录均在userfolder。此阶段未执行真实终态inventory或改变训练/推理。

## B完成与C零输出初始化失败

正式训练已完成2500个optimizer step、40000次样本暴露、129734216个非padding token，epoch=2.0，NaN/Inf=False、OOM=False，五个checkpoint全部保存。同步train_begin→train_end计时25774.874464秒（包含五次eval/save），worker wall计时26053.937670秒；第五次固定1000样本eval loss=0.02731489。worker_result.json是最终依据，progress.json停留在最后on_step_end，不能据其四checkpoint列表误判缺少第五个。

自动等待器随后于22:41 UTC启动C（PID892732，engine892830），GPU预检375MiB通过；vLLM0.11 V1/Flash Attention/PunicaWrapperGPU已经加载本地Qwen3-1.7B，但LoRA dummy profile进入Triton主机C扩展编译时因没有C compiler失败，非OOM且六策略均零输出。现有系统、既有环境和缓存经独立只读复核均缺C前端、assembler和匹配libc开发sysroot，不能只填CC路径。相关PID包括原watcher724148和训练724147均已退出，GPU1恢复375MiB。原始11文件已保留于eval_rollouts/initialization_attempts/missing_c_compiler_20261117T224125Z并逐项SHA核对；归档收据model_initialization_failure_archive.json记录实际零输出/退出及旧CUDA1证据。未重新训练、未丢弃轨迹。

任务05禁止改变ctxc/reader及他人环境安装内容，并未沿用任务02的禁止安装条款。为完成本地LoRA推理，将必要构建工具链独立准备在userfolder/toolchains，缓存/临时文件同样外置，禁代理，不改既有训练/reader/ctxc安装。先CPU编译证明再设置Task05 wrapper，随后复用原wait_train_then_eval.py只启动C；不改生成设置/模板/模型超参，不调用外部模型推理API。

根盘另在22:20:56 UTC短暂增加241053696B，下一采样22:21:26已恢复。其间未做删除/迁移；聚焦元数据与可写fd复核未发现Task05遗漏的root路径，不能据恢复后的快照归因已消失分配，也不扣除未知增长。详见root_disk_transient_20261118.json。

## 外置工具链完成与C重新启动

独立工具链位于/root/userfolder/ctx-compressor/toolchains/gcc13_20261118/envs/gcc，GCC/G++13.4.0、sysroot2.17，19个官方conda-forge包，约1.7GB全部外置。TOOLCHAIN_SHA256SUMS记录74项下载包/关键工具/元数据/CPU证明，需随最终主SHA单独复核。1061个原文件比对为1060字节不变，仅Task05 wrapper授权变更；conda可能touch原registry/history的元数据，未宣称零root元数据接触。C/C++程序、reader Python扩展answer42和实际Triton driver.c编译加载均通过，未调用CUDA方法。独立只读复核通过。

Task05 wrapper仅增加CC/CXX/PATH及两条可执行检查，未改LD_LIBRARY_PATH，最终SHA4020ac2387b865381458c02fa7b9af0c68b605d2f7853e5f0670ec30da017ab6；runner仍649ecd4685a484d976a0f0015b7eb0780989f05809c157757176643097c7d7d5。22:53:38 UTC重新启动既有wait_train_then_eval.py，watcher898552、runner898553、engine898637；再次启动前GPU1为375MiB。没有重跑B。实际V1/Flash Attention/PunicaWrapperGPU初始化22.989秒完成，engine tokenizer逐token校验通过。

运行证据见evaluation_launch_engine_runtime.json：启动进程离线标志、禁代理、CC路径已实读；已改标题VLLM::EngineCore的/proc/environ不暴露所选字段，因此不以该空读取声称worker变量为空或已验证其值。独立NVML实测engine PID绑定既定物理GPU1 UUID。此条仅记录C启动，不提前宣称六策略评测完成。

## 外置工具链完整依赖清单补充

任务05 §5要求大文件记录路径与SHA256，但未逐字列出新增编译工具链整棵树。原 `TOOLCHAIN_SHA256SUMS` 的74项覆盖19个下载包和关键工具、元数据、CPU证明，不等同于实际解包后的全部头文件、库和sysroot覆盖。为使C阶段新增的独立构建依赖可审计，另补固定工具链根目录的完整清单；保留原74项清单，不扩大严格finalizer的三个产物根目录，也不把依赖清单完成视作任务验收通过。

新增CPU审计脚本为 `/root/workplace/ctx-compressor/scripts/inventory_sft_toolchain.py`，SHA256为 `a19529e1dfea7ccfa045f105e516679edaa38e0fb8941be10062e2656fb53d99`。它只清点 `/root/userfolder/ctx-compressor/toolchains/gcc13_20261118`，不跟随符号链接、不调用编译器或GPU、不扫描C输出；执行前后核对可见进程的toolchain写入FD，未发现活动写入，并验证全部文件集、元数据和链接目标稳定。

完整依赖manifest为 `/root/workplace/ctx-compressor/eval/sft_run/toolchain_full_artifact_manifest.json`，大小1,938,386 bytes，SHA256为 `cfb4c85776f3201c2c9cf111ab4c278e4c57e283313cfa8e625dbadbb4f89e97`。已同步mirror，外置工具链本体未复制。清单逐项记录相对固定root的普通文件路径、bytes、SHA256，以及不解引用的符号链接原始目标；共9,887个普通文件路径、230个符号链接、784个目录（含root），对应5,522个普通文件唯一inode。

大小分三种口径：按路径累计逻辑字节为2,349,284,307 bytes，硬链接会重复计入；按唯一inode去重后的普通文件逻辑字节为1,744,699,797 bytes；`du -sx -B1`的整树分配空间为1,761,914,880 bytes（包含目录和链接分配）。这三项均不是本任务新增根盘占用，也不代替最终原始df增量。Manifest另保存唯一普通inode的st_blocks分配空间及实际哈希读取字节，以免混淆逻辑大小和分配大小。

历史462项测试收据中的82个Python源码SHA仍逐项匹配。新审计脚本是在该次pytest后新增，未被462项历史测试覆盖；清单明确记录这一边界，验证依据为本次真实全树执行、逐文件哈希及前后稳定性/写FD核验，没有重复全套测试或伪称已有测试覆盖新脚本。独立eval agent随后只读复核脚本和manifest，确认固定路径、完整覆盖、前后稳定性、硬链接计数与源码SHA一致，无阻断；该复核未重跑测试、重建清单或使用GPU。

Manifest状态为 `additional_dependency_inventory_complete`，`task_acceptance_status=not_evaluated`。`complete`仅指这棵额外依赖树的穷举清单完成；C最终闭环结果、进程退出、GPU释放、原始根盘300000000-byte门槛及最终任务状态仍须分别验收。


### C 期间 GPU 共享占用观测（2026-11-18T01:42:51.239621+00:00）

资源日志在 01:41 UTC 左右记录整卡显存短暂升至 46307 MiB。随后只读 NVML 核对，本任务 engine PID 898637 约 34700 MiB，另有非本任务进程占用；峰值时缺少逐进程快照，不把整个峰值归因给任何单一进程。观测收据为 `eval/sft_run/evaluation_gpu_accounting_20261118T0142.json`，附原资源日志已读前缀长度/SHA与窗口内样本。未管理其他进程、未改超参，推理仍运行且未报告 OOM。GPU 观测不用于解释或扣除来源未明的根盘增量。


## 任务05最终交付：计算完成，资源验收未通过

记录时间：2026-11-18T04:02:22.428837+00:00。A/B/C 计算与独立审计均已结束；任务整体状态为 **failed**，原因是资源验收采样中 `/` 的原始增量为 **834,052,096 bytes**，超过 **300,000,000 bytes**。未扣除未知来源占用。只使用物理 GPU 1；16 个记录到的本任务 PID 已退出，终态 GPU 占用 375 MiB，本任务 GPU 进程为 0。

全部6模型实际结束，模型输出总n=83736。全量独立审计以及最终报告轻量一致性检查通过，选择 `/root/userfolder/ctx-compressor/runs/sft_20k_20261117/checkpoints/checkpoint-2000`。最终资源失败导出记录根盘增量834052096 bytes，GPU已释放，151项外置文件完整。主报告为failed且missing_inputs为空，loss CSV/SVG仍为2500个train点和5个heldout点，无平滑。最终文档追加及SHA/mirror交付从此处继续，详见主README/NOTES/HANDOFF的本节。
