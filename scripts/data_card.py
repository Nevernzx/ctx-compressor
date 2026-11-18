"""Assemble the Chinese data card from verified remote metadata and audit reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "data/raw/MANIFEST.json").read_text())
    verification = json.loads((ROOT / "eval/acceptance/data_verification.json").read_text())
    if not verification["passed"]:
        raise SystemExit("Refusing to produce a completed data card before verification passes")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    source_notes = (ROOT / "data_notes.md").read_text()
    source_sections = "## 最终结果" + source_notes.split("## 最终结果", 1)[1].split("## 失败地址与保守处理", 1)[0]
    lines = ["# 数据卡：任务 01", "",
             "数据只存放在 5880；本卡不包含原始数据副本。所有文件来自免费直连来源，原始格式未转换。",
             f"最终离线核验：**{len(verification['files'])}/{len(entries)} 文件通过**，错误 {len(verification['errors'])}。详见 `data/raw/MANIFEST.json` 与 `eval/acceptance/data_verification.json`。", "",
             source_sections, "## 单条样例的实际字段结构", "",
             "下列结构直接取自 MANIFEST 中相应文件第一条记录的解析结果；值为实际类型名，空数组保留为空，未臆造其中的元素类型。", ""]
    examples = (
        "musique/musique_ans_v1.0_train.jsonl",
        "musique/musique_ans_v1.0_test.jsonl",
        "hotpotqa/hotpot_dev_distractor_v1.json",
        "hotpotqa/hotpot_dev_fullwiki_v1.json",
        "2wikimultihopqa/dev.json",
    )
    for name in examples:
        lines.extend([f"### `{name}`", "", "```json",
                      json.dumps(entries[name]["sample_field_structure"], ensure_ascii=False, indent=2),
                      "```", ""])
    audit = (ROOT / "eval/acceptance/data_audit.md").read_text()
    summary = audit.split("### 2 / 3 / 4 跳各 5 条原始 question_decomposition", 1)[0]
    lines.extend([summary,
                  "2 / 3 / 4 跳各 5 条完整原始分解样例已逐条打印在 [README](README.md) 和 [数据核对全文](eval/acceptance/data_audit.md)。", "",
                  "下载失败尝试、可复现命令与各步耗时见 [HANDOFF](HANDOFF.md)；镜像许可元数据冲突与保守处理见 [NOTES](NOTES.md)。", ""])
    (ROOT / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"DATA_CARD.md generated from {len(entries)} verified manifest entries")


if __name__ == "__main__":
    main()
