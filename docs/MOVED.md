# 文件位置对照（GitHub 版本）

为了让根目录只保留面向读者的内容，2026-11-26 把下列文件挪进了 `docs/` 和 `requirements/`。只调整了位置和相对链接，内容没有改动。

| 原位置（根目录） | 现位置 |
|---|---|
| `README.md`（逐任务实验记录） | `docs/LAB_NOTEBOOK.md` |
| `NOTES.md` | `docs/NOTES.md` |
| `HANDOFF.md` | `docs/HANDOFF.md` |
| `SHA256SUMS` | `docs/SHA256SUMS` |
| `DATA_CARD.md` | `docs/DATA_CARD.md` |
| `data_notes.md` | `docs/data_notes.md` |
| `notes_*.md`（6 份） | `docs/notes/` |
| `requirements.lock.txt` | `requirements/lock.txt` |
| `requirements.reader.lock.txt` | `requirements/reader.lock.txt` |
| `requirements.train.lock.txt` | `requirements/train.lock.txt` |

实验时的工作目录仍保持原来的布局。所以 `scripts/`、`tests/` 和各任务签名清单（`eval/*/TASK*_SHA256SUMS`）里引用的仍是**原位置**，按上表换算即可。
