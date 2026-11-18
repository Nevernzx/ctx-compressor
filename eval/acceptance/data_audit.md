## MuSiQue 原始数据核对

核对时间（UTC）：2026-11-17T03:31:12.032983+00:00。仅做任务 01 规定的字段、重叠及段落长度核对。

MuSiQue-Ans train / dev：19,938 / 2,417 题。
按 `(title, paragraph_text)` 精确去重后：train **84,559** 段，dev **21,100** 段。

| 重叠口径 | 交集数 | train 分母及比例 | dev 分母及比例 |
|---|---:|---:|---:|
| 标题 + 正文完全一致 | 3,697 | 84,559；4.372095% | 21,100；17.521327% |
| 标题完全一致（唯一标题） | 4,260 | 71,090；5.992404% | 17,629；24.164729% |

共有标题对应 train 13,896 段（占 train 去重段落 16.433496%）、dev 7,453 段（占 dev 去重段落 35.322275%）；标题可对应不同正文。去重和重叠均不做大小写、空白或 Unicode 归一化。

dev 去重段落按 `标题 + 换行 + 正文` 渲染；Qwen3-1.7B tokenizer.json 离线分词，`add_special_tokens=False`。P50 / P90 / P99 = **103 / 205 / 363.01 token**；共 21,100 段。分位数采用 NumPy 线性插值；预算是这些片段 token 数的和，不含输入模板元数据。

### 子问题占位符核对

dev 跳数分布：`{"2": 1252, "3": 760, "4": 405}`。
`#k` 出现在 2,417 题、3,636 跳，共 3,987 次；`>>` 出现在 1,594 题、2,304 跳。
`#k` 按题内第 k 个已完成子问题的答案替换；`实体 >> 关系` 保留为普通检索字符串。
疑似其他引用写法：0 题 / 0 跳。检测范围：非标准 #、[n]、$n、数字或 ref/answer/hop XML 标签、花括号引用；无法由有限正则证明不存在任意自然语言引用。
`paragraph_support_idx` 缺字段 0 跳、空值 0 跳，涉及 0 题。
经过引用、证据索引和语料映射校验，可用于模拟 2,417 题，保守跳过 0 题；原因计数：`{}`。

### 2 / 3 / 4 跳各 5 条原始 question_decomposition

按 dev 文件顺序取各类前 5 题；以下保留原始字段及占位符，供人工核对。

#### 2 跳

题号：`2hop__460946_294723`

问题：Who is the spouse of the Green performer?

```json
[
  {
    "id": 460946,
    "question": "Green >> performer",
    "answer": "Steve Hillage",
    "paragraph_support_idx": 10
  },
  {
    "id": 294723,
    "question": "#1 >> spouse",
    "answer": "Miquette Giraudy",
    "paragraph_support_idx": 5
  }
]
```

题号：`2hop__252311_366220`

问题：Who founded the company that distributed the film UHF?

```json
[
  {
    "id": 252311,
    "question": "UHF >> distributed by",
    "answer": "Orion Pictures",
    "paragraph_support_idx": 10
  },
  {
    "id": 366220,
    "question": "#1 >> founded by",
    "answer": "Mike Medavoy",
    "paragraph_support_idx": 6
  }
]
```

题号：`2hop__701895_752697`

问题：What administrative territorial entity is the owner of Ciudad Deportiva located?

```json
[
  {
    "id": 701895,
    "question": "Ciudad Deportiva >> owned by",
    "answer": "Nuevo Laredo",
    "paragraph_support_idx": 11
  },
  {
    "id": 752697,
    "question": "#1 >> located in the administrative territorial entity",
    "answer": "Tamaulipas",
    "paragraph_support_idx": 3
  }
]
```

题号：`2hop__259228_793698`

问题：Where is Ulrich Walter's employer headquartered?

```json
[
  {
    "id": 259228,
    "question": "Ulrich Walter >> employer",
    "answer": "German Aerospace Center",
    "paragraph_support_idx": 14
  },
  {
    "id": 793698,
    "question": "#1 >> headquarters location",
    "answer": "Cologne",
    "paragraph_support_idx": 11
  }
]
```

题号：`2hop__481349_302087`

问题：Which company owns the manufacturer of Learjet 60?

```json
[
  {
    "id": 481349,
    "question": "Learjet 60 >> manufacturer",
    "answer": "Bombardier Aerospace",
    "paragraph_support_idx": 10
  },
  {
    "id": 302087,
    "question": "#1 >> owned by",
    "answer": "Bombardier Inc.",
    "paragraph_support_idx": 5
  }
]
```

#### 3 跳

题号：`3hop1__454441_55349_651302`

问题：What is the birthplace of the man who does the voice of Stan on the series that includes the episode The Hobbit?

```json
[
  {
    "id": 454441,
    "question": "The Hobbit >> part of the series",
    "answer": "South Park",
    "paragraph_support_idx": 0
  },
  {
    "id": 55349,
    "question": "who does the voice of stan on #1",
    "answer": "Trey Parker",
    "paragraph_support_idx": 7
  },
  {
    "id": 651302,
    "question": "#2 >> place of birth",
    "answer": "Denver",
    "paragraph_support_idx": 9
  }
]
```

题号：`3hop1__145290_443779_52195`

问题：Who is the president of the new country who has jointly established a Commission on Truth and Friendship with the birth country of Satyo Husodo?

```json
[
  {
    "id": 145290,
    "question": "Where was Satyo Husodo born?",
    "answer": "Indonesia",
    "paragraph_support_idx": 8
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 7
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 16
  }
]
```

题号：`3hop1__105767_443779_52195`

问题：Who is the president of the newly declared independent country that is part of the commission of truth and friendship with the country that eats Kemplang?

```json
[
  {
    "id": 105767,
    "question": "Which was the country for Kemplang?",
    "answer": "Indonesia",
    "paragraph_support_idx": 0
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 3
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 19
  }
]
```

题号：`3hop1__105126_443779_52195`

问题：Who is the president of the newly declared independent country part of the Commission of Truth and Friendship with the country where Gado-gado is found?

```json
[
  {
    "id": 105126,
    "question": "The country for Gado-gado was what?",
    "answer": "Indonesia",
    "paragraph_support_idx": 13
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 9
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 2
  }
]
```

题号：`3hop1__103881_443779_52195`

问题：Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country that Tony Gunawan is from?

```json
[
  {
    "id": 103881,
    "question": "Where was Tony Gunawan from?",
    "answer": "Indonesia",
    "paragraph_support_idx": 17
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 7
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 16
  }
]
```

#### 4 跳

题号：`4hop1__93963_170667_443779_52195`

问题：Who is the president of the newly declared independent country, that established the Timor Leste Commission of Truth and Friendship, with the country containing the airport that includes Lion Air?

```json
[
  {
    "id": 93963,
    "question": "What airport is Lion Air part of?",
    "answer": "Juanda International Airport",
    "paragraph_support_idx": 13
  },
  {
    "id": 170667,
    "question": "#1 >> country",
    "answer": "Indonesia",
    "paragraph_support_idx": 3
  },
  {
    "id": 443779,
    "question": "#2 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 14
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #3",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 11
  }
]
```

题号：`4hop3__862_846_326964_7713`

问题：How long was the city where the Yongle emperor greeted the person the Ming court thought the representatives were sent by the capitol of Yaxing Coach's headquarters location?

```json
[
  {
    "id": 862,
    "question": "Who did the Ming court think the representatives were sent by?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 0
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 3
  },
  {
    "id": 326964,
    "question": "Yaxing Coach >> headquarters location",
    "answer": "Yangzhou",
    "paragraph_support_idx": 6
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 15
  }
]
```

题号：`4hop3__862_846_613770_7713`

问题：How long was the place where the Yongle Emperor greeted who the Ming court thought representatives were sent by the capitol of the city in which Guangling District is located?

```json
[
  {
    "id": 862,
    "question": "Who did the Ming court think the representatives were sent by?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 2
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 6
  },
  {
    "id": 613770,
    "question": "Guangling District >> located in the administrative territorial entity",
    "answer": "Yangzhou",
    "paragraph_support_idx": 17
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 4
  }
]
```

题号：`4hop3__857_846_613770_7713`

问题：How long was the place where the Yongle Emperor greeted the person to whom the edict was addressed the capitol of the area where Guangling District was located?

```json
[
  {
    "id": 857,
    "question": "Who was the edict addressed to?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 11
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 3
  },
  {
    "id": 613770,
    "question": "Guangling District >> located in the administrative territorial entity",
    "answer": "Yangzhou",
    "paragraph_support_idx": 19
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 17
  }
]
```

题号：`4hop3__857_846_326964_7713`

问题：How long had the city where the Yongle emperor greeted the person to whom the edict was addressed been the capitol of Yaxing Coach's headquarters location?

```json
[
  {
    "id": 857,
    "question": "Who was the edict addressed to?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 7
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 13
  },
  {
    "id": 326964,
    "question": "Yaxing Coach >> headquarters location",
    "answer": "Yangzhou",
    "paragraph_support_idx": 5
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 4
  }
]
```

