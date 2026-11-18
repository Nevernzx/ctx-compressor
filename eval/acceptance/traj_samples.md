# 任务 01：真实 MuSiQue-Ans dev 轨迹验收

固定种子：20261117；3 个任务，每任务 4 道题，12 题互不重复；每个任务分别运行 reselect 与 append。
预算 **3000 token，占位，正式值由 §6 试验定**。本报告不执行 §6 可行性试验。
语料与 BM25 索引只来自 dev，共 21,100 个去重段落；每跳 top-20，未命中的证据替换末位，再带种子打乱。
片段 token 只计算 `标题 + 换行 + 正文`，与保留文本一致；不包含模板头及编号元数据。
因非法引用或证据映射跳过 0 题。当前回合 recall 的分母遵循 PLAN：只计已执行各跳中当前仍可获得的证据。

## 自动检查汇总

结果：**全部通过**。

| 检查 | 通过次数 |
|---|---:|
| resolved_queries | 58 |
| gold_in_candidates | 58 |
| budget_respected | 58 |
| legal_sft_targets | 58 |
| append_prefix_preserved | 29 |
| stable_local_ids | 58 |
| step_count_matches_hops | 3 |
| distinct_questions | 12 |

失败列表：`[]`。

## 任务 1

题号：`["2hop__49409_67668","2hop__543853_124498","2hop__75714_148696","2hop__39231_17335"]`

1. What character does the actor who played Barney on How I Met Your Mother play in Batman Under the Red Hood?
2. Who is the father of the Labo M performer?
3. What is the position of the first governor general of India?
4. When did the person who took over the territory and made it part of their empire die?

步数 8，等于所选 4 题跳数之和；自然命中 6 / 8。

### 模式：reselect

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：who played barney on how i met your mother
- 替换后的查询：who played barney on how i met your mother
- 证据段：编号 **9**，语料 pid 8193；标题：Neil Patrick Harris
- 证据正文前 200 字符："Neil Patrick Harris (born June 15, 1973) is an American actor, comedian, magician, and singer, known primarily for his comedy roles on television and his dramatic and musical stage roles. On televisio"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[9]`
- oracle / SFT 动作：`{"add":[9],"drop":[]}`
- 执行后保留：`[9]`
- 执行后 token：129 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：who does #1 play in batman under the red hood
- 替换后的查询：who does Neil Patrick Harris play in batman under the red hood
- 证据段：编号 **24**，语料 pid 6473；标题：Batman: Under the Red Hood
- 证据正文前 200 字符："Batman: Under the Red Hood is a 2010 American animated superhero direct - to - video film produced by Warner Bros. Animation and released by Warner Home Video. It is the eighth feature in the DC Unive"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,18,23,24,25,26,27,9,28,29,30,31,32,33,34,35,36,37]`
- 新返回编号（去除已保留）：`[20,21,22,18,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]`
- 已保留编号：`[9]`
- 当前可用金标准编号：`[9,24]`
- oracle / SFT 动作：`{"add":[24],"drop":[]}`
- 执行后保留：`[9,24]`
- 执行后 token：284 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 2 题，第 1 跳）

- 原始子问题：Labo M >> performer
- 替换后的查询：Labo M >> performer
- 证据段：编号 **52**，语料 pid 1613；标题：Labo M
- 证据正文前 200 字符："Labo M (2003) is the third studio album by French singer-songwriter Matthieu Chedid in his persona as -M-. It's an all-instrumental work which, considering that one of -M-'s trademarks had previously "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]`
- 新返回编号（去除已保留）：`[38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]`
- 已保留编号：`[9,24]`
- 当前可用金标准编号：`[9,24,52]`
- oracle / SFT 动作：`{"add":[52],"drop":[]}`
- 执行后保留：`[9,24,52]`
- 执行后 token：401 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 2 跳）

- 原始子问题：The father of #1 is whom?
- 替换后的查询：The father of Matthieu Chedid is whom?
- 证据段：编号 **66**，语料 pid 1586；标题：Matthieu Chedid
- 证据正文前 200 字符："Matthieu Chedid was born in Boulogne-Billancourt, Hauts-de-Seine, France, the son of French singer Louis Chedid, and the grandson of the Egyptian-born French writer and poet of Lebanese descent Andrée"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[58,59,60,61,52,62,63,64,65,66,67,68,47,69,70,71,72,73,74,75]`
- 新返回编号（去除已保留）：`[58,59,60,61,62,63,64,65,66,67,68,47,69,70,71,72,73,74,75]`
- 已保留编号：`[9,24,52]`
- 当前可用金标准编号：`[9,24,52,66]`
- oracle / SFT 动作：`{"add":[66],"drop":[]}`
- 执行后保留：`[9,24,52,66]`
- 执行后 token：485 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 3 题，第 1 跳）

- 原始子问题：who was the first govenor general of india
- 替换后的查询：who was the first govenor general of india
- 证据段：编号 **82**，语料 pid 12097；标题：Governor-General of India
- 证据正文前 200 字符："Governors - General served at the pleasure of the sovereign, though the practice was to have them serve five - year terms. Governors - General could have their commission rescinded; and if one was rem"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]`
- 新返回编号（去除已保留）：`[76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]`
- 已保留编号：`[9,24,52,66]`
- 当前可用金标准编号：`[9,24,52,66,82]`
- oracle / SFT 动作：`{"add":[82],"drop":[]}`
- 执行后保留：`[9,24,52,66,82]`
- 执行后 token：581 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 3 题，第 2 跳）

- 原始子问题：What is the position of #1 ?
- 替换后的查询：What is the position of Warren Hastings ?
- 证据段：编号 **107**，语料 pid 12069；标题：Warren Hastings
- 证据正文前 200 字符："Warren Hastings (6 December 1732 – 22 August 1818), an English statesman, was the first Governor of the Presidency of Fort William (Bengal), the head of the Supreme Council of Bengal, and thereby the "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[87,96,97,92,98,82,99,95,100,101,77,102,103,104,85,105,106,107,108,79]`
- 新返回编号（去除已保留）：`[87,96,97,92,98,99,95,100,101,77,102,103,104,85,105,106,107,108,79]`
- 已保留编号：`[9,24,52,66,82]`
- 当前可用金标准编号：`[9,24,52,66,82,107]`
- oracle / SFT 动作：`{"add":[107],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107]`
- 执行后 token：710 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 4 题，第 1 跳）

- 原始子问题：Who took over the territory and made it part of their empire?
- 替换后的查询：Who took over the territory and made it part of their empire?
- 证据段：编号 **118**，语料 pid 10574；标题：Tajikistan
- 证据正文前 200 字符："The earliest recorded history of the region dates back to about 500 BCE when much, if not all, of modern Tajikistan was part of the Achaemenid Empire. Some authors have also suggested that in the 7th "
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]`
- 新返回编号（去除已保留）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]`
- 已保留编号：`[9,24,52,66,82,107]`
- 当前可用金标准编号：`[9,24,52,66,82,107,118]`
- oracle / SFT 动作：`{"add":[118],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107,118]`
- 执行后 token：961 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 4 题，第 2 跳）

- 原始子问题：When did #1 die?
- 替换后的查询：When did Alexander the Great die?
- 证据段：编号 **132**，语料 pid 10580；标题：Hellenistic period
- 证据正文前 200 字符："The Hellenistic period covers the period of ancient Greek (Hellenic) history and Mediterranean history between the death of Alexander the Great in 323 BC and the emergence of the Roman Empire as signi"
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[129,130,67,131,132,133,134,135,136,137,138,139,140,141,142,118,143,144,145,146]`
- 新返回编号（去除已保留）：`[129,130,67,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146]`
- 已保留编号：`[9,24,52,66,82,107,118]`
- 当前可用金标准编号：`[9,24,52,66,82,107,118,132]`
- oracle / SFT 动作：`{"add":[132],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107,118,132]`
- 执行后 token：1218 / 3000；recall：1.000000
- 自动检查：通过

<details><summary>首步模型输入模板示例（已保留 / 新返回）</summary>

```text
任务问题：
1. What character does the actor who played Barney on How I Met Your Mother play in Batman Under the Red Hood?
2. Who is the father of the Labo M performer?
3. What is the position of the first governor general of India?
4. When did the person who took over the territory and made it part of their empire die?
模式：reselect
预算：3000 token
已用 token：0

已保留：
（空）

新返回：
[编号 0 | 248 token]
Barney Stinson
Barney Stinson is one of five main characters on How I Met Your Mother. He is a manipulative, oversexed businessman in his thirties who always wears a suit, likes women with ``daddy issues ''and is frequently willing to offer his (sometimes hypocritical) opinion. Throughout the earlier seasons, Barney is a huge womanizer, and has been described as a`` high - functioning sociopath'' by his best friend, Ted Mosby (Josh Radnor). Barney has a plethora of strategies and rules designed to meet women, sleep with them, and discard them. Through several seasons of the show, four (4) of the main characters are couples, as Ted began dating Robin Scherbatsky (Cobie Smulders) and Ted's roommate Marshall Eriksen (Jason Segel) becomes engaged and later married to Lily Aldrin (Alyson Hannigan). This leaves Barney the only single character, and, according to Harris, Barney is ``resentful ''that the other characters have paired up. Later on, in season 5, he dates Robin. They end up breaking up not long after, once they both realize they are making one another miserable.
[编号 1 | 106 token]
How I Met Your Mother
How I Met Your Mother (often abbreviated to HIMYM) is an American sitcom that originally aired on CBS from September 19, 2005, to March 31, 2014. The series follows the main character, Ted Mosby, and his group of friends in New York City's Manhattan. As a framing device, Ted, in the year 2030, recounts to his son and daughter the events that led him to meet their mother.
[编号 2 | 200 token]
How I Met Your Mother
The series concerns the adventures of Ted Mosby (played by Josh Radnor) narrating the story of how he met the mother of his children. The story goes into a flashback and starts in 2005 with a 27 - year - old Ted Mosby living in New York City and working as an architect; the narrative deals primarily with his best friends, including the long - lasting couple Marshall Eriksen (Jason Segel) and Lily Aldrin (Alyson Hannigan); the eccentric, womanizing - playboy Barney Stinson (Neil Patrick Harris); and news reporter Robin Scherbatsky (Cobie Smulders). The lives of all characters are entwined in each others. The series explores many storylines, including a ``will they or wo n't they ''relationship between Robin and each of the two single male friends, Marshall and Lily's relationship, and the ups and downs of the characters' careers.
[编号 3 | 111 token]
Karma (How I Met Your Mother)
``Karma ''How I Met Your Mother episode Episode no. Season 7 Episode 18 Directed by Pamela Fryman Written by Stephen Lloyd Production code 7ALH18 Original air date February 27, 2012 (2012 - 02 - 27) Recurring / Guest appearances Becki Newton (Quinn / Karma) Ellen D. Williams (Patrice) Episode chronology ← Previous`` No Pressure'' Next → ``The Broath ''
[编号 4 | 192 token]
How I Met Your Mother
The series concerns the adventures of Ted Mosby (played by Josh Radnor) narrating the story of how he met the mother of his children. The story goes into a flashback and starts in 2005 with a 27-year-old Ted Mosby living in New York City and working as an architect; the narrative deals primarily with his best friends, including the long-lasting couple Marshall Eriksen (Jason Segel) and Lily Aldrin (Alyson Hannigan), womanizing-playboy Barney Stinson (Neil Patrick Harris), and news reporter Robin Scherbatsky (Cobie Smulders). The lives of all characters are entwined in each others. The series explores many storylines, including a "will they or won't they" relationship between Robin and each of the two single male friends, Marshall and Lily's relationship, and the ups and downs of the characters' careers.
[编号 5 | 123 token]
The Mother (How I Met Your Mother)
The story of how Ted met The Mother is the framing device behind the series; many facts about her are revealed throughout the series, including the fact that Ted once unwittingly owned her umbrella before accidentally leaving it behind in her apartment. Ted and The Mother meet at the Farhampton train station following Barney Stinson and Robin Scherbatsky's wedding; this scene is shown in ``Last Forever '', the series finale. The Mother's death from an unspecified terminal illness in 2024, also revealed in the series finale, received a mixed reaction from fans.
[编号 6 | 91 token]
How I Met Your Mother
While the traditional love - story structure begins when the romantic leads first encounter each other, How I Met Your Mother does not introduce Ted's wife (Cristin Milioti) until the eighth - season finale, and only announces her name (Tracy McConnell) during the series finale. The show instead focuses on Ted's prior relationships and his dissatisfaction with those women, thus setting the stage for his eventual happiness with Tracy.
[编号 7 | 74 token]
Ted Mosby
Theodore Evelyn ``Ted ''Mosby is a fictional character and the main protagonist in the U.S. television sitcom How I Met Your Mother, portrayed by Josh Radnor. Ted also serves as the show's narrator from the future, voiced by Bob Saget, as he tells his children the`` long version'' of how he met their mother.
[编号 8 | 71 token]
Ted Mosby
Theodore Evelyn Mosby is a fictional character and the main protagonist in the U.S. television sitcom How I Met Your Mother, portrayed by Josh Radnor. Ted also serves as the show's narrator from the future, voiced by Bob Saget, as he tells his children the ``long version ''of how he met their mother.
[编号 9 | 129 token]
Neil Patrick Harris
Neil Patrick Harris (born June 15, 1973) is an American actor, comedian, magician, and singer, known primarily for his comedy roles on television and his dramatic and musical stage roles. On television, he is known for playing the title character on Doogie Howser, M.D. (1989 -- 1993), Barney Stinson on How I Met Your Mother (2005 -- 2014, for which he was nominated for four Emmy Awards), and Count Olaf on A Series of Unfortunate Events (2017 onward).
[编号 10 | 107 token]
How I Met Your Mother (season 1)
The first season of How I Met Your Mother, an American sitcom created by Carter Bays and Craig Thomas, premiered on CBS in the United States on September 19, 2005 and concluded on May 15, 2006. The season was directed by Pamela Fryman and produced by Bays & Thomas Productions and 20th Century Fox Television. It consists of 22 episodes, each running approximately 22 minutes in length.
[编号 11 | 101 token]
How I Met Your Mother
How I Met Your Mother (often abbreviated to HIMYM) is an American sitcom that originally aired on CBS from September 19, 2005 to March 31, 2014. The series follows the main character, Ted Mosby, and his group of friends in Manhattan. As a framing device, Ted, in the year 2030, recounts to his son and daughter the events that led him to meet their mother.
[编号 12 | 47 token]
How I Met Your Mother
Bob Saget as Future Ted Mosby (voice only, uncredited), Ted's future self, telling his children the story of how he met their mother, in the year 2030.
[编号 13 | 312 token]
List of How I Met Your Mother episodes
How I Met Your Mother is an American sitcom written and created by Carter Bays and Craig Thomas. It was first aired on September 19, 2005 on CBS with a thirty - minute pilot episode, and finished its planned nine - year run on March 31, 2014 after 208 episodes. Set in present - day Manhattan, the series follows the social and romantic lives of Ted Mosby (Josh Radnor) and his four best friends, Marshall Eriksen (Jason Segel), Robin Scherbatsky (Cobie Smulders), Lily Aldrin (Alyson Hannigan), and Barney Stinson (Neil Patrick Harris), as Ted seeks out his future wife (Cristin Milioti). The show tells this story through the framing device of ``future ''Ted (Bob Saget) as an unreliable narrator who is recounting to his son and daughter the events that led him to their mother. In Seasons 1 -- 8, episodes are typically set to a similar timeline as its real - world airdate, with the season finale generally taking place in the spring and the next season premiere in the fall, usually with a quick recap of the events that took place during the summer. The final season (season 9) deviates from this format by taking place immediately after the end of the previous season and encompassing only the weekend of Barney's and Robin's wedding. The series finale covers several years that follow.
[编号 14 | 151 token]
List of How I Met Your Mother characters
Played by Britney Spears; appears in ``Ten Sessions ''and`` Everything Must Go''. Abby is Stella's receptionist. Very energetic and scatterbrained, she developed an obsessive crush on Ted while he was pursuing Stella. Barney ended up sleeping with her. It was announced Spears would appear on the show in March 2008; most critics complimented her acting in ``Ten Sessions '', with Segel noting that she improvised a few`` really good'' lines. However, her scenes in ``Everything Must Go ''were described as`` rushed and awkward'', and her character as ``silly ''. An article in Bustle ranked her top in a list of guest stars on How I Met Your Mother.
[编号 15 | 115 token]
The Mother (How I Met Your Mother)
Tracy McConnell (colloquial: ``The Mother '') is the title character from the CBS television sitcom How I Met Your Mother. The show, narrated by Future Ted, tells the story of how Ted Mosby met The Mother. Tracy McConnell appears in 8 episodes from`` Lucky Penny'' to ``The Time Travelers ''as an unseen character; she was first seen fully in`` Something New'' and was promoted to a main character in season 9. The Mother is played by Cristin Milioti.
[编号 16 | 113 token]
The Mother (How I Met Your Mother)
Tracy McConnell, better known as ``The Mother '', is the title character from the CBS television sitcom How I Met Your Mother. The show, narrated by Future Ted, tells the story of how Ted Mosby met The Mother. Tracy McConnell appears in 8 episodes from`` Lucky Penny'' to ``The Time Travelers ''as an unseen character; she was first seen fully in`` Something New'' and was promoted to a main character in season 9. The Mother is played by Cristin Milioti.
[编号 17 | 124 token]
The Mother (How I Met Your Mother)
Tracy McConnell (colloquial: ``The Mother '') is the title character from the CBS television sitcom How I Met Your Mother. The show, narrated by Future Ted (Bob Saget), tells the story of how Ted Mosby (Josh Radnor) met The Mother. Tracy McConnell appears in eight episodes, from`` Lucky Penny'' to ``The Time Travelers '', as an unseen character; she was first seen fully in`` Something New'' and was promoted to a main character in season 9. The Mother is played by Cristin Milioti.
[编号 18 | 197 token]
How I Met Your Mother
The series concerns the adventures of Ted Mosby (played by Josh Radnor) narrating the story of how he met the mother of his children. The story goes into a flashback and starts in 2005 with a 27 - year - old Ted Mosby living in New York City and working as an architect; the narrative deals primarily with his best friends, including the long - lasting couple Marshall Eriksen (Jason Segel) and Lily Aldrin (Alyson Hannigan), womanizing - playboy Barney Stinson (Neil Patrick Harris), and news reporter Robin Scherbatsky (Cobie Smulders). The lives of all characters are entwined in each others. The series explores many storylines, including a ``will they or wo n't they ''relationship between Robin and each of the two single male friends, Marshall and Lily's relationship, and the ups and downs of the characters' careers.
[编号 19 | 76 token]
Bad News (How I Met Your Mother)
``Bad News ''is the 13th episode of the sixth season of the CBS sitcom How I Met Your Mother and their 125th episode overall. It aired January 3, 2011. TV Guide listed`` Bad News'' in its list of 2011's Top TV Episodes.
```

</details>

### 模式：append

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：who played barney on how i met your mother
- 替换后的查询：who played barney on how i met your mother
- 证据段：编号 **9**，语料 pid 8193；标题：Neil Patrick Harris
- 证据正文前 200 字符："Neil Patrick Harris (born June 15, 1973) is an American actor, comedian, magician, and singer, known primarily for his comedy roles on television and his dramatic and musical stage roles. On televisio"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[9]`
- oracle / SFT 动作：`{"add":[9],"drop":[]}`
- 执行后保留：`[9]`
- 执行后 token：129 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：who does #1 play in batman under the red hood
- 替换后的查询：who does Neil Patrick Harris play in batman under the red hood
- 证据段：编号 **24**，语料 pid 6473；标题：Batman: Under the Red Hood
- 证据正文前 200 字符："Batman: Under the Red Hood is a 2010 American animated superhero direct - to - video film produced by Warner Bros. Animation and released by Warner Home Video. It is the eighth feature in the DC Unive"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,18,23,24,25,26,27,9,28,29,30,31,32,33,34,35,36,37]`
- 新返回编号（去除已保留）：`[20,21,22,18,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]`
- 已保留编号：`[9]`
- 当前可用金标准编号：`[9,24]`
- oracle / SFT 动作：`{"add":[24],"drop":[]}`
- 执行后保留：`[9,24]`
- 执行后 token：284 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 2 题，第 1 跳）

- 原始子问题：Labo M >> performer
- 替换后的查询：Labo M >> performer
- 证据段：编号 **52**，语料 pid 1613；标题：Labo M
- 证据正文前 200 字符："Labo M (2003) is the third studio album by French singer-songwriter Matthieu Chedid in his persona as -M-. It's an all-instrumental work which, considering that one of -M-'s trademarks had previously "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]`
- 新返回编号（去除已保留）：`[38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]`
- 已保留编号：`[9,24]`
- 当前可用金标准编号：`[9,24,52]`
- oracle / SFT 动作：`{"add":[52],"drop":[]}`
- 执行后保留：`[9,24,52]`
- 执行后 token：401 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 2 跳）

- 原始子问题：The father of #1 is whom?
- 替换后的查询：The father of Matthieu Chedid is whom?
- 证据段：编号 **66**，语料 pid 1586；标题：Matthieu Chedid
- 证据正文前 200 字符："Matthieu Chedid was born in Boulogne-Billancourt, Hauts-de-Seine, France, the son of French singer Louis Chedid, and the grandson of the Egyptian-born French writer and poet of Lebanese descent Andrée"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[58,59,60,61,52,62,63,64,65,66,67,68,47,69,70,71,72,73,74,75]`
- 新返回编号（去除已保留）：`[58,59,60,61,62,63,64,65,66,67,68,47,69,70,71,72,73,74,75]`
- 已保留编号：`[9,24,52]`
- 当前可用金标准编号：`[9,24,52,66]`
- oracle / SFT 动作：`{"add":[66],"drop":[]}`
- 执行后保留：`[9,24,52,66]`
- 执行后 token：485 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 3 题，第 1 跳）

- 原始子问题：who was the first govenor general of india
- 替换后的查询：who was the first govenor general of india
- 证据段：编号 **82**，语料 pid 12097；标题：Governor-General of India
- 证据正文前 200 字符："Governors - General served at the pleasure of the sovereign, though the practice was to have them serve five - year terms. Governors - General could have their commission rescinded; and if one was rem"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]`
- 新返回编号（去除已保留）：`[76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]`
- 已保留编号：`[9,24,52,66]`
- 当前可用金标准编号：`[9,24,52,66,82]`
- oracle / SFT 动作：`{"add":[82],"drop":[]}`
- 执行后保留：`[9,24,52,66,82]`
- 执行后 token：581 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 3 题，第 2 跳）

- 原始子问题：What is the position of #1 ?
- 替换后的查询：What is the position of Warren Hastings ?
- 证据段：编号 **107**，语料 pid 12069；标题：Warren Hastings
- 证据正文前 200 字符："Warren Hastings (6 December 1732 – 22 August 1818), an English statesman, was the first Governor of the Presidency of Fort William (Bengal), the head of the Supreme Council of Bengal, and thereby the "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[87,96,97,92,98,82,99,95,100,101,77,102,103,104,85,105,106,107,108,79]`
- 新返回编号（去除已保留）：`[87,96,97,92,98,99,95,100,101,77,102,103,104,85,105,106,107,108,79]`
- 已保留编号：`[9,24,52,66,82]`
- 当前可用金标准编号：`[9,24,52,66,82,107]`
- oracle / SFT 动作：`{"add":[107],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107]`
- 执行后 token：710 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 4 题，第 1 跳）

- 原始子问题：Who took over the territory and made it part of their empire?
- 替换后的查询：Who took over the territory and made it part of their empire?
- 证据段：编号 **118**，语料 pid 10574；标题：Tajikistan
- 证据正文前 200 字符："The earliest recorded history of the region dates back to about 500 BCE when much, if not all, of modern Tajikistan was part of the Achaemenid Empire. Some authors have also suggested that in the 7th "
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]`
- 新返回编号（去除已保留）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]`
- 已保留编号：`[9,24,52,66,82,107]`
- 当前可用金标准编号：`[9,24,52,66,82,107,118]`
- oracle / SFT 动作：`{"add":[118],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107,118]`
- 执行后 token：961 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 4 题，第 2 跳）

- 原始子问题：When did #1 die?
- 替换后的查询：When did Alexander the Great die?
- 证据段：编号 **132**，语料 pid 10580；标题：Hellenistic period
- 证据正文前 200 字符："The Hellenistic period covers the period of ancient Greek (Hellenic) history and Mediterranean history between the death of Alexander the Great in 323 BC and the emergence of the Roman Empire as signi"
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[129,130,67,131,132,133,134,135,136,137,138,139,140,141,142,118,143,144,145,146]`
- 新返回编号（去除已保留）：`[129,130,67,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146]`
- 已保留编号：`[9,24,52,66,82,107,118]`
- 当前可用金标准编号：`[9,24,52,66,82,107,118,132]`
- oracle / SFT 动作：`{"add":[132],"drop":[]}`
- 执行后保留：`[9,24,52,66,82,107,118,132]`
- 执行后 token：1218 / 3000；recall：1.000000
- 自动检查：通过

## 任务 2

题号：`["2hop__412633_15781","3hop1__92991_78276_68042","3hop1__66872_709625_84283","2hop__606439_127399"]`

1. Who, along with the author of An Essay on the History of Civil Society, criticized the idea of social contract theorists?
2. When did the majority party in the body which determines the rules of the US House and US Senate gain control of the House?
3. Who owns the record label of the Till I Can Gain Control Again songwriter?
4. What year did the group that performed From Them, Through Us, to You form?

步数 10，等于所选 4 题跳数之和；自然命中 10 / 10。

### 模式：reselect

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：An Essay on the History of Civil Society >> author
- 替换后的查询：An Essay on the History of Civil Society >> author
- 证据段：编号 **16**，语料 pid 11205；标题：An Essay on the History of Civil Society
- 证据正文前 200 字符："An Essay on the History of Civil Society is a book by the Scottish Enlightenment philosopher Adam Ferguson, first published in 1767. The \"Essay\" established Ferguson's reputation in Britain and throug"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[16]`
- oracle / SFT 动作：`{"add":[16],"drop":[]}`
- 执行后保留：`[16]`
- 执行后 token：51 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：Who, along with #1 , criticized the idea of social contract theorists?
- 替换后的查询：Who, along with Adam Ferguson , criticized the idea of social contract theorists?
- 证据段：编号 **1**，语料 pid 11195；标题：Age of Enlightenment
- 证据正文前 200 字符："Though much of Enlightenment political thought was dominated by social contract theorists, both David Hume and Adam Ferguson criticized this camp. Hume's essay Of the Original Contract argues that gov"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,23,24,25,26,27,28,29,30,31,12,32,1,16,33,34,35,36]`
- 新返回编号（去除已保留）：`[20,21,22,23,24,25,26,27,28,29,30,31,12,32,1,33,34,35,36]`
- 已保留编号：`[16]`
- 当前可用金标准编号：`[16,1]`
- oracle / SFT 动作：`{"add":[1],"drop":[]}`
- 执行后保留：`[16,1]`
- 执行后 token：241 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 2 题，第 1 跳）

- 原始子问题：who determines the rules of the us house and us senate
- 替换后的查询：who determines the rules of the us house and us senate
- 证据段：编号 **44**，语料 pid 15198；标题：Standing Rules of the United States Senate
- 证据正文前 200 字符："The Standing Rules of the Senate are the parliamentary procedures adopted by the United States Senate that govern its procedure. The Senate's power to establish rules derives from Article One, Section"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]`
- 新返回编号（去除已保留）：`[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]`
- 已保留编号：`[16,1]`
- 当前可用金标准编号：`[16,1,44]`
- oracle / SFT 动作：`{"add":[44],"drop":[]}`
- 执行后保留：`[16,1,44]`
- 执行后 token：303 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 2 跳）

- 原始子问题：which party is the majority party in the #1
- 替换后的查询：which party is the majority party in the The Senate
- 证据段：编号 **68**，语料 pid 15143；标题：Party leaders of the United States Senate
- 证据正文前 200 字符："The Senate is currently composed of 51 Republicans, 47 Democrats, and 2 independents, both of whom caucus with the Democrats."
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[57,58,59,60,46,51,61,62,45,63,64,65,66,67,68,69,70,71,72,73]`
- 新返回编号（去除已保留）：`[57,58,59,60,46,51,61,62,45,63,64,65,66,67,68,69,70,71,72,73]`
- 已保留编号：`[16,1,44]`
- 当前可用金标准编号：`[16,1,44,68]`
- oracle / SFT 动作：`{"add":[68],"drop":[]}`
- 执行后保留：`[16,1,44,68]`
- 执行后 token：340 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 2 题，第 3 跳）

- 原始子问题：when did the #2 gain control of the house
- 替换后的查询：when did the Republicans gain control of the house
- 证据段：编号 **87**，语料 pid 15150；标题：2010 United States House of Representatives elections
- 证据正文前 200 字符："Republicans regained control of the chamber they had lost in the 2006 midterm elections, picking up a net total of 63 seats and erasing the gains Democrats made in 2006 and 2008. Although the sitting "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[74,75,76,77,78,79,80,81,82,83,73,84,85,61,86,87,88,89,90,71]`
- 新返回编号（去除已保留）：`[74,75,76,77,78,79,80,81,82,83,73,84,85,61,86,87,88,89,90,71]`
- 已保留编号：`[16,1,44,68]`
- 当前可用金标准编号：`[16,1,44,68,87]`
- oracle / SFT 动作：`{"add":[87],"drop":[]}`
- 执行后保留：`[16,1,44,68,87]`
- 执行后 token：642 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 3 题，第 1 跳）

- 原始子问题：who wrote till i can gain control again
- 替换后的查询：who wrote till i can gain control again
- 证据段：编号 **78**，语料 pid 14536；标题：'Til I Gain Control Again
- 证据正文前 200 字符："``'Til I Gain Control Again ''is a country song written by Rodney Crowell and originally recorded by Emmylou Harris in 1975. The song was included on her 1975 studio album Elite Hotel and was performe"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[78,91,92,93,94,95,96,97,98,99,100,101,102,86,103,104,105,106,107,108]`
- 新返回编号（去除已保留）：`[78,91,92,93,94,95,96,97,98,99,100,101,102,86,103,104,105,106,107,108]`
- 已保留编号：`[16,1,44,68,87]`
- 当前可用金标准编号：`[16,1,44,68,87,78]`
- oracle / SFT 动作：`{"add":[78],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78]`
- 执行后 token：867 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 3 题，第 2 跳）

- 原始子问题：#1 >> record label
- 替换后的查询：Rodney Crowell >> record label
- 证据段：编号 **112**，语料 pid 14540；标题：Ain't Living Long Like This
- 证据正文前 200 字符："Ain't Living Long Like This is the debut studio album by American country music singer-songwriter Rodney Crowell, released in 1978 by Warner Bros. Records. It failed to enter the Top Country Albums ch"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,78,123,124,125,126,127]`
- 新返回编号（去除已保留）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]`
- 已保留编号：`[16,1,44,68,87,78]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112]`
- oracle / SFT 动作：`{"add":[112],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112]`
- 执行后 token：1086 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 3 题，第 3 跳）

- 原始子问题：who is the owner of #2
- 替换后的查询：who is the owner of Warner Bros. Records
- 证据段：编号 **142**，语料 pid 14524；标题：Warner Records
- 证据正文前 200 字符："Warner Bros. Records Parent company Warner Music Group Founded March 19, 1958; 60 years ago (1958 - 03 - 19) Founder James Conkling Distributor (s) Self - distributed (In the US) WEA International (Ou"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147]`
- 新返回编号（去除已保留）：`[128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147]`
- 已保留编号：`[16,1,44,68,87,78,112]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142]`
- oracle / SFT 动作：`{"add":[142],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142]`
- 执行后 token：1189 / 3000；recall：1.000000
- 自动检查：通过

#### 第 9 步（第 4 题，第 1 跳）

- 原始子问题：From Them, Through Us, to You >> performer
- 替换后的查询：From Them, Through Us, to You >> performer
- 证据段：编号 **154**，语料 pid 11272；标题：From Them, Through Us, to You
- 证据正文前 200 字符："From Them, Through Us, to You is the debut album by Chicago-based rock band Madina Lake. The album was released in the United States on March 27, 2007 via Roadrunner Records; it was released a day ear"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[148,149,150,151,152,102,153,154,155,156,157,158,159,160,161,162,163,164,165,166]`
- 新返回编号（去除已保留）：`[148,149,150,151,152,102,153,154,155,156,157,158,159,160,161,162,163,164,165,166]`
- 已保留编号：`[16,1,44,68,87,78,112,142]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142,154]`
- oracle / SFT 动作：`{"add":[154],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142,154]`
- 执行后 token：1253 / 3000；recall：1.000000
- 自动检查：通过

#### 第 10 步（第 4 题，第 2 跳）

- 原始子问题：Which year witnessed the formation of #1 ?
- 替换后的查询：Which year witnessed the formation of Madina Lake ?
- 证据段：编号 **165**，语料 pid 11249；标题：Madina Lake
- 证据正文前 200 字符："Madina Lake is an American alternative rock band formed in Chicago in 2005. Madina Lake released their debut album \"From Them, Through Us, to You\" through Roadrunner Records on March 27, 2007. Madina "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[167,168,169,170,171,172,173,154,174,175,176,177,178,179,180,181,182,183,184,165]`
- 新返回编号（去除已保留）：`[167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,165]`
- 已保留编号：`[16,1,44,68,87,78,112,142,154]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142,154,165]`
- oracle / SFT 动作：`{"add":[165],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142,154,165]`
- 执行后 token：1353 / 3000；recall：1.000000
- 自动检查：通过

<details><summary>首步模型输入模板示例（已保留 / 新返回）</summary>

```text
任务问题：
1. Who, along with the author of An Essay on the History of Civil Society, criticized the idea of social contract theorists?
2. When did the majority party in the body which determines the rules of the US House and US Senate gain control of the House?
3. Who owns the record label of the Till I Can Gain Control Again songwriter?
4. What year did the group that performed From Them, Through Us, to You form?
模式：reselect
预算：3000 token
已用 token：0

已保留：
（空）

新返回：
[编号 0 | 56 token]
Trenton Lee Stewart
Trenton Lee Stewart (born 1970) is an American author best known for the Mysterious Benedict Society series. Stewart is a graduate of Hendrix College and the Iowa Writers' Workshop. He lives in Little Rock, Arkansas.
[编号 1 | 190 token]
Age of Enlightenment
Though much of Enlightenment political thought was dominated by social contract theorists, both David Hume and Adam Ferguson criticized this camp. Hume's essay Of the Original Contract argues that governments derived from consent are rarely seen, and civil government is grounded in a ruler's habitual authority and force. It is precisely because of the ruler's authority over-and-against the subject, that the subject tacitly consents; Hume says that the subjects would "never imagine that their consent made him sovereign", rather the authority did so. Similarly, Ferguson did not believe citizens built the state, rather polities grew out of social development. In his 1767 An Essay on the History of Civil Society, Ferguson uses the four stages of progress, a theory that was very popular in Scotland at the time, to explain how humans advance from a hunting and gathering society to a commercial and civil society without "signing" a social contract.
[编号 2 | 62 token]
Robert J. Richards
Robert J. Richards (born 1942) is an author and the Morris Fishbein Distinguished Service Professor of the History of Science and Medicine at the University of Chicago. He has written or edited seven books about the history of science as well as dozens of articles.
[编号 3 | 112 token]
The New England Quarterly
The New England Quarterly is a peer-reviewed academic journal consisting of articles on New England's cultural, literary, political, and social history. The journal contains essays, interpretations of traditional texts, essay reviews and book reviews. "The New England Quarterly" was established in 1928 and is published by MIT Press for The New England Quarterly Inc., a nonprofit sponsored by the University of Massachusetts Boston and the Colonial Society of Massachusetts, and supported by the Massachusetts Cultural Council. MIT Press began publishing the journal in 2007.
[编号 4 | 42 token]
Gideon's Spies
Gideon's Spies: The Secret History of the Mossad is a 1999 book by Welsh author Gordon Thomas on the Mossad, the Israeli intelligence service.
[编号 5 | 88 token]
Henry Brainerd McClellan
Henry Brainerd McClellan (October 17, 1840–October 1, 1904) was an officer and adjutant general in the Confederate States Army during the American Civil War (Civil War), a teacher and author. He was a professor at Sayre Female Institute in Lexington, Kentucky for 35 years after the war.
[编号 6 | 187 token]
Pendennis
The History of Pendennis: His Fortunes and Misfortunes, His Friends and His Greatest Enemy (1848–50) is a novel by the English author William Makepeace Thackeray. It is set in 19th-century England, particularly in London. The main hero is a young English gentleman Arthur Pendennis, who is born in the country and sets out for London to seek his place in life and society. The novel took two years for Thackeray to write and, in line with other Thackeray's works, most notably "Vanity Fair", it offers an insightful and satiric picture of human character and aristocratic society. The characters include the snobbish social hanger-on Major Pendennis and the tipsy Captain Costigan. Miss Amory and Sir Francis Clavering are somewhat reminiscent of Becky Sharp and Sir Pitt from "Vanity Fair".
[编号 7 | 117 token]
PEN/Diamonstein-Spielvogel Award for the Art of the Essay
The PEN/Diamonstein-Spielvogel Award for the Art of the Essay is awarded by the PEN American Center to an author for a book of original collected essays. The award was founded by PEN Member and author Barbaralee Diamonstein and Carl Spielvogel, former "New York Times" columnist, "to preserve the dignity and esteem that the essay form imparts to literature." The winner receives a cash award of $10,000.
[编号 8 | 44 token]
Minnesota History Center
The Minnesota History Center is a museum and library that serves as the headquarters of the Minnesota Historical Society. It is near downtown Saint Paul, Minnesota, and is considered one of Minnesota's finest public buildings.
[编号 9 | 172 token]
Civil disobedience
Following the Peterloo massacre of 1819, poet Percy Shelley wrote the political poem The Mask of Anarchy later that year, that begins with the images of what he thought to be the unjust forms of authority of his time—and then imagines the stirrings of a new form of social action. It is perhaps the first modern[vague] statement of the principle of nonviolent protest. A version was taken up by the author Henry David Thoreau in his essay Civil Disobedience, and later by Gandhi in his doctrine of Satyagraha. Gandhi's Satyagraha was partially influenced and inspired by Shelley's nonviolence in protest and political action. In particular, it is known that Gandhi would often quote Shelley's Masque of Anarchy to vast audiences during the campaign for a free India.
[编号 10 | 57 token]
Cas Mudde
Cas Mudde (born 3 June 1967) is a Dutch political scientist who focuses on political extremism and populism in Europe and the United States. His research includes the areas of political parties, extremism, democracy, civil society and European politics.
[编号 11 | 90 token]
This Misery of Boots
This Misery of Boots is a 1907 political tract by H. G. Wells advocating socialism. Published by the Fabian Society, "This Misery of Boots" is the expansion of a 1905 essay with the same name. Its five chapters condemn private property in land and means of production and calls for their expropriation by the state "not for profit, but for service."
[编号 12 | 66 token]
Balance of power (international relations)
The principle involved in preserving the balance of power as a conscious goal of foreign policy, as David Hume pointed out in his Essay on the Balance of Power, is as old as history, and was used by Greeks such as Thucydides both as political theorists and as practical statesmen.
[编号 13 | 62 token]
F. A. Forbes
F. A. Forbes (16 March 1869 – 1936) was the "nom de plume" of Mother Frances Alice Monica Forbes, RSCJ, a member of the Society of the Sacred Heart from Scotland and a religious author.
[编号 14 | 123 token]
John Dear
John Dear (born August 15, 1959) is an American Catholic priest, Christian pacifist, vegetarianism advocate, author and lecturer, and a former member of the Society of Jesus. He has been arrested over 75 times in acts of nonviolent civil disobedience against war, injustice and nuclear weapons as part of his "consistent ethic of nonviolence". Dear has been nominated several times for the Nobel Peace Prize, in January 2008 by Archbishop Desmond Tutu and most recently with Leo Rebello for the 2015 award.
[编号 15 | 137 token]
Sébastien Fath
Sébastien Fath (born 1968 in Strasbourg) is a French professional historian and a Ph.D at the Sorbonne University. Also trained in Sociology, he is the main French specialist in the study of Evangelical Protestantism. Author of sixteen books, he is a permanent researcher at the National Center for Scientific Research (CNRS). He is a full member of the GSRL (Groupe Sociétés Religions Laïcités), a research team working on religion and secularism. As a social scientist and a citizen, cross-cultural reflexion on Civil Society, Politics and Religion is his focus.
[编号 16 | 51 token]
An Essay on the History of Civil Society
An Essay on the History of Civil Society is a book by the Scottish Enlightenment philosopher Adam Ferguson, first published in 1767. The "Essay" established Ferguson's reputation in Britain and throughout Europe.
[编号 17 | 92 token]
Pennsylvania German Society
The Pennsylvania German Society is a non-profit, educational organization dedicated to studying the Pennsylvania German people and their 330-year history in the United States and Canada. The Society works to preserve and promote the history, culture, religion, and dialect of the Pennsylvania Germans (also commonly known as the "Pennsylvania Dutch"). It was founded in 1891 and became a founding member of the Pennsylvania Federation of Historical Societies.
[编号 18 | 202 token]
Edmund Burke
On 25 February 1757, Burke signed a contract with Robert Dodsley to write a "history of England from the time of Julius Caesar to the end of the reign of Queen Anne", its length being eighty quarto sheets (640 pages), nearly 400,000 words. It was to be submitted for publication by Christmas 1758. Burke completed the work to the year 1216 and stopped; it was not published until after Burke's death, being included in an 1812 collection of his works, entitled An Essay Towards an Abridgement of the English History. G. M. Young did not value Burke's history and claimed that it was "demonstrably a translation from the French". Lord Acton, on commenting on the story that Burke stopped his history because David Hume published his, said "it is ever to be regretted that the reverse did not occur".
[编号 19 | 58 token]
Barbara Newman
Barbara Jane Newman is an American medievalist, literary critic, religious historian, and author. She is Professor of English and Religion, and John Evans Professor of Latin, at Northwestern University. Newman was elected in 2017 to the American Philosophical Society.
```

</details>

### 模式：append

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：An Essay on the History of Civil Society >> author
- 替换后的查询：An Essay on the History of Civil Society >> author
- 证据段：编号 **16**，语料 pid 11205；标题：An Essay on the History of Civil Society
- 证据正文前 200 字符："An Essay on the History of Civil Society is a book by the Scottish Enlightenment philosopher Adam Ferguson, first published in 1767. The \"Essay\" established Ferguson's reputation in Britain and throug"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[16]`
- oracle / SFT 动作：`{"add":[16],"drop":[]}`
- 执行后保留：`[16]`
- 执行后 token：51 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：Who, along with #1 , criticized the idea of social contract theorists?
- 替换后的查询：Who, along with Adam Ferguson , criticized the idea of social contract theorists?
- 证据段：编号 **1**，语料 pid 11195；标题：Age of Enlightenment
- 证据正文前 200 字符："Though much of Enlightenment political thought was dominated by social contract theorists, both David Hume and Adam Ferguson criticized this camp. Hume's essay Of the Original Contract argues that gov"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,23,24,25,26,27,28,29,30,31,12,32,1,16,33,34,35,36]`
- 新返回编号（去除已保留）：`[20,21,22,23,24,25,26,27,28,29,30,31,12,32,1,33,34,35,36]`
- 已保留编号：`[16]`
- 当前可用金标准编号：`[16,1]`
- oracle / SFT 动作：`{"add":[1],"drop":[]}`
- 执行后保留：`[16,1]`
- 执行后 token：241 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 2 题，第 1 跳）

- 原始子问题：who determines the rules of the us house and us senate
- 替换后的查询：who determines the rules of the us house and us senate
- 证据段：编号 **44**，语料 pid 15198；标题：Standing Rules of the United States Senate
- 证据正文前 200 字符："The Standing Rules of the Senate are the parliamentary procedures adopted by the United States Senate that govern its procedure. The Senate's power to establish rules derives from Article One, Section"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]`
- 新返回编号（去除已保留）：`[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]`
- 已保留编号：`[16,1]`
- 当前可用金标准编号：`[16,1,44]`
- oracle / SFT 动作：`{"add":[44],"drop":[]}`
- 执行后保留：`[16,1,44]`
- 执行后 token：303 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 2 跳）

- 原始子问题：which party is the majority party in the #1
- 替换后的查询：which party is the majority party in the The Senate
- 证据段：编号 **68**，语料 pid 15143；标题：Party leaders of the United States Senate
- 证据正文前 200 字符："The Senate is currently composed of 51 Republicans, 47 Democrats, and 2 independents, both of whom caucus with the Democrats."
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[57,58,59,60,46,51,61,62,45,63,64,65,66,67,68,69,70,71,72,73]`
- 新返回编号（去除已保留）：`[57,58,59,60,46,51,61,62,45,63,64,65,66,67,68,69,70,71,72,73]`
- 已保留编号：`[16,1,44]`
- 当前可用金标准编号：`[16,1,44,68]`
- oracle / SFT 动作：`{"add":[68],"drop":[]}`
- 执行后保留：`[16,1,44,68]`
- 执行后 token：340 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 2 题，第 3 跳）

- 原始子问题：when did the #2 gain control of the house
- 替换后的查询：when did the Republicans gain control of the house
- 证据段：编号 **87**，语料 pid 15150；标题：2010 United States House of Representatives elections
- 证据正文前 200 字符："Republicans regained control of the chamber they had lost in the 2006 midterm elections, picking up a net total of 63 seats and erasing the gains Democrats made in 2006 and 2008. Although the sitting "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[74,75,76,77,78,79,80,81,82,83,73,84,85,61,86,87,88,89,90,71]`
- 新返回编号（去除已保留）：`[74,75,76,77,78,79,80,81,82,83,73,84,85,61,86,87,88,89,90,71]`
- 已保留编号：`[16,1,44,68]`
- 当前可用金标准编号：`[16,1,44,68,87]`
- oracle / SFT 动作：`{"add":[87],"drop":[]}`
- 执行后保留：`[16,1,44,68,87]`
- 执行后 token：642 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 3 题，第 1 跳）

- 原始子问题：who wrote till i can gain control again
- 替换后的查询：who wrote till i can gain control again
- 证据段：编号 **78**，语料 pid 14536；标题：'Til I Gain Control Again
- 证据正文前 200 字符："``'Til I Gain Control Again ''is a country song written by Rodney Crowell and originally recorded by Emmylou Harris in 1975. The song was included on her 1975 studio album Elite Hotel and was performe"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[78,91,92,93,94,95,96,97,98,99,100,101,102,86,103,104,105,106,107,108]`
- 新返回编号（去除已保留）：`[78,91,92,93,94,95,96,97,98,99,100,101,102,86,103,104,105,106,107,108]`
- 已保留编号：`[16,1,44,68,87]`
- 当前可用金标准编号：`[16,1,44,68,87,78]`
- oracle / SFT 动作：`{"add":[78],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78]`
- 执行后 token：867 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 3 题，第 2 跳）

- 原始子问题：#1 >> record label
- 替换后的查询：Rodney Crowell >> record label
- 证据段：编号 **112**，语料 pid 14540；标题：Ain't Living Long Like This
- 证据正文前 200 字符："Ain't Living Long Like This is the debut studio album by American country music singer-songwriter Rodney Crowell, released in 1978 by Warner Bros. Records. It failed to enter the Top Country Albums ch"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,78,123,124,125,126,127]`
- 新返回编号（去除已保留）：`[109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]`
- 已保留编号：`[16,1,44,68,87,78]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112]`
- oracle / SFT 动作：`{"add":[112],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112]`
- 执行后 token：1086 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 3 题，第 3 跳）

- 原始子问题：who is the owner of #2
- 替换后的查询：who is the owner of Warner Bros. Records
- 证据段：编号 **142**，语料 pid 14524；标题：Warner Records
- 证据正文前 200 字符："Warner Bros. Records Parent company Warner Music Group Founded March 19, 1958; 60 years ago (1958 - 03 - 19) Founder James Conkling Distributor (s) Self - distributed (In the US) WEA International (Ou"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147]`
- 新返回编号（去除已保留）：`[128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147]`
- 已保留编号：`[16,1,44,68,87,78,112]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142]`
- oracle / SFT 动作：`{"add":[142],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142]`
- 执行后 token：1189 / 3000；recall：1.000000
- 自动检查：通过

#### 第 9 步（第 4 题，第 1 跳）

- 原始子问题：From Them, Through Us, to You >> performer
- 替换后的查询：From Them, Through Us, to You >> performer
- 证据段：编号 **154**，语料 pid 11272；标题：From Them, Through Us, to You
- 证据正文前 200 字符："From Them, Through Us, to You is the debut album by Chicago-based rock band Madina Lake. The album was released in the United States on March 27, 2007 via Roadrunner Records; it was released a day ear"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[148,149,150,151,152,102,153,154,155,156,157,158,159,160,161,162,163,164,165,166]`
- 新返回编号（去除已保留）：`[148,149,150,151,152,102,153,154,155,156,157,158,159,160,161,162,163,164,165,166]`
- 已保留编号：`[16,1,44,68,87,78,112,142]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142,154]`
- oracle / SFT 动作：`{"add":[154],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142,154]`
- 执行后 token：1253 / 3000；recall：1.000000
- 自动检查：通过

#### 第 10 步（第 4 题，第 2 跳）

- 原始子问题：Which year witnessed the formation of #1 ?
- 替换后的查询：Which year witnessed the formation of Madina Lake ?
- 证据段：编号 **165**，语料 pid 11249；标题：Madina Lake
- 证据正文前 200 字符："Madina Lake is an American alternative rock band formed in Chicago in 2005. Madina Lake released their debut album \"From Them, Through Us, to You\" through Roadrunner Records on March 27, 2007. Madina "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[167,168,169,170,171,172,173,154,174,175,176,177,178,179,180,181,182,183,184,165]`
- 新返回编号（去除已保留）：`[167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,165]`
- 已保留编号：`[16,1,44,68,87,78,112,142,154]`
- 当前可用金标准编号：`[16,1,44,68,87,78,112,142,154,165]`
- oracle / SFT 动作：`{"add":[165],"drop":[]}`
- 执行后保留：`[16,1,44,68,87,78,112,142,154,165]`
- 执行后 token：1353 / 3000；recall：1.000000
- 自动检查：通过

## 任务 3

题号：`["3hop1__36497_73181_68042","4hop1__436202_765799_282674_759393","2hop__370564_71701","2hop__512773_346751"]`

1. When did the party with a majority in the body the President calls on for support of his appointments to the USAF gain control of the House?
2. What is the seat of the county sharing a border with the county in which J. P. Hayes was born?
3. What year did the japanese get to the current administrative territorial entity of the Century Lotus Stadium and the rest of the guangdong province?
4. Bancroft's county borders what county?

步数 11，等于所选 4 题跳数之和；自然命中 9 / 11。

### 模式：reselect

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：Upon whom does the President call on for support in his appointments to the USAF?
- 替换后的查询：Upon whom does the President call on for support in his appointments to the USAF?
- 证据段：编号 **12**，语料 pid 15350；标题：United States Air Force
- 证据正文前 200 字符："The Department of the Air Force is one of three military departments within the Department of Defense, and is managed by the civilian Secretary of the Air Force, under the authority, direction, and co"
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[12]`
- oracle / SFT 动作：`{"add":[12],"drop":[]}`
- 执行后保留：`[12]`
- 执行后 token：129 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：what is the majority political party in the #1 now
- 替换后的查询：what is the majority political party in the Senate now
- 证据段：编号 **27**，语料 pid 15143；标题：Party leaders of the United States Senate
- 证据正文前 200 字符："The Senate is currently composed of 51 Republicans, 47 Democrats, and 2 independents, both of whom caucus with the Democrats."
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]`
- 新返回编号（去除已保留）：`[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]`
- 已保留编号：`[12]`
- 当前可用金标准编号：`[12,27]`
- oracle / SFT 动作：`{"add":[27],"drop":[]}`
- 执行后保留：`[12,27]`
- 执行后 token：166 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 1 题，第 3 跳）

- 原始子问题：when did the #2 gain control of the house
- 替换后的查询：when did the Republicans gain control of the house
- 证据段：编号 **50**，语料 pid 15150；标题：2010 United States House of Representatives elections
- 证据正文前 200 字符："Republicans regained control of the chamber they had lost in the 2006 midterm elections, picking up a net total of 63 seats and erasing the gains Democrats made in 2006 and 2008. Although the sitting "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[40,31,41,42,43,44,45,46,47,48,49,50,51,52,53,23,54,55,56,57]`
- 新返回编号（去除已保留）：`[40,31,41,42,43,44,45,46,47,48,49,50,51,52,53,23,54,55,56,57]`
- 已保留编号：`[12,27]`
- 当前可用金标准编号：`[12,27,50]`
- oracle / SFT 动作：`{"add":[50],"drop":[]}`
- 执行后保留：`[12,27,50]`
- 执行后 token：468 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 1 跳）

- 原始子问题：J. P. Hayes >> place of birth
- 替换后的查询：J. P. Hayes >> place of birth
- 证据段：编号 **73**，语料 pid 17607；标题：J. P. Hayes
- 证据正文前 200 字符："Hayes was born and raised in Appleton, Wisconsin. He attended the University of Texas at El Paso, where he was a member of the golf team. He majored in marketing and graduated in 1988. In 1989, he tur"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]`
- 新返回编号（去除已保留）：`[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]`
- 已保留编号：`[12,27,50]`
- 当前可用金标准编号：`[12,27,50,73]`
- oracle / SFT 动作：`{"add":[73],"drop":[]}`
- 执行后保留：`[12,27,50,73]`
- 执行后 token：531 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 2 题，第 2 跳）

- 原始子问题：#1 >> capital of
- 替换后的查询：Appleton >> capital of
- 证据段：编号 **93**，语料 pid 17567；标题：John C. Petersen
- 证据正文前 200 字符："John C. Petersen (November 2, 1842 – July 10, 1887) was an American butcher and farmer from Appleton, Wisconsin who served as a member of the Wisconsin State Assembly from Outagamie County. He was ele"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[78,73,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]`
- 新返回编号（去除已保留）：`[78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]`
- 已保留编号：`[12,27,50,73]`
- 当前可用金标准编号：`[12,27,50,73,93]`
- oracle / SFT 动作：`{"add":[93],"drop":[]}`
- 执行后保留：`[12,27,50,73,93]`
- 执行后 token：631 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 2 题，第 3 跳）

- 原始子问题：#2 >> shares border with
- 替换后的查询：Outagamie County >> shares border with
- 证据段：编号 **114**，语料 pid 17577；标题：Pulaski High School
- 证据正文前 200 字符："Pulaski High School is a public high school in Pulaski, Wisconsin, in Brown County, Wisconsin (school district also serves parts of Shawano, Outagamie and Oconto counties), that serves students in gra"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,93,113,114,115]`
- 新返回编号（去除已保留）：`[97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]`
- 已保留编号：`[12,27,50,73,93]`
- 当前可用金标准编号：`[12,27,50,73,93,114]`
- oracle / SFT 动作：`{"add":[114],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114]`
- 执行后 token：699 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 2 题，第 4 跳）

- 原始子问题：#3 >> capital
- 替换后的查询：Brown County >> capital
- 证据段：编号 **127**，语料 pid 17572；标题：Jerome Quinn
- 证据正文前 200 字符："Born in Green Bay, Wisconsin, Quinn was a realtor and served on the Green Bay Common Council, the Brown County, Wisconsin Board of Supervisors, the local Board of Education, and the Wisconsin State As"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[110,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]`
- 新返回编号（去除已保留）：`[110,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]`
- 已保留编号：`[12,27,50,73,93,114]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127]`
- oracle / SFT 动作：`{"add":[127],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127]`
- 执行后 token：763 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 3 题，第 1 跳）

- 原始子问题：Century Lotus Stadium >> located in the administrative territorial entity
- 替换后的查询：Century Lotus Stadium >> located in the administrative territorial entity
- 证据段：编号 **148**，语料 pid 3701；标题：Century Lotus Stadium
- 证据正文前 200 字符："Century Lotus Stadium () is a multi-purpose stadium in Foshan, China. It is currently used mostly for football matches. The stadium holds 36,686 people and was built in 2005. It hosted a first round m"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154]`
- 新返回编号（去除已保留）：`[135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154]`
- 已保留编号：`[12,27,50,73,93,114,127]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148]`
- oracle / SFT 动作：`{"add":[148],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148]`
- 执行后 token：858 / 3000；recall：1.000000
- 自动检查：通过

#### 第 9 步（第 3 题，第 2 跳）

- 原始子问题：what year did the japanese get to #1 and the rest of guangdong province
- 替换后的查询：what year did the japanese get to Foshan and the rest of guangdong province
- 证据段：编号 **172**，语料 pid 3660；标题：Twenty-First Army (Japan)
- 证据正文前 200 字符："On October 12, the 18th and 104th Divisions landed, followed by command units the following day. By October 21, the provincial capital of Guangzhou was under Japanese control. The IJA 5th Division con"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174]`
- 新返回编号（去除已保留）：`[155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174]`
- 已保留编号：`[12,27,50,73,93,114,127,148]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172]`
- oracle / SFT 动作：`{"add":[172],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172]`
- 执行后 token：953 / 3000；recall：1.000000
- 自动检查：通过

#### 第 10 步（第 4 题，第 1 跳）

- 原始子问题：Bancroft >> located in the administrative territorial entity
- 替换后的查询：Bancroft >> located in the administrative territorial entity
- 证据段：编号 **190**，语料 pid 759；标题：North Hastings High School
- 证据正文前 200 字符："North Hastings High School (NHHS) is a high school located in Bancroft, Ontario, Canada serving students in the northern portion of Hastings County and part of the Hastings and Prince Edward District "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[175,143,176,177,178,179,180,181,141,182,183,184,185,186,139,187,188,189,190,191]`
- 新返回编号（去除已保留）：`[175,143,176,177,178,179,180,181,141,182,183,184,185,186,139,187,188,189,190,191]`
- 已保留编号：`[12,27,50,73,93,114,127,148,172]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172,190]`
- oracle / SFT 动作：`{"add":[190],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172,190]`
- 执行后 token：1033 / 3000；recall：1.000000
- 自动检查：通过

#### 第 11 步（第 4 题，第 2 跳）

- 原始子问题：#1 >> shares border with
- 替换后的查询：Hastings County >> shares border with
- 证据段：编号 **192**，语料 pid 758；标题：York River (Ontario)
- 证据正文前 200 字符："The York River is a river in Renfrew County, Hastings County and Haliburton County in Ontario, Canada. The river is in the Saint Lawrence River drainage basin, and flows from the southern extension of"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[104,192,108,111,113,190,110,100,105,97,193,106,194,99,102,195,107,196,98,103]`
- 新返回编号（去除已保留）：`[104,192,108,111,113,110,100,105,97,193,106,194,99,102,195,107,196,98,103]`
- 已保留编号：`[12,27,50,73,93,114,127,148,172,190]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172,190,192]`
- oracle / SFT 动作：`{"add":[192],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172,190,192]`
- 执行后 token：1093 / 3000；recall：1.000000
- 自动检查：通过

<details><summary>首步模型输入模板示例（已保留 / 新返回）</summary>

```text
任务问题：
1. When did the party with a majority in the body the President calls on for support of his appointments to the USAF gain control of the House?
2. What is the seat of the county sharing a border with the county in which J. P. Hayes was born?
3. What year did the japanese get to the current administrative territorial entity of the Century Lotus Stadium and the rest of the guangdong province?
4. Bancroft's county borders what county?
模式：reselect
预算：3000 token
已用 token：0

已保留：
（空）

新返回：
[编号 0 | 193 token]
Separation of powers under the United States Constitution
Executive power is vested, with exceptions and qualifications, in the President. By law (Section 2.) the president becomes the Commander in Chief of the Army and Navy, Militia of several states when called into service, has power to make treaties and appointments to office "with the Advice and Consent of the Senate," receive Ambassadors and Public Ministers, and "take care that the laws be faithfully executed" (Section 3.) By using these words, the Constitution does not require the president to personally enforce the law; rather, officers subordinate to the president may perform such duties. The Constitution empowers the president to ensure the faithful execution of the laws made by Congress and approved by the President. Congress may itself terminate such appointments, by impeachment, and restrict the president. Bodies such as the War Claims Commission, the Interstate Commerce Commission and the Federal Trade Commission—all quasi-judicial—often have direct Congressional oversight.
[编号 1 | 74 token]
Presidential Succession Act
(1) If, by reason of death, resignation, removal from office, inability, or failure to qualify, there is neither a President nor Vice President to discharge the powers and duties of the office of President, then the Speaker of the House of Representatives shall, upon his resignation as Speaker and as Representative in Congress, act as President.
[编号 2 | 114 token]
Jack McFarland
In season three, Jack undertakes a search for his biological father, which ends when he discovers that his father died several years back. Jack does, however, meet Elliot (Michael Angarano), his own biological son, whom he fathered by depositing at a sperm bank as a teenager so that he could buy a leather jacket. He later discovers that the boy's mother Bonnie (Rosie O'Donnell) is a lesbian; she was a nurse at the sperm bank who stole Jack's deposit and was inseminated with it.
[编号 3 | 147 token]
Air-One Emergency Response Coalition
The AIR-ONE Emergency Response Coalition (formerly named the Law Enforcement Aviation Coalition) supports agencies who provide air support to law enforcement, emergency management and fire/EMS agencies in Illinois, United States. Known by their "AIR-ONE" call sign, the helicopters are called upon by agencies who would otherwise not have air support capabilities available. AIR-ONE is a not-for-profit, 501(c)3, charitable organization, managed and staffed by volunteers from the aviation, law enforcement, emergency management and fire-rescue sectors. As the founding and coordinating agency, the Winthrop Harbor (Illinois) Police Department provides 24-hour dispatch services for all of the AIR-ONE helicopters.
[编号 4 | 114 token]
Gilbert Gottfried
In March 2011, Gottfried made a series of jokes on his Twitter account about the earthquake disaster in Japan. Aflac, which does 75% of its business in Japan, responded by dismissing Gottfried from voicing its duck mascot on March 14, 2011, and announced a casting call for his replacement as the voice of the duck. He was replaced by Daniel McKeague (who did an impression of Gottfried) on April 26, 2011.
[编号 5 | 166 token]
Dwight D. Eisenhower
Due to a complete estrangement between the two as a result of campaigning, Truman and Eisenhower had minimal discussions about the transition of administrations. After selecting his budget director, Joseph M. Dodge, Eisenhower asked Herbert Brownell and Lucius Clay to make recommendations for his cabinet appointments. He accepted their recommendations without exception; they included John Foster Dulles and George M. Humphrey with whom he developed his closest relationships, and one woman, Oveta Culp Hobby. Eisenhower's cabinet, consisting of several corporate executives and one labor leader, was dubbed by one journalist, "Eight millionaires and a plumber." The cabinet was notable for its lack of personal friends, office seekers, or experienced government administrators. He also upgraded the role of the National Security Council in planning all phases of the Cold War.
[编号 6 | 103 token]
999 (emergency telephone number)
999 or 112 is used to contact the emergency services upon witnessing or being involved in an emergency. In the United Kingdom, the numbers 999 and 112 both connect to the same service, and there is no priority or charge for either of them. Callers dialling 911, the USA's emergency number, will be transferred to the 999 call system if the call is made within the United Kingdom.
[编号 7 | 113 token]
Supreme Court of India
As per the Constitution, as held by the court in the Three Judges' Cases -- (1982, 1993, 1998), a judge is appointed to the Supreme Court by the President of India on the recommendation of the collegium -- a closed group of the Chief Justice of India, the four most senior judges of the court and the senior-most judge hailing from the high court of a prospective appointee. This has resulted in a Memorandum of Procedure being followed, for the appointments.
[编号 8 | 234 token]
Kievan Rus'
Ahmad ibn Fadlan, an Arab traveler during the 10th century, provided one of the earliest written descriptions of the Rus': "They are as tall as a date palm, blond and ruddy, so that they do not need to wear a tunic nor a cloak; rather the men among them wear garments that only cover half of his body and leaves one of his hands free." Liutprand of Cremona, who was twice an envoy to the Byzantine court (949 and 968), identifies the "Russi" with the Norse ("the Russi, whom we call Norsemen by another name") but explains the name as a Greek term referring to their physical traits ("A certain people made up of a part of the Norse, whom the Greeks call [...] the Russi on account of their physical features, we designate as Norsemen because of the location of their origin."). Leo the Deacon, a 10th-century Byzantine historian and chronicler, refers to the Rus' as "Scythians" and notes that they tended to adopt Greek rituals and customs.
[编号 9 | 133 token]
Dell
Dell service and support brands include the Dell Solution Station (extended domestic support services, previously "Dell on Call"), Dell Support Center (extended support services abroad), Dell Business Support (a commercial service-contract that provides an industry-certified technician with a lower call-volume than in normal queues), Dell Everdream Desktop Management ("Software as a Service" remote-desktop management, originally a SaaS company founded by Elon Musk's cousin, Lyndon Rive, which Dell bought in 2007), and Your Tech Team (a support-queue available to home users who purchased their systems either through Dell's website or through Dell phone-centers).
[编号 10 | 146 token]
1912 United States presidential election
Roosevelt had served as president from 1901 to 1909, and Taft had won the 1908 Republican president nomination with Roosevelt's support. Displeased with Taft's actions as president, Roosevelt challenged Taft at the 1912 Republican National Convention. After Taft and his conservative allies narrowly prevailed at the Republican convention, Roosevelt rallied his progressive supporters and launched a third party bid. With the support of William Jennings Bryan and other progressives, Wilson won the Democratic Party's presidential nomination on the 46th ballot. Meanwhile, the Socialist Party renominated its perennial standard - bearer, Eugene V. Debs.
[编号 11 | 208 token]
Fairchild Republic A-10 Thunderbolt II
The Fairchild Republic A-10 Thunderbolt II is a single - seat, twin turbofan engine, straight wing jet aircraft developed by Fairchild - Republic for the United States Air Force (USAF). Commonly referred to by the nicknames ``Warthog ''or`` Hog'', its official name comes from the Republic P - 47 Thunderbolt, a World War II fighter - bomber effective at attacking ground targets. The A-10 was designed for close air support (CAS) of friendly ground troops, attacking armored vehicles and tanks, and providing quick - action support against enemy ground forces. It entered service in 1976 and is the only production - built aircraft that has served in the USAF that was designed solely for CAS. Its secondary mission is to provide forward air controller -- airborne (FAC - A) support, by directing other aircraft in attacks on ground targets. Aircraft used primarily in this role are designated OA - 10.
[编号 12 | 129 token]
United States Air Force
The Department of the Air Force is one of three military departments within the Department of Defense, and is managed by the civilian Secretary of the Air Force, under the authority, direction, and control of the Secretary of Defense. The senior officials in the Office of the Secretary are the Under Secretary of the Air Force, four Assistant Secretaries of the Air Force and the General Counsel, all of whom are appointed by the President with the advice and consent of the Senate. The senior uniformed leadership in the Air Staff is made up of the Chief of Staff of the Air Force and the Vice Chief of Staff of the Air Force.
[编号 13 | 451 token]
Director of the National Security Agency
# Director Photo Service Term President (s) served under MG Ralph Canine USA 1952 -- 1956 Harry S. Truman Dwight D. Eisenhower Lt Gen John Samford USAF 1956 -- 1960 Dwight D. Eisenhower VADM Laurence Frost USN 1960 -- 1962 Dwight D. Eisenhower John F. Kennedy Lt Gen Gordon Blake USAF 1962 -- 1965 John F. Kennedy Lyndon B. Johnson LTG Marshall Carter USA 1965 -- 1969 Lyndon B. Johnson Richard Nixon 6 VADM Noel Gayler USN 1969 -- 1972 Richard Nixon 7 Lt Gen Samuel C. Phillips USAF 1972 -- 1973 Richard Nixon 8 Lt Gen Lew Allen USAF 1973 -- 1977 Richard Nixon Gerald Ford Jimmy Carter 9 VADM Bobby Ray Inman USN 1977 -- 1981 Jimmy Carter Ronald Reagan 10 Lt Gen Lincoln Faurer USAF 1981 -- 1985 Ronald Reagan 11 LTG William Odom USA 1985 -- 1988 Ronald Reagan 12 VADM William Studeman USN 1988 -- 1992 Ronald Reagan George H.W. Bush 13 VADM John M. McConnell USN 1992 -- 1996 George H.W. Bush Bill Clinton 14 Lt Gen Kenneth A. Minihan USAF 1996 -- 1999 Bill Clinton 15 Lt Gen Michael Hayden USAF 1999 -- 2005 Bill Clinton George W. Bush 16 LTG / GEN Keith B. Alexander USA August 1, 2005 -- March 28, 2014 George W. Bush Barack Obama 17 ADM Michael S. Rogers USN April 2, 2014 -- present Barack Obama Donald Trump
[编号 14 | 222 token]
United States Air Force
The fighter aircraft of the USAF are small, fast, and maneuverable military aircraft primarily used for air-to-air combat. Many of these fighters have secondary ground-attack capabilities, and some are dual-roled as fighter-bombers (e.g., the F-16 Fighting Falcon); the term "fighter" is also sometimes used colloquially for dedicated ground-attack aircraft. Other missions include interception of bombers and other fighters, reconnaissance, and patrol. The F-16 is currently used by the USAF Air Demonstration squadron, the Thunderbirds, while a small number of both man-rated and non-man-rated F-4 Phantom II are retained as QF-4 aircraft for use as Full Scale Aerial Targets (FSAT) or as part of the USAF Heritage Flight program. These extant QF-4 aircraft are being replaced in the FSAT role by early model F-16 aircraft converted to QF-16 configuration. The USAF has 2,025 fighters in service as of September 2012.
[编号 15 | 152 token]
President-elect of the United States
No constitutional provision or federal law requires electors to vote according to the results of their state's popular vote, though some states bind their electors to their pledges by state law. Historically, there have been only a few instances of ``faithless electors ''casting their ballots for a candidate to whom they were not pledged, and such instances have never resulted in changing the final outcome of a presidential election. The popular vote does not declare the presidency but the Electoral College vote does. Even if the popular vote goes to one candidate, another may win the electoral vote and the presidency, as has happened in 1876, 1888, 2000 and 2016.
[编号 16 | 124 token]
John Kerry
During his bid to be elected president in 2004, Kerry frequently criticized President George W. Bush for the Iraq War. While Kerry had initially voted in support of authorizing President Bush to use force in dealing with Saddam Hussein, he voted against an $87 billion supplemental appropriations bill to pay for the subsequent war. His statement on March 16, 2004, "I actually did vote for the $87 billion before I voted against it," helped the Bush campaign to paint him as a flip-flopper and has been cited as contributing to Kerry's defeat.
[编号 17 | 81 token]
GNU Manifesto
The GNU Manifesto was written by Richard Stallman and published in March 1985 in "Dr. Dobb's Journal of Software Tools" as an explanation of goals of the GNU Project, and as a call for support and participation in developing GNU, a free software computer operating system. It is held in high regard within the free software movement as a fundamental philosophical source.
[编号 18 | 144 token]
State of the Union
The address fulfills rules in Article II, Section 3 of the U.S. Constitution, requiring the President to periodically ``give to the Congress Information of the State of the Union, and recommend to their Consideration such measures as he shall judge necessary and expedient. ''During most of the country's first century, the President primarily only submitted a written report to Congress. After 1913, Woodrow Wilson, the 28th U.S. President, began the regular practice of delivering the address to Congress in person as a way to rally support for his agenda. With the advent of radio and television, the address is now broadcast live across the country on many networks.
[编号 19 | 143 token]
Windows 8
Internet Explorer 10 is included as both a desktop program and a touch-optimized app, and includes increased support for HTML5, CSS3, and hardware acceleration. The Internet Explorer app does not support plugins or ActiveX components, but includes a version of Adobe Flash Player that is optimized for touch and low power usage. Initially, Adobe Flash would only work on sites included on a "Compatibility View" whitelist; however, after feedback from users and additional compatibility tests, an update in March 2013 changed this behavior to use a smaller blacklist of sites with known compatibility issues instead, allowing Flash to be used on most sites by default. The desktop version does not contain these limitations.
```

</details>

### 模式：append

#### 第 1 步（第 1 题，第 1 跳）

- 原始子问题：Upon whom does the President call on for support in his appointments to the USAF?
- 替换后的查询：Upon whom does the President call on for support in his appointments to the USAF?
- 证据段：编号 **12**，语料 pid 15350；标题：United States Air Force
- 证据正文前 200 字符："The Department of the Air Force is one of three military departments within the Department of Defense, and is managed by the civilian Secretary of the Air Force, under the authority, direction, and co"
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 新返回编号（去除已保留）：`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]`
- 已保留编号：`[]`
- 当前可用金标准编号：`[12]`
- oracle / SFT 动作：`{"add":[12],"drop":[]}`
- 执行后保留：`[12]`
- 执行后 token：129 / 3000；recall：1.000000
- 自动检查：通过

#### 第 2 步（第 1 题，第 2 跳）

- 原始子问题：what is the majority political party in the #1 now
- 替换后的查询：what is the majority political party in the Senate now
- 证据段：编号 **27**，语料 pid 15143；标题：Party leaders of the United States Senate
- 证据正文前 200 字符："The Senate is currently composed of 51 Republicans, 47 Democrats, and 2 independents, both of whom caucus with the Democrats."
- 自然命中：否（已替换检索末位候选）
- 本步工具返回编号（含已保留的重复返回）：`[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]`
- 新返回编号（去除已保留）：`[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]`
- 已保留编号：`[12]`
- 当前可用金标准编号：`[12,27]`
- oracle / SFT 动作：`{"add":[27],"drop":[]}`
- 执行后保留：`[12,27]`
- 执行后 token：166 / 3000；recall：1.000000
- 自动检查：通过

#### 第 3 步（第 1 题，第 3 跳）

- 原始子问题：when did the #2 gain control of the house
- 替换后的查询：when did the Republicans gain control of the house
- 证据段：编号 **50**，语料 pid 15150；标题：2010 United States House of Representatives elections
- 证据正文前 200 字符："Republicans regained control of the chamber they had lost in the 2006 midterm elections, picking up a net total of 63 seats and erasing the gains Democrats made in 2006 and 2008. Although the sitting "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[40,31,41,42,43,44,45,46,47,48,49,50,51,52,53,23,54,55,56,57]`
- 新返回编号（去除已保留）：`[40,31,41,42,43,44,45,46,47,48,49,50,51,52,53,23,54,55,56,57]`
- 已保留编号：`[12,27]`
- 当前可用金标准编号：`[12,27,50]`
- oracle / SFT 动作：`{"add":[50],"drop":[]}`
- 执行后保留：`[12,27,50]`
- 执行后 token：468 / 3000；recall：1.000000
- 自动检查：通过

#### 第 4 步（第 2 题，第 1 跳）

- 原始子问题：J. P. Hayes >> place of birth
- 替换后的查询：J. P. Hayes >> place of birth
- 证据段：编号 **73**，语料 pid 17607；标题：J. P. Hayes
- 证据正文前 200 字符："Hayes was born and raised in Appleton, Wisconsin. He attended the University of Texas at El Paso, where he was a member of the golf team. He majored in marketing and graduated in 1988. In 1989, he tur"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]`
- 新返回编号（去除已保留）：`[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]`
- 已保留编号：`[12,27,50]`
- 当前可用金标准编号：`[12,27,50,73]`
- oracle / SFT 动作：`{"add":[73],"drop":[]}`
- 执行后保留：`[12,27,50,73]`
- 执行后 token：531 / 3000；recall：1.000000
- 自动检查：通过

#### 第 5 步（第 2 题，第 2 跳）

- 原始子问题：#1 >> capital of
- 替换后的查询：Appleton >> capital of
- 证据段：编号 **93**，语料 pid 17567；标题：John C. Petersen
- 证据正文前 200 字符："John C. Petersen (November 2, 1842 – July 10, 1887) was an American butcher and farmer from Appleton, Wisconsin who served as a member of the Wisconsin State Assembly from Outagamie County. He was ele"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[78,73,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]`
- 新返回编号（去除已保留）：`[78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]`
- 已保留编号：`[12,27,50,73]`
- 当前可用金标准编号：`[12,27,50,73,93]`
- oracle / SFT 动作：`{"add":[93],"drop":[]}`
- 执行后保留：`[12,27,50,73,93]`
- 执行后 token：631 / 3000；recall：1.000000
- 自动检查：通过

#### 第 6 步（第 2 题，第 3 跳）

- 原始子问题：#2 >> shares border with
- 替换后的查询：Outagamie County >> shares border with
- 证据段：编号 **114**，语料 pid 17577；标题：Pulaski High School
- 证据正文前 200 字符："Pulaski High School is a public high school in Pulaski, Wisconsin, in Brown County, Wisconsin (school district also serves parts of Shawano, Outagamie and Oconto counties), that serves students in gra"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,93,113,114,115]`
- 新返回编号（去除已保留）：`[97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]`
- 已保留编号：`[12,27,50,73,93]`
- 当前可用金标准编号：`[12,27,50,73,93,114]`
- oracle / SFT 动作：`{"add":[114],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114]`
- 执行后 token：699 / 3000；recall：1.000000
- 自动检查：通过

#### 第 7 步（第 2 题，第 4 跳）

- 原始子问题：#3 >> capital
- 替换后的查询：Brown County >> capital
- 证据段：编号 **127**，语料 pid 17572；标题：Jerome Quinn
- 证据正文前 200 字符："Born in Green Bay, Wisconsin, Quinn was a realtor and served on the Green Bay Common Council, the Brown County, Wisconsin Board of Supervisors, the local Board of Education, and the Wisconsin State As"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[110,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]`
- 新返回编号（去除已保留）：`[110,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]`
- 已保留编号：`[12,27,50,73,93,114]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127]`
- oracle / SFT 动作：`{"add":[127],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127]`
- 执行后 token：763 / 3000；recall：1.000000
- 自动检查：通过

#### 第 8 步（第 3 题，第 1 跳）

- 原始子问题：Century Lotus Stadium >> located in the administrative territorial entity
- 替换后的查询：Century Lotus Stadium >> located in the administrative territorial entity
- 证据段：编号 **148**，语料 pid 3701；标题：Century Lotus Stadium
- 证据正文前 200 字符："Century Lotus Stadium () is a multi-purpose stadium in Foshan, China. It is currently used mostly for football matches. The stadium holds 36,686 people and was built in 2005. It hosted a first round m"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154]`
- 新返回编号（去除已保留）：`[135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154]`
- 已保留编号：`[12,27,50,73,93,114,127]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148]`
- oracle / SFT 动作：`{"add":[148],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148]`
- 执行后 token：858 / 3000；recall：1.000000
- 自动检查：通过

#### 第 9 步（第 3 题，第 2 跳）

- 原始子问题：what year did the japanese get to #1 and the rest of guangdong province
- 替换后的查询：what year did the japanese get to Foshan and the rest of guangdong province
- 证据段：编号 **172**，语料 pid 3660；标题：Twenty-First Army (Japan)
- 证据正文前 200 字符："On October 12, the 18th and 104th Divisions landed, followed by command units the following day. By October 21, the provincial capital of Guangzhou was under Japanese control. The IJA 5th Division con"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174]`
- 新返回编号（去除已保留）：`[155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174]`
- 已保留编号：`[12,27,50,73,93,114,127,148]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172]`
- oracle / SFT 动作：`{"add":[172],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172]`
- 执行后 token：953 / 3000；recall：1.000000
- 自动检查：通过

#### 第 10 步（第 4 题，第 1 跳）

- 原始子问题：Bancroft >> located in the administrative territorial entity
- 替换后的查询：Bancroft >> located in the administrative territorial entity
- 证据段：编号 **190**，语料 pid 759；标题：North Hastings High School
- 证据正文前 200 字符："North Hastings High School (NHHS) is a high school located in Bancroft, Ontario, Canada serving students in the northern portion of Hastings County and part of the Hastings and Prince Edward District "
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[175,143,176,177,178,179,180,181,141,182,183,184,185,186,139,187,188,189,190,191]`
- 新返回编号（去除已保留）：`[175,143,176,177,178,179,180,181,141,182,183,184,185,186,139,187,188,189,190,191]`
- 已保留编号：`[12,27,50,73,93,114,127,148,172]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172,190]`
- oracle / SFT 动作：`{"add":[190],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172,190]`
- 执行后 token：1033 / 3000；recall：1.000000
- 自动检查：通过

#### 第 11 步（第 4 题，第 2 跳）

- 原始子问题：#1 >> shares border with
- 替换后的查询：Hastings County >> shares border with
- 证据段：编号 **192**，语料 pid 758；标题：York River (Ontario)
- 证据正文前 200 字符："The York River is a river in Renfrew County, Hastings County and Haliburton County in Ontario, Canada. The river is in the Saint Lawrence River drainage basin, and flows from the southern extension of"
- 自然命中：是
- 本步工具返回编号（含已保留的重复返回）：`[104,192,108,111,113,190,110,100,105,97,193,106,194,99,102,195,107,196,98,103]`
- 新返回编号（去除已保留）：`[104,192,108,111,113,110,100,105,97,193,106,194,99,102,195,107,196,98,103]`
- 已保留编号：`[12,27,50,73,93,114,127,148,172,190]`
- 当前可用金标准编号：`[12,27,50,73,93,114,127,148,172,190,192]`
- oracle / SFT 动作：`{"add":[192],"drop":[]}`
- 执行后保留：`[12,27,50,73,93,114,127,148,172,190,192]`
- 执行后 token：1093 / 3000；recall：1.000000
- 自动检查：通过

