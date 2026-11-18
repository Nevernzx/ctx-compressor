# Qwen3-8B 阅读小样

前5个任务；oracle|reselect|mid 与 no_compress 各20题。
模板、greedy、max_tokens=48、bf16、关闭thinking与全量保持一致。

自动检查：`{"n":40,"length_finished":1,"empty_predictions":0,"long_answers_over_20_words":1,"unclosed_think":0,"max_no_compress_prompt_tokens":32382,"automatic_pass":false}`。

## no_compress / task 0 / question 0

- 问题：What character does the actor who played Barney on How I Met Your Mother play in Batman Under the Red Hood?
- 标准答案："Nightwing / Dick Grayson"；aliases：["Nightwing","Dick Grayson","Batman","Robin"]
- prompt token：17946；output token：3；finish：stop
- 原始输出："Nightwing"
- 解析结果："Nightwing"
- EM：1.0；F1：1.0

## no_compress / task 0 / question 1

- 问题：Who is the father of the Labo M performer?
- 标准答案："Louis Chedid"；aliases：[]
- prompt token：17935；output token：5；finish：stop
- 原始输出："Louis Chedid"
- 解析结果："Louis Chedid"
- EM：1.0；F1：1.0

## no_compress / task 0 / question 2

- 问题：What is the position of the first governor general of India?
- 标准答案："Governor-General of India"；aliases：["Viceroy of India"]
- prompt token：17936；output token：4；finish：stop
- 原始输出："Warren Hastings"
- 解析结果："Warren Hastings"
- EM：0.0；F1：0.0

## no_compress / task 0 / question 3

- 问题：When did the person who took over the territory and made it part of their empire die?
- 标准答案："323 BC"；aliases：[]
- prompt token：17942；output token：5；finish：stop
- 原始输出："1818"
- 解析结果："1818"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 0 / question 0

- 问题：What character does the actor who played Barney on How I Met Your Mother play in Batman Under the Red Hood?
- 标准答案："Nightwing / Dick Grayson"；aliases：["Nightwing","Dick Grayson","Batman","Robin"]
- prompt token：1291；output token：7；finish：stop
- 原始输出："Nightwing / Dick Grayson"
- 解析结果："Nightwing / Dick Grayson"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 0 / question 1

- 问题：Who is the father of the Labo M performer?
- 标准答案："Louis Chedid"；aliases：[]
- prompt token：1280；output token：5；finish：stop
- 原始输出："Louis Chedid"
- 解析结果："Louis Chedid"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 0 / question 2

- 问题：What is the position of the first governor general of India?
- 标准答案："Governor-General of India"；aliases：["Viceroy of India"]
- prompt token：1281；output token：7；finish：stop
- 原始输出："Governor-General of British India"
- 解析结果："Governor-General of British India"
- EM：0.0；F1：0.8571428571428571

## oracle|reselect|mid / task 0 / question 3

- 问题：When did the person who took over the territory and made it part of their empire die?
- 标准答案："323 BC"；aliases：[]
- prompt token：1287；output token：5；finish：stop
- 原始输出："1818"
- 解析结果："1818"
- EM：0.0；F1：0.0

## no_compress / task 1 / question 0

- 问题：Who, along with the author of An Essay on the History of Civil Society, criticized the idea of social contract theorists?
- 标准答案："David Hume"；aliases：["Hume"]
- prompt token：21323；output token：3；finish：stop
- 原始输出："Adam Ferguson"
- 解析结果："Adam Ferguson"
- EM：0.0；F1：0.0

## no_compress / task 1 / question 1

- 问题：When did the majority party in the body which determines the rules of the US House and US Senate gain control of the House?
- 标准答案："the 2010 election"；aliases：[]
- prompt token：21324；output token：5；finish：stop
- 原始输出："1994"
- 解析结果："1994"
- EM：0.0；F1：0.0

## no_compress / task 1 / question 2

- 问题：Who owns the record label of the Till I Can Gain Control Again songwriter?
- 标准答案："Warner Music Group"；aliases：["Warner Music"]
- prompt token：21314；output token：6；finish：stop
- 原始输出："Warner Bros. Records"
- 解析结果："Warner Bros. Records"
- EM：0.0；F1：0.4

## no_compress / task 1 / question 3

- 问题：What year did the group that performed From Them, Through Us, to You form?
- 标准答案："2005"；aliases：[]
- prompt token：21316；output token：5；finish：stop
- 原始输出："2005"
- 解析结果："2005"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 1 / question 0

- 问题：Who, along with the author of An Essay on the History of Civil Society, criticized the idea of social contract theorists?
- 标准答案："David Hume"；aliases：["Hume"]
- prompt token：1429；output token：4；finish：stop
- 原始输出："David Hume"
- 解析结果："David Hume"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 1 / question 1

- 问题：When did the majority party in the body which determines the rules of the US House and US Senate gain control of the House?
- 标准答案："the 2010 election"；aliases：[]
- prompt token：1430；output token：5；finish：stop
- 原始输出："2010"
- 解析结果："2010"
- EM：0.0；F1：0.6666666666666666

## oracle|reselect|mid / task 1 / question 2

- 问题：Who owns the record label of the Till I Can Gain Control Again songwriter?
- 标准答案："Warner Music Group"；aliases：["Warner Music"]
- prompt token：1420；output token：5；finish：stop
- 原始输出："Warner Music Group"
- 解析结果："Warner Music Group"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 1 / question 3

- 问题：What year did the group that performed From Them, Through Us, to You form?
- 标准答案："2005"；aliases：[]
- prompt token：1422；output token：5；finish：stop
- 原始输出："2005"
- 解析结果："2005"
- EM：1.0；F1：1.0

## no_compress / task 2 / question 0

- 问题：When did the party with a majority in the body the President calls on for support of his appointments to the USAF gain control of the House?
- 标准答案："the 2010 election"；aliases：[]
- prompt token：22401；output token：5；finish：stop
- 原始输出："2010"
- 解析结果："2010"
- EM：0.0；F1：0.6666666666666666

## no_compress / task 2 / question 1

- 问题：What is the seat of the county sharing a border with the county in which J. P. Hayes was born?
- 标准答案："Green Bay"；aliases：[]
- prompt token：22395；output token：5；finish：stop
- 原始输出："Hastings County"
- 解析结果："Hastings County"
- EM：0.0；F1：0.0

## no_compress / task 2 / question 2

- 问题：What year did the japanese get to the current administrative territorial entity of the Century Lotus Stadium and the rest of the guangdong province?
- 标准答案："November 5"；aliases：[]
- prompt token：22399；output token：5；finish：stop
- 原始输出："1980"
- 解析结果："1980"
- EM：0.0；F1：0.0

## no_compress / task 2 / question 3

- 问题：Bancroft's county borders what county?
- 标准答案："Haliburton County"；aliases：[]
- prompt token：22381；output token：5；finish：stop
- 原始输出："Hastings County"
- 解析结果："Hastings County"
- EM：0.0；F1：0.5

## oracle|reselect|mid / task 2 / question 0

- 问题：When did the party with a majority in the body the President calls on for support of his appointments to the USAF gain control of the House?
- 标准答案："the 2010 election"；aliases：[]
- prompt token：1173；output token：5；finish：stop
- 原始输出："2010"
- 解析结果："2010"
- EM：0.0；F1：0.6666666666666666

## oracle|reselect|mid / task 2 / question 1

- 问题：What is the seat of the county sharing a border with the county in which J. P. Hayes was born?
- 标准答案："Green Bay"；aliases：[]
- prompt token：1167；output token：3；finish：stop
- 原始输出："Appleton"
- 解析结果："Appleton"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 2 / question 2

- 问题：What year did the japanese get to the current administrative territorial entity of the Century Lotus Stadium and the rest of the guangdong province?
- 标准答案："November 5"；aliases：[]
- prompt token：1171；output token：5；finish：stop
- 原始输出："1940"
- 解析结果："1940"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 2 / question 3

- 问题：Bancroft's county borders what county?
- 标准答案："Haliburton County"；aliases：[]
- prompt token：1153；output token：5；finish：stop
- 原始输出："Hastings County"
- 解析结果："Hastings County"
- EM：0.0；F1：0.5

## no_compress / task 3 / question 0

- 问题：When did the maker of Acura Legend, the owner of Scion, and Nissan open US assembly plants?
- 标准答案："1981"；aliases：[]
- prompt token：29761；output token：6；finish：stop
- 原始输出："1980s"
- 解析结果："1980s"
- EM：0.0；F1：0.0

## no_compress / task 3 / question 1

- 问题：What is the Arabic dictionary meaning, of the word for the majority religion in the part of British India, that became India when the country that released Mushkil was created?
- 标准答案："the country of India"；aliases：["IND","IN","India","in","Republic of India"]
- prompt token：29773；output token：2；finish：stop
- 原始输出："Islam"
- 解析结果："Islam"
- EM：0.0；F1：0.0

## no_compress / task 3 / question 2

- 问题：Who was the artist who did a duet with Beyonce in the single "Deja Vu" talking about in the song "Cry"?
- 标准答案："three different relationships he had in the past"；aliases：[]
- prompt token：29768；output token：3；finish：stop
- 原始输出："Jay-Z"
- 解析结果："Jay-Z"
- EM：0.0；F1：0.0

## no_compress / task 3 / question 3

- 问题：What movement did the person who wanted to reform and address the church that the general population converted to lead?
- 标准答案："German Renaissance"；aliases：[]
- prompt token：29760；output token：6；finish：stop
- 原始输出："Protestant Reformation"
- 解析结果："Protestant Reformation"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 3 / question 0

- 问题：When did the maker of Acura Legend, the owner of Scion, and Nissan open US assembly plants?
- 标准答案："1981"；aliases：[]
- prompt token：2067；output token：5；finish：stop
- 原始输出："1981"
- 解析结果："1981"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 3 / question 1

- 问题：What is the Arabic dictionary meaning, of the word for the majority religion in the part of British India, that became India when the country that released Mushkil was created?
- 标准答案："the country of India"；aliases：["IND","IN","India","in","Republic of India"]
- prompt token：2079；output token：2；finish：stop
- 原始输出："Islam"
- 解析结果："Islam"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 3 / question 2

- 问题：Who was the artist who did a duet with Beyonce in the single "Deja Vu" talking about in the song "Cry"?
- 标准答案："three different relationships he had in the past"；aliases：[]
- prompt token：2074；output token：3；finish：stop
- 原始输出："Jay-Z"
- 解析结果："Jay-Z"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 3 / question 3

- 问题：What movement did the person who wanted to reform and address the church that the general population converted to lead?
- 标准答案："German Renaissance"；aliases：[]
- prompt token：2066；output token：3；finish：stop
- 原始输出："Reformation"
- 解析结果："Reformation"
- EM：0.0；F1：0.0

## no_compress / task 4 / question 0

- 问题：By approximately what percentage did the Black Death reduce the population of the region that the Soviet Union feared invasion from?
- 标准答案："30% to 65%"；aliases：[]
- prompt token：32358；output token：4；finish：stop
- 原始输出："30%"
- 解析结果："30%"
- EM：0.0；F1：0.5

## no_compress / task 4 / question 1

- 问题：Who won the Indy Car Race in the largest populated city of the state where the performer of Tijuana Moods is from?
- 标准答案："Mario Andretti"；aliases：[]
- prompt token：32362；output token：48；finish：length
- 原始输出："Tijuana Moods is an album by Charles Mingus. The largest populated city in the state where Charles Mingus is from (California) is Los Angeles. The Indy Car Race in Los Angeles is the 2001 Honda"
- 解析结果："Tijuana Moods is an album by Charles Mingus. The largest populated city in the state where Charles Mingus is from (California) is Los Angeles. The Indy Car Race in Los Angeles is the 2001 Honda"
- EM：0.0；F1：0.0

## no_compress / task 4 / question 2

- 问题：How many miles of highways are in the city where the Yongle emperor greeted the person to whom the edict was addressed?
- 标准答案："140 mi"；aliases：[]
- prompt token：32361；output token：4；finish：stop
- 原始输出："140"
- 解析结果："140"
- EM：0.0；F1：0.6666666666666666

## no_compress / task 4 / question 3

- 问题：A country gained control of Florida after the war in which Charles Edward Nugent saw action. Besides being in different parts of that country, what other differences are there between Real Madrid and Hostafrancs's city's team?
- 标准答案："two cities"；aliases：[]
- prompt token：32382；output token：23；finish：stop
- 原始输出："Hostafrancs is a neighborhood in Barcelona, while Real Madrid is a football club based in Madrid."
- 解析结果："Hostafrancs is a neighborhood in Barcelona, while Real Madrid is a football club based in Madrid"
- EM：0.0；F1：0.0

## oracle|reselect|mid / task 4 / question 0

- 问题：By approximately what percentage did the Black Death reduce the population of the region that the Soviet Union feared invasion from?
- 标准答案："30% to 65%"；aliases：[]
- prompt token：1892；output token：4；finish：stop
- 原始输出："65%"
- 解析结果："65%"
- EM：0.0；F1：0.5

## oracle|reselect|mid / task 4 / question 1

- 问题：Who won the Indy Car Race in the largest populated city of the state where the performer of Tijuana Moods is from?
- 标准答案："Mario Andretti"；aliases：[]
- prompt token：1896；output token：5；finish：stop
- 原始输出："Mario Andretti"
- 解析结果："Mario Andretti"
- EM：1.0；F1：1.0

## oracle|reselect|mid / task 4 / question 2

- 问题：How many miles of highways are in the city where the Yongle emperor greeted the person to whom the edict was addressed?
- 标准答案："140 mi"；aliases：[]
- prompt token：1895；output token：4；finish：stop
- 原始输出："140"
- 解析结果："140"
- EM：0.0；F1：0.6666666666666666

## oracle|reselect|mid / task 4 / question 3

- 问题：A country gained control of Florida after the war in which Charles Edward Nugent saw action. Besides being in different parts of that country, what other differences are there between Real Madrid and Hostafrancs's city's team?
- 标准答案："two cities"；aliases：[]
- prompt token：1916；output token：6；finish：stop
- 原始输出："Real Madrid and FC Barcelona"
- 解析结果："Real Madrid and FC Barcelona"
- EM：0.0；F1：0.0


## 人工与独立复查后的处理

40条解析无错误、输入无截断。1条（2.5%）模型输出为35词解释并用满固定48-token额度；另1条为16词解释句。均保留原始输出并按固定规则评分，不修饰答案。初版自动检查把输出额度命中误当成输入截断而停止，尚未生成全量。复查纠正此额外工程门槛，模板、参数和解析均不变；复用原始40条小样并继续唯一一次12,000条全量。详见[复查记录](smoke_review.json)和[初次运行记录](run_config_smoke_initial.json)。
