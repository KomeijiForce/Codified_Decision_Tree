# Codified Decision Tree (CDT)

**一种从给定故事线中提取深层、可验证、结构化角色行为的算法。**

<img width="1451" height="774" alt="image" src="https://github.com/user-attachments/assets/d12431f8-c91f-4551-92ee-213fc75e97c6" />

<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - 让幻想照进现实

本仓库包含：

* 算法实现：基于角色的“场景-行动对（scene-action pairs）”构建 **Codified Decision Trees（CDT）**
* 基准评测：评估由 CDT 驱动的角色扮演（Role-playing）表现
* 自动画像：将 CDT 自动转换为更易读的 wiki 风格文本的脚本

## 如何使用？

* **初始化**

你需要在项目根目录创建一个 `constant.py` 文件，并在其中填入你的 OpenAI 和 HuggingFace token：

```python
openai_key = "..."
hf_token = "..."
```

你也可以参考下面的环境配置：

```
torch: 2.7.1+cu126
transformers: 4.55.0
sentence_transformers: 5.1.0
sklearn: 1.7.1
openai: 2.14.0
```

* **构建 CDT**

  <img width="1842" height="862" alt="main_fig_v2_cropped_cropped-1" src="https://github.com/user-attachments/assets/b686ce21-5b92-4987-9374-8197223e84bb" />

对于论文实验中涉及的角色，你可以使用 `build_cdt.sh` 脚本来复现对应的 CDT：

```sh
python codified_decision_tree.py \
  --character "Kasumi" \
  --engine "gpt-4.1" \
  --max_depth 3 \
  --threshold_accept 0.8 \
  --threshold_reject 0.5 \
  --threshold_filter 0.8 \
  --device_id 1
```

你也可以使用 `codified_decision_tree.py` 中提供的 `CDT_Node` 类，为任意角色构建 CDT：

```python
CDT_Node(character, goal_topic, pairs, built_statements, depth, established_statements, gate_path,
max_depth, threshold_accept, threshold_reject, threshold_filter)
```

参数说明：

* `character`：角色名称；

* `goal_topic`：你希望 CDT 聚焦的目标（话题/维度/方面）；

* `pairs`：训练数据，格式为：`[{"scene": "...", "action": "..."}, {"scene": "...", "action": "..."}, ...]`，其中 `character` 在 `scene` 中执行对应的 `action`；

* `built_statements`：用于节点生长，保持为 `None`；

* `depth`：用于基于深度的终止条件，保持为 `1`；

* `established_statements`：用于多样化（diversification），保持为 `[]`；

* `gate_path`：用于多样化（diversification），保持为 `[]`；

* `depth`：用于基于深度的终止条件，建议设置为 `3`；

* `threshold_accept`：控制“陈述（statement）接受”的精度阈值；

* `threshold_reject`：控制“假设（hypothesis）废除/驳回”的精度阈值；

* `threshold_filter`：控制“门控（gate）接受”的过滤强度阈值；

* `device_id`：你希望运行算法使用的 GPU 编号。

* **Grounding（支撑检索 / 依据提取）**

当你已经构建出一个 `CDT_Node`（例如 `cdt_tree`）后，可以使用 `cdt_tree.traverse(scene)`，为输入的 `scene` 从 CDT 中提取对应的 grounding 语句（支撑信息）。

* **基准评测 CDT**
  基准评测通过 `run_benchmark.sh` 在两个基准上运行：[Fine-grained Fandom Benchmark](https://huggingface.co/datasets/KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences) 与 [Bandori Conversational Benchmark](https://huggingface.co/datasets/KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences)。这些链接中包含 16 条主线故事里主要角色的行动序列（action sequences），并会通过 `load_ar_pairs` 函数进一步处理并拆分为训练集和测试集。

```python
python run_benchmark.py \
  --character "Kasumi" \
  --method "cdt_package" \
  --engine "gpt-4.1" \
  --eval_engine "gpt-4.1" \
  --generator_path "meta-llama/Llama-3.1-8B-Instruct" \
  --device_id 1
```

* **Wikification（WIKI 化）**

我们提供了一个示例 notebook，用于将 CDT 转换为更易读的 wiki 风格角色百科，见此处：
[Wikification.ipynb](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/Wikification.ipynb)

该 notebook 会接收以下参数作为输入（下方给出示例值）：

```
character = "戸山香澄"
cdt_id = "Kasumi"
lang = "Chinese"
content = f"{character}"
note = '''
# Notes
Kasumi -> 戸山香澄
Arisa -> 市谷有咲
Rimi -> 牛込里美
Tae -> 花园多惠
Saaya -> 山吹沙绫
'''
```

* `character`：你希望在 wiki 页面中使用的角色名（显示名）
* `cdt_id`：用于构建 CDT 时使用的角色名（ID）
* `lang`：你希望生成 wiki 页面所使用的语言
* `content`：生成该章节前的前置内容
* `note`：额外的 wikification 指引信息

wikification 的输出结果会类似于：

```
戸山香澄

- 香澄的身份（Kasumi's identity） -

戸山香澄是Poppin'Party乐队的主唱兼吉他手，以其充满活力和感染力的性格著称。她极度重视团队的凝聚力和共同体验，经常在关键时刻强调“大家在一起”的重要性。无论是面对新的挑战、突发状况，还是团队成员之间的讨论，香澄总是以充满情感和表现力的方式回应，经常用夸张或感叹的语气表达自己的惊喜、兴奋或期待。

在团队中，香澄不仅是气氛的带动者，也是情感的纽带。她喜欢用亲昵、热情的语言表达对市谷有咲、牛込里美、花园多惠和山吹沙绫等成员的关心和依赖，尤其在情感高涨或低落的时刻，主动寻求或给予安慰和支持。当团队士气低落或成员对自身价值产生怀疑时，香澄会主动重申每个人在团队中的重要性，并用积极、鼓励的话语强化大家的归属感。

香澄在面对外界反馈或批评时，常常表现出明显的情绪反应，并倾向于寻求队友的肯定和支持。她也会在团队讨论各自角色或贡献时，强调团队的独特性和共同目标，时常用热情洋溢的语言重申Poppin'Party的身份和理想。无论是团队目标受到质疑，还是成员表达不安，香澄都会用坚定和充满活力的态度，带领大家回归初心，强化团队的凝聚力和共同信念。

总的来说，戸山香澄是Poppin'Party不可或缺的核心人物，她以积极、热情和富有感染力的个性，持续影响并维系着团队的团结与共同成长。

- 香澄的性格（Kasumi's Personality） -

戸山香澄以情感外露、积极乐观的性格著称。她在面对各种情绪时，总是毫不掩饰地表达自己的感受，无论是兴奋、惊喜，还是困惑和不安。香澄极度重视团队的凝聚力，喜欢通过主动寻求团队成员的参与和共鸣，营造“大家在一起”的氛围。她在遇到困难或挑战时，常常以坚定的态度重新振作，并积极提出替代方案或新点子，带动团队士气。

香澄在团队互动中，善于用夸张、幽默或戏谑的方式表达自己，尤其在与市谷有咲、牛込里美、花园多惠和山吹沙绫等亲密伙伴相处时，常常展现出俏皮、亲昵的一面。当团队成员感到不安或自我怀疑时，香澄会主动给予鼓励和支持，强调每个人在Poppin'Party中的独特价值，并用热情洋溢的话语强化团队的归属感和共同目标。

面对外界的反馈、批评或玩笑，香澄通常以轻松幽默的态度回应，有时还会用夸张的表情或言语化解尴尬，进一步拉近与队友的距离。她善于通过自发、显著的情感反应影响团队氛围，无论是用戏剧化的表现吸引注意，还是用积极的行动带动大家前进。

总的来说，戸山香澄是团队中不可或缺的情感核心，她以真挚、热情和富有感染力的个性，持续激励并团结着Poppin'Party的每一位成员。

- 香澄的能力（Kasumi's Ability） -
...
- 香澄的人际关系（Kasumi's Relationship） -
...
- 戸山香澄与牛込里美的互动（Kasumi's interaction with Rimi） -
...
- 戸山香澄与花园多惠的互动（Kasumi's interaction with Tae） -
...
- 戸山香澄与山吹沙绫的互动（Kasumi's interaction with Saaya） -
...
- 戸山香澄与市谷有咲的互动（Kasumi's interaction with Arisa） -
...
```

完整的 wikification 结果可在此处查看：
[profiles/%E6%88%B8%E5%B1%B1%E9%A6%99%E6%BE%84.wikified.profile.txt](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/profiles/%E6%88%B8%E5%B1%B1%E9%A6%99%E6%BE%84.wikified.profile.txt)

## 基准评测结果

<img width="1024" height="448" alt="image" src="https://github.com/user-attachments/assets/c16bcce1-9645-4981-bb66-d758bc5ab0a1" />

<img width="2560" height="1088" alt="image" src="https://github.com/user-attachments/assets/72e6d8f0-c231-4034-978f-74e8fa316f7d" />

## 引用

```bibtex
@article{codified_profile,
  title={Codifying Character Logic in Role-Playing},
  author={Letian Peng, Kun Zhou, Longfei Yun, Yupeng Hou, and Jingbo Shang},
  year={2026}
}
```
