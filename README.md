# Codified Decision Tree (CDT) [\<Link to Paper\>](https://arxiv.org/pdf/2601.10080)
**An algorithm to derive deep, validated, structured character behaviors from given storylines.**

<img width="1451" height="774" alt="image" src="https://github.com/user-attachments/assets/d12431f8-c91f-4551-92ee-213fc75e97c6" />

<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - Let there be fantasy

This repo includes:
- Algorithm implementation: Constructing **Codified Decision Trees** based on scene-action pairs of characters
- Benchmarking: The performance of CDT-driven Role-playing
- Automatic profiling: Conversion script from CDT to reader-friendly wiki texts.

[[中文版README]](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/README_ZH.md) [[日本語README]](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/README_JA.md)

## How to use?
- **Initialization**

You have to create a `constant.py` file at the root path, and put your openai and huggingface token there:
```python
openai_key = "..."
hf_token = "..."
```
You may also follow the configuration below:
```
torch: 2.7.1+cu126
transformers: 4.55.0
sentence_transformers: 5.1.0
sklearn: 1.7.1
openai: 2.14.0
```
- **Construct CDTs**
<img width="1842" height="862" alt="main_fig_v2_cropped_cropped-1" src="https://github.com/user-attachments/assets/b686ce21-5b92-4987-9374-8197223e84bb" />


For characters involved in the paper's experiments, you can use the `build_cdt.sh` script to reproduce the CDTs:
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

You can build CDT for any character using the `CDT_Node` class given in `codified_decision_tree.py`:
```python
CDT_Node(character, goal_topic, pairs, built_statements, depth, established_statements, gate_path,
max_depth, threshold_accept, threshold_reject, threshold_filter)
```

Parameters:
- `character`: The name for your character;
- `goal_topic`: The goal (topic/aspect) you want the CDT to focus on;
- `pairs`: The training data for your CDT, in the format: `[{"scene": "...", "action": "..."}, {"scene": "...", "action": "..."}, ...]` where `character` takes the `action` in the `scene`;
- `built_statements`: Used for node growth, keep it `None`;
- `depth`: Used for depth-based termination, keep it `1`;
- `established_statements`: Used for diversification, keep `[]`;
- `gate_path`: Used for diversification, keep `[]`;
- `depth`: Used for depth-based termination, recommended to be set to `3`;
- `threshold_accept`: The parameter controlling the precision for statement acceptance;
- `threshold_reject`: The parameter controlling the precision for hypothesis abolishment;
- `threshold_filter`: The parameter controlling the filtering effect for gate acceptance;
- `device_id`: The GPU id you want to run the algorithm on.

- **Grounding**

With a constructed `CDT_Node` (e.g., `cdt_tree`), use `cdt_tree.traverse(scene)` to fetch grounding statements on the CDT for the input `scene`.

- **Benchmark CDTs**
The benchmarking is run by `run_benchmark.sh` on the two benchmarks: [Fine-grained Fandom Benchmark](https://huggingface.co/datasets/KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences) and [Bandori Conversational Benchmark](https://huggingface.co/datasets/KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences). The links include the action sequences of main characters in 16 storylines, which are further processed and split by the `load_ar_pairs` function into training and test sets.

```python
python run_benchmark.py \
  --character "Kasumi" \
  --method "cdt_package" \
  --engine "gpt-4.1" \
  --eval_engine "gpt-4.1" \
  --generator_path "meta-llama/Llama-3.1-8B-Instruct" \
  --device_id 1
```

- **Wikification**

**[TODO]** An end-to-end CDT profiling pipeline will soon be released!

The example notebook to wikify CDTs into reader-friendly profiles is provided [here](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/Wikification.ipynb):

It takes the following parameters as input for wikification with example values:

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

- `character`: The character name you want to use in the wiki page,
- `cdt_id`: The character name used to build the CDT,
- `lang`: The language you want for the wiki page,
- `content`: The preceding content before writing this section,
- `note`: Other guidance for the wikification.

The wikification result would be like:
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

Full wikified content can be found [here](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/profiles/%E6%88%B8%E5%B1%B1%E9%A6%99%E6%BE%84.wikified.profile.txt)

## Benchmark Results
<img width="1024" height="448" alt="image" src="https://github.com/user-attachments/assets/c16bcce1-9645-4981-bb66-d758bc5ab0a1" />

<img width="2560" height="1088" alt="image" src="https://github.com/user-attachments/assets/72e6d8f0-c231-4034-978f-74e8fa316f7d" />


## Citation
```bibtex
@article{codified_profile,
  title={Codifying Character Logic in Role-Playing},
  author={Letian Peng, Kun Zhou, Longfei Yun, Yupeng Hou, and Jingbo Shang},
  journal={arXiv preprint arXiv:2601.10080},
  year={2026}
}
```
