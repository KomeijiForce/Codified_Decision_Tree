# Codified Decision Tree (CDT)
**An algorithm to derive deep, validated, structured character behaviors from given storylines.**

<img width="1451" height="774" alt="image" src="https://github.com/user-attachments/assets/d12431f8-c91f-4551-92ee-213fc75e97c6" />

<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - Let there be fantasy

This repo includes:
- Algorithm implementation: Constructing **Codified Decision Trees** based on scene-action pairs of characters
- Benchmarking: The performance of CDT-driven Role-playing
- Automatic profiling: Conversion script from CDT to reader-friendly wiki texts.

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

Adjustable parameters:
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

## Benchmark Results
<img width="2591" height="1063" alt="image" src="https://github.com/user-attachments/assets/eaa223e0-4e57-4bcf-a44f-62d652b04509" />


<img width="2617" height="1108" alt="image" src="https://github.com/user-attachments/assets/2e2c9953-cc19-4029-bb18-6abd0a3da18d" />

