# Codified Decision Tree (CDT)
**An algorithm to derive deep, validated, structured character behaviors from given storylines.**

<img width="1451" height="774" alt="image" src="https://github.com/user-attachments/assets/d12431f8-c91f-4551-92ee-213fc75e97c6" />

<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - Let there be fantasy

This repo includes:
- Algorithm implementation: Constructing **Codified Decision Trees** based on scene-action pairs of characters
- Benchmarking: The performance of CDT-driven Role-playing
- Automatic profiling: Conversion script from CDT to reader-friendly wiki texts.

## How to use?
- Initialization
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
- Construct CDTs
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


## Benchmark Results
<img width="2591" height="1063" alt="image" src="https://github.com/user-attachments/assets/eaa223e0-4e57-4bcf-a44f-62d652b04509" />


<img width="2617" height="1108" alt="image" src="https://github.com/user-attachments/assets/2e2c9953-cc19-4029-bb18-6abd0a3da18d" />

