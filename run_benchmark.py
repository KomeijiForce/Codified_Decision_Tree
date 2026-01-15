import os
import openai
import json, jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import re
from collections import defaultdict
from constant import openai_key, hf_token
from datasets import load_dataset
from nltk import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
from tqdm.notebook import tqdm    
from sklearn.cluster import KMeans
from openai import OpenAI
from copy import deepcopy
import pickle

# character = 'Kasumi'
# method = "cdt_package"
# engine = "gpt-4.1"
# eval_engine = "gpt-4.1"
# openai.api_key = openai_key
# generator_path = "meta-llama/Llama-3.1-8B-Instruct"
# surface_embedder_path, generator_embedder_path = "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-8B"
# discriminator_path = "KomeijiForce/deberta-v3-base-rp-nli"
# device_id = 1

import argparse

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run CDT-based RP generation with configurable models"
    )

    # ── Core identity ────────────────────────────────────────────────
    parser.add_argument("--character", type=str, default="Kasumi")
    parser.add_argument("--method", type=str, default="cdt_package")

    # ── Engines ──────────────────────────────────────────────────────
    parser.add_argument("--engine", type=str, default="gpt-4.1")
    parser.add_argument("--eval_engine", type=str, default="gpt-4.1")

    # ── Generator / Embedding models ─────────────────────────────────
    parser.add_argument(
        "--generator_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--surface_embedder_path",
        type=str,
        default="Qwen/Qwen3-Embedding-8B"
    )
    parser.add_argument(
        "--generator_embedder_path",
        type=str,
        default="Qwen/Qwen3-8B"
    )

    # ── Discriminator ────────────────────────────────────────────────
    parser.add_argument(
        "--discriminator_path",
        type=str,
        default="KomeijiForce/deberta-v3-base-rp-nli"
    )

    # ── Device ───────────────────────────────────────────────────────
    parser.add_argument("--device_id", type=int, default=1)

    return parser

args = build_arg_parser().parse_args()
# ── Identity ─────────────────────────────────────────────────────
character = args.character
method = args.method

# ── Engines ──────────────────────────────────────────────────────
engine = args.engine
eval_engine = args.eval_engine

# ── Models ───────────────────────────────────────────────────────
generator_path = args.generator_path
surface_embedder_path = args.surface_embedder_path
generator_embedder_path = args.generator_embedder_path
discriminator_path = args.discriminator_path

# ── Device ───────────────────────────────────────────────────────
device_id = args.device_id

client = OpenAI(api_key=openai_key)

all_characters = json.load(open("all_characters.json"))
character2artifact = {character:artifact for artifact in all_characters for character in all_characters[artifact]["major"]}
band2members = json.load(open("band2members.json"))
leave_bar = True

tokenizer = AutoTokenizer.from_pretrained(generator_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(generator_path, token=hf_token)
device = torch.device(f"cuda:{device_id}")
model = model.to(device)

classifier_tokenizer = AutoTokenizer.from_pretrained(discriminator_path)
classifier = AutoModelForSequenceClassification.from_pretrained(discriminator_path)
classifier = classifier.to(device)

def load_ar_pairs(character):

    artifact = character2artifact[character]

    if not os.path.exists(f"data/title2action_series.{artifact}.json"):

        title2action_series = defaultdict(list)

        if artifact not in band2members:
            hf_path = "KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences"
        else:
            hf_path = "KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences"
        for data in load_dataset(hf_path)["train"]:
            if data["artifact"] == artifact:
                title2action_series[data["title"]].append(data)

        json.dump(title2action_series, open(f"data/title2action_series.{artifact}.json", "w"))

    else:
        title2action_series = json.load(open(f"data/title2action_series.{artifact}.json"))

    all_actions = []
    pairs = []
    last_character = []
    for title in title2action_series:
        action_series = title2action_series[title]
        for item in action_series:
            all_actions.append(item["action"])
            if "character" in item:
                item["characters"] = [item["character"]]
            if character in item["characters"]:
                scene = "\n".join(all_actions[-1-10:-1])
                pairs.append({**item, "scene": scene, "last_character": last_character})
            last_character = item["characters"]

    return {"train": pairs[:len(pairs)//2], "test": pairs[len(pairs)//2:]}

def generate(prompt, engine=engine):
        
    response = client.responses.create(
        model=engine,
        temperature=1e-8,
        input=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    ).output_text
    
    return response

def generate_llama(prompt):

    prompt = prompt+"\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.encode("\n")[-1]
        )

    generated_text = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    generated_text = generated_text.strip().split("\n")[0].rstrip("assistant")
    
    return generated_text

def check_scene(text, question):
    
    prompt = f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."

    with torch.no_grad():
        logits = classifier(**classifier_tokenizer(prompt, return_tensors="pt").to(device)).logits[0]
        choice = logits.argmax(-1).item()

    return [False, None, True][choice]

def evaluate(character, d, method):
    
    scene = d['condition']
    question = d['question']
    action = d['action']
    last_character = d['last_character']

    prompt = f'''# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence.'''

    if method == "vanilla":
        grounding = None
    elif method == "extract_then_aggregate":
        grounding = open(f"profiles/{character}.profile.eta.txt").read()
    elif method == "human_profile":
        grounding = open(f"profiles/{character}.profile.txt").read()
    if method == "cdt_package":
        statements = []
        with open(f"packages/{character}.cdt.v3.1.package.relation.pkl", "rb") as f:
            cdts = pickle.load(f)
        topic2cdt = cdts["topic2cdt"]
        rel_topic2cdt = cdts["rel_topic2cdt"]
        for topic in topic2cdt:
            cdt_tree = topic2cdt[topic]
            statements.append(f"# {topic}")
            statements.extend(cdt_tree.traverse(scene))
        for c in last_character:
            topic = f"{character}'s interaction with {c}"
            if topic in rel_topic2cdt:
                cdt_tree = rel_topic2cdt[topic]
                statements.append(f"# {topic}")
                statements.extend(cdt_tree.traverse(scene))
        grounding = "\n".join(statements)
    
    prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"
    
    prediction = generate_llama(prompt)

    score_instruction = f'''# Scene
{scene}

Your Response: {prediction}
Ground Truth: {action}

Score the action of {character} in the response based on the ground truth.
A: The ground truth entails the action of {character} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {character} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {character} in the response. (Following a contradicted character logic.)

Output in json: 
```json
{{
"reasoning": "...",
"score": "A/B/C"
}}
```'''
    
    print("-"*100)
    print(prediction)
    print("."*100)
    print(action)
    print("-"*100)

    score = generate(score_instruction, eval_engine)
    
    score = re.findall(r'"score": "(.*?)"', score, re.DOTALL)[0].strip()
    score = {"A": 100, "B": 50, "C": 0}[score]
    
    return score

def benchmark(character, method, return_list=False):
    
    scores = []

    trpb = load_ar_pairs(character)["test"]
    bar = tqdm(trpb)
    for d in bar:
        d = {"condition": d["scene"], "question": f"What'll be {character}'s next action in response to the current scene?", 
             "action": d["action"], "last_character": d["last_character"]}
        score = evaluate(character, d, method)
            
        scores.append(score)
        print(f"#{len(scores)} # NLI Score: {np.mean(scores):.2f}", )
        print("="*100)
        bar.set_description(f"Score={np.mean(scores):.4}")

    if return_list:
        return scores
    else:
        return np.mean(scores)

class CDT_Node:
    def __init__(self, character, goal_topic, pairs, built_statements=None, depth=1, established_statements=[], gate_path=[], max_depth=3, threshold_accept=0.8, threshold_reject=0.5, threshold_filter=0.8):
        self.statements = []
        self.gates = [] # 1 gate -> 1 child
        self.children = []
        self.depth = depth

        if built_statements is not None:
            assert(pairs is None)
            self.statements = built_statements
        elif len(pairs) <= 8 or self.depth > max_depth:
            pass
        else:
            clusters = select_cluster_centers(character, pairs, n_in_cluster_case=16, n_in_cluster_sample=8, n_max_cluster=8, bs=8)
            statement_candidates, gates = [], []
            for cluster in tqdm(clusters, desc="Making Hypotheses...", leave=leave_bar):
                statement_candidates_cluster, gates_cluster = make_hypothesis(cluster, character, goal_topic, established_statements+self.statements, gate_path)
                statement_candidates.extend(statement_candidates_cluster)
                gates.extend(gates_cluster)

            gates, statement_candidates = summarize_triggers(character, gates, statement_candidates)
            global_statements, gated_statements = [], []
            remained_gates = []

            for gate, statement_candidate in zip(gates, statement_candidates):
                res, _ = validate_hypothesis(character, pairs, None, statement_candidate)
                correctness = res["True"]/(res["True"]+res["False"]+1e-8)+1e-8
                if correctness >= threshold_accept:
                    global_statements.append(statement_candidate)
                else:
                    gated_statements.append(statement_candidate)
                    remained_gates.append(gate)

            self.statements.extend(global_statements)

            for gate, statement_candidate in zip(remained_gates, gated_statements):
                res, filtered_pairs = validate_hypothesis(character, pairs, gate, statement_candidate)
                correctness = res["True"]/(res["True"]+res["False"]+1e-8)+1e-8
                broadness = 1-res["Irrelevant"]/sum(res.values())
                if broadness <= threshold_filter:
                    if correctness <= threshold_reject:
                        continue
                    elif correctness >= threshold_accept:
                        self.gates.append(gate)
                        self.children.append(CDT_Node(character, goal_topic, None, built_statements=[statement_candidate],
                            depth=depth+1, established_statements=established_statements+self.statements,
                                                      gate_path=gate_path+[gate], max_depth=max_depth,
                                                      threshold_accept=threshold_accept, threshold_reject=threshold_reject, threshold_filter=threshold_filter))
                    else:
                        self.gates.append(gate)
                        self.children.append(CDT_Node(character, goal_topic, filtered_pairs,
                            depth=depth+1, established_statements=established_statements+self.statements,
                                                      gate_path=gate_path+[gate], max_depth=max_depth,
                                                      threshold_accept=threshold_accept, threshold_reject=threshold_reject, threshold_filter=threshold_filter))

    def traverse(self, scene):
        statements = deepcopy(self.statements)
        for gate, child in zip(self.gates, self.children):
            if check_scene(scene, gate):
                statements.extend(child.traverse(scene))
        return statements

benchmark(character, method)