import os
import openai
import re, json, jsonlines, pickle
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from collections import defaultdict, Counter
from constant import openai_key, hf_token
from datasets import load_dataset
from nltk import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from openai import OpenAI
from copy import deepcopy
import argparse

openai.api_key = openai_key
leave_bar = True
client = OpenAI(api_key=openai_key)

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CDT / RP profiling & wikification pipeline"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4.1",
        help="The LLM engine for profiling"
    )

    # ── Wiki / language settings ─────────────────────────────────────
    parser.add_argument(
        "--character_wiki_name",
        type=str,
        default="平泽唯",
        help="Target character name in wiki language"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="Chinese",
        help="Output language"
    )
    parser.add_argument(
        "--note",
        type=str,
        default="Yui -> 平泽唯; Ritsu -> 田井中律; Mio -> 秋山澪; Mugi -> 琴吹䌷; Azusa -> 中野梓",
        help="Name mapping or annotation note"
    )

    # ── Profiling targets ────────────────────────────────────────────
    parser.add_argument(
        "--profiling_character",
        type=str,
        default="Yui",
        help="Main character to profile"
    )
    parser.add_argument(
        "--main_characters",
        type=str,
        nargs="+",
        default=["Yui", "Ritsu", "Mio", "Mugi", "Azusa"],
        help="List of main characters"
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="+",
        default=["identity", "personality", "ability", "relationship"],
        help="List of main chapter topics"
    )

    # ── IO paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--storyline_path",
        type=str,
        default="kon.storyline.part.txt",
        help="Input storyline file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="profiles/Yui.wikified.profile.txt",
        help="Output wikified profile path"
    )

    # ── Device (string to allow multi-device formats) ────────────────
    parser.add_argument(
        "--device_id",
        type=str,
        default="1",
        help="Device id(s), e.g. '0', '1'"
    )

    return parser

args = build_arg_parser().parse_args()
engine = args.engine

# ── Wiki / language ──────────────────────────────────────────────
character_wiki_name = args.character_wiki_name
lang = args.lang
note = args.note

# ── Profiling ────────────────────────────────────────────────────
profiling_character = args.profiling_character
main_characters = args.main_characters
topics = args.topics

# ── IO ───────────────────────────────────────────────────────────
storyline_path = args.storyline_path
output_path = args.output_path

# ── Device ───────────────────────────────────────────────────────
device_id = args.device_id

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

def fill_in_instruction(paragraph, main_characters):
    return f'''{paragraph}

Parse the paragraph above into a series of actions taken by characters, the format should be

[action 1] (character 1)
[action 2] (character 2)
[action 3] (character 3) 
...

character should either be a main character {main_characters} or \'Other\' or \'Environment\''''

def parse_scene_to_actions(text):
    pattern = r'\[([^\[\]]+?)\] \((.*?)\)'
    results = []

    cursor = 0
    clean_text = text  # We'll clean the condition on the fly

    for match in re.finditer(pattern, text):
        start, end = match.span()
        action = match.group(1).strip()
        character = match.group(2).strip()
        
        results.append({
            'action': action,
            'character': character
        })

    return results

def generative_encode(texts):
    embedding = generator_embedding(**generator_tokenizer(texts, return_tensors="pt", padding=True).to(device), output_hidden_states=True).hidden_states[-1][:, -1]
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding = embedding.detach().cpu().numpy()
    return embedding

def surface_encode(texts):
    embedding = surface_embedding.encode(texts, convert_to_tensor=True)
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding = embedding.detach().cpu().numpy()
    return embedding

def select_cluster_centers(character, pairs, n_in_cluster_case, n_in_cluster_sample, n_max_cluster=8, bs=8):
    
    actions = [pair["action"] for pair in pairs]
    scenes = [pair["scene"] for pair in pairs]
    
    with torch.no_grad():
        document_embeddings = []
        for idx in tqdm(range(0, len(actions), bs), desc="Embedding...", leave=leave_bar):
            scenes_batch = [scene+f"\n\nThus, {character} decides to" for scene in scenes[idx:idx+bs]]
            actions_batch = actions[idx:idx+bs]
            document_embeddings.append(np.concatenate([generative_encode(scenes_batch), surface_encode(actions_batch)], -1))
    
        document_embeddings = np.concatenate(document_embeddings, 0)
    
    n_clusters = min(int(np.ceil(document_embeddings.shape[0]/n_in_cluster_case)), n_max_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    kmeans.fit(document_embeddings)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    clusters = []
    
    for centroid in centroids:
        distances = (((document_embeddings - centroid) ** 2).sum(-1)) ** 0.5
        cluster = []
        for idx in distances.argsort(-1)[:n_in_cluster_sample]:
            cluster.append(pairs[idx])
        clusters.append(cluster)

    return clusters

def check_scene(texts, questions):
    
    prompts = [f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown." for text, question in zip(texts, questions)]

    with torch.no_grad():
        logits = classifier(**classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(device)).logits
        choices = logits.argmax(-1)

    return [[False, None, True][choice.item()] for choice in choices]

def check_statement_probs(character, actions, statements):
    
    prompts = [f'''Character: {character}

Action: {action}

Statement: {statement}

Question: Does the statement provide correct grounding, which directly supports the character to take the action?

yes: the action involves direct information from the statement.

no: the action indicates the statement's assertion is not always correctly.

unknown: the action is irrelevant to the statement or the causal relationship cannot be determined

Directly answer only yes/no/unknown.''' for action, statement in zip(actions, statements)]

    with torch.no_grad():
        logits = classifier(**classifier_tokenizer(prompts, return_tensors="pt", padding=True).to(device)).logits
        probs = logits.softmax(-1).sum(0).detach().cpu().numpy()

    return probs

def make_hypothesis(cluster, character, goal_topic, established_statements, gate_path, k=3):

    action_scene_context = "\n\n".join(["# Scene:\n"+pair["scene"]+"\n# Action:\n"+pair["action"] for pair in cluster])
    established_statement_verbalized = "\n".join(established_statements) if len(established_statements) > 0 else "N/A"
    gate_path_verbalized = "\n".join(gate_path) if len(gate_path) > 0 else "N/A"
    
    prompt = f'''# Scene-Action Pairs
{action_scene_context}

# Established Statements
{established_statement_verbalized}

# Already Proposed Common Points
{gate_path_verbalized}

# Task

Your task is to build the grounding logic for an AI system to understand the behavior of {character} (Current topic: "{goal_topic}"), assert the AI system has no prior knowledge of {character}.
To do this, please propose hypotheses for the general behavior logic of {character} based on the given action-scene pairs, complete the task step by step:

1. What's the main feature of {character}'s behavior  (Focus on the current topic: "{goal_topic}") shown in the given scene-action pairs, **other than the already established statements**?

2. Summarize {k} potential common points (grounding statements) of the actions taken by {character} in the given scenes about the focused topic: "{goal_topic}", **which is other than the already established statements**.
- The grounding statements should be general, avoiding too specific action descriptions. (except when it's a common skill of the character)
- Consider the grounding statements in a general way.
- The grounding statements should be concise, informative, and general sentences.
- Never be assertive! Always make objective description of the character rather than making assertive causal relations.

3. Summarize {k} potential common points of the given scenes that trigger each behavior, **which should be different from already proposed common points.**
- The question should be simple, not ambiguous, and specific to a subset of scenes rather than always applicable.
- Focus on the **next action** when asking! Don't ask whether certain event is involved, instead ask whether the scene might trigger potential behavior for {character}'s **next action**.
- Directly include "{character}'s next action" in the question!

4. Output the hypothesized scene-action triggers in the following format:
```python
action_hypotheses = [] # A list of syntactically complete statements (always mentioning {character})
scene_check_hypotheses = [] # A list of syntactically complete questions to check the given scene (always mentioning {character})
```
'''
    
    response = generate(prompt)
    
    code_str = re.findall("```python(.*?)```", response, re.DOTALL)[0].strip()
    local_vars = {}
    exec(code_str, globals(), local_vars)
    
    action_hypotheses = local_vars["action_hypotheses"]
    scene_check_hypotheses = local_vars["scene_check_hypotheses"]
    
    return action_hypotheses, scene_check_hypotheses

def validate_hypothesis(character, pairs, hypothesized_question, hypothesized_action):
    # When hypothesized_question is None, always check the statement

    res = defaultdict(int)
    filtered_pairs = []
    relevance_all = []

    bs = 64
    
    for idx in tqdm(range(0, len(pairs), bs), desc="Filtering Scenes...", leave=leave_bar):

        pairs_batch = pairs[idx:idx+bs]
        
        scenes, actions = [pair["scene"] for pair in pairs_batch], [pair["action"] for pair in pairs_batch]
        
        if hypothesized_question is None:
            relevance = [True for _ in pairs_batch]
        else:
            relevance = check_scene(scenes, [hypothesized_question for pair in pairs_batch])
        relevance_all.extend(relevance)

    for pair, rel in zip(pairs, relevance_all):
        if rel:
            filtered_pairs.append(pair)
        else:
            res["Irrelevant"] += 1.0

    for idx in tqdm(range(0, len(filtered_pairs), bs), desc="Validating Statements...", leave=leave_bar):

        pairs_batch = filtered_pairs[idx:idx+bs]
        
        scenes, actions = [pair["scene"] for pair in pairs_batch], [pair["action"] for pair in pairs_batch]
        
        score = check_statement_probs(character, actions, [hypothesized_action for pair in pairs_batch])

        res["False"] += score[0]
        res["None"] += score[1]
        res["True"] += score[2]

    return res, filtered_pairs

def summarize_triggers(character, gates, statement_candidates):

    paired_hypotheses = [{"scene_check_hypothesis": gate, "action_hypothesis": statement_candidate}
                            for gate, statement_candidate in zip(gates, statement_candidates)]
        
    if len(paired_hypotheses) > 8:
        
        boost_prompt = f"""
# Task: Summarize & Compress Scene–Action Hypothesis Pairs into Top 8

You are given a list of paired hypotheses. Each pair contains:
- "scene_check_hypothesis": a question about {character}'s next action
- "action_hypothesis": a general behavioral grounding statement about {character}

Input pairs:
{paired_hypotheses}

## Goal
Produce a rewritten, deduplicated, and compressed set of **exactly 8** pairs that capture the **most important** and **most general** behavioral grounding logic for {{character}}.

Rewriting is allowed and encouraged to increase:
- generality
- coverage across different subsets of scenes
- clarity
- non-assertiveness

## Selection Principles (prioritized)
1. **Coverage**: The 8 pairs should collectively cover the widest range of distinct behavioral patterns and distinct scene triggers.
2. **Centrality**: Prefer pairs that reflect recurring or core behaviors across many scene-action pairs.
3. **Specificity without overfitting**: Keep statements general; only keep a specific skill/ability if it appears repeatedly and broadly.
4. **Non-redundancy**: Each of the 8 pairs must represent a meaningfully different behavior/trigger from the others.
5. **Pair coherence**: The scene_check_hypothesis must plausibly test for the corresponding action_hypothesis (do not mismatch them).

## Dedup & Merge Rules
- You may merge multiple similar input pairs into one rewritten pair.
- If two candidate pairs overlap heavily in either the action or the scene question, combine them into a single more general pair.
- Do not preserve original wording when a clearer/general rewrite is possible.

## Constraints to Preserve
### scene_check_hypothesis
- Must be a **single, simple question**
- Must explicitly contain the exact phrase: "{character}'s next action"
- Must target **scene conditions** that could trigger that next action
- Must be applicable to a **subset** of scenes (not a universal always-true condition)

### action_hypothesis
- Must be a **single, concise sentence**
- Must be **non-assertive** (use “may”, “tends to”, “often appears to”, “is described as”, “is observed as”, etc.)
- Must not invent backstory or assume prior knowledge

## Output Format (JSON only)
Return exactly 8 pairs:

```json
{{
  "top8_pairs": [
    {{
      "scene_check_hypothesis": "...",
      "action_hypothesis": "..."
    }},
    {{
      "scene_check_hypothesis": "...",
      "action_hypothesis": "..."
    }},
    ...
  ]
}}
````

## Quality Checklist (must satisfy)

* Exactly 8 pairs.
* No two action_hypothesis items mean the same thing.
* No two scene_check_hypothesis questions ask the same trigger.
* Each scene_check_hypothesis clearly tests for {character}'s next action.
* Each action_hypothesis is general, grounded, and non-assertive.
  """

        response = generate(boost_prompt)
        paired_hypotheses = json.loads(re.findall("```json(.*?)```", response, re.DOTALL)[0].strip())["top8_pairs"]
        
    gates = [pair["scene_check_hypothesis"] for pair in paired_hypotheses]
    statement_candidates = [pair["action_hypothesis"] for pair in paired_hypotheses]

    return gates, statement_candidates

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

    def verbalize(self):
        text = ""
        statements = deepcopy(self.statements)
        text = text + "\n".join(statements) + "\n"
        for gate, child in zip(self.gates, self.children):
            text += f"if \"{gate}\" is \"True\":\n"
            text += "\n".join([f"\t{line}" for line in child.verbalize().split("\n")])+"\n"
        if text.strip() == "":
            text = "pass"
        return text

def wikify(character, content, cdt_tree_verbalized, attribute, note, lang="English"):

    prompt = f'''# Wiki:
{content}

# Pseudo code

{cdt_tree_verbalized}

# Task
The pseudo code above describes the character behavior logic of {character}. Summarize the information from the given code for "{attribute}" section in a wiki style to entail the given character Wiki.
{note}
Output in the following format (using {lang} and the name: {character}):
- {attribute} (The title should also be translated to target language: {lang}) -

{{Content}}'''

    section = generate(prompt)

    return section

raw_storyline = open(storyline_path).read()

surface_embedder_path, generator_embedder_path = "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-0.6B"
discriminator_path = "KomeijiForce/deberta-v3-base-rp-nli"

device = torch.device(f"cuda:{device_id}")

surface_tokenizer = AutoTokenizer.from_pretrained(surface_embedder_path)
generator_tokenizer = AutoTokenizer.from_pretrained(generator_embedder_path)
generator_tokenizer.padding_side = "left"

surface_embedding = SentenceTransformer(surface_embedder_path, device=device, model_kwargs={"torch_dtype": torch.float16})
generator_embedding = AutoModelForCausalLM.from_pretrained(generator_embedder_path, torch_dtype=torch.float16).to(device)

classifier_tokenizer = AutoTokenizer.from_pretrained(discriminator_path)
classifier = AutoModelForSequenceClassification.from_pretrained(discriminator_path)

classifier = classifier.to(device)

main_characters_jojo = ["Jotaro", "Polnareff", "Joseph", "DIO", "Kakyoin", "Avdol", "Iggy"]

icl_inputs = [
    "In the year 1987, 17-year-old delinquent Jotaro Kujo, grandson of Joseph Joestar, and son of Holy Kujo, is detained after beating up three armed men and a trained boxer. Despite Holy's insistence, he refuses to leave his cell, claiming that he is possessed by an evil spirit. His prison mates confirm, desperate to get out of the shared cell.",
    "The guards notice Jotaro somehow drinking a can of beer and reading Weekly Shōnen Jump while listening to the radio within his cell, confused as to how he retrieved all those items. Jotaro claims the evil spirit brings things to him.",
    "Near the Canary Islands in 1983, four years earlier, off the coast of Africa, a group of fishermen uncover an extremely heavy 100-year-old coffin from the water, believing it to be a large treasure chest.",
    "In the present, Joseph Joestar arrives in Japan and is greeted by Holy. Joseph brings along a mysterious partner as the three head to the prison. The prison guards stumble upon Jotaro\'s prison cell filled with a considerable amount of other various objects, from books to jackets on coat hangers to even a guitar and a computer. Joseph confronts him and orders him to come out so that they can go home. Jotaro refuses at first due to his \"evil spirit\" and tears the little finger off of Joseph\'s prosthetic hand. Joseph is left surprised, as he didn\'t even notice when Jotaro snatched the finger.",
    "Joseph reveals his partner to be a fortune teller by the name of Muhammad Avdol, who he met three years ago in Egypt. Avdol reveals a spirit of his own called Magician's Red, which is a humanoid with a bird-like head and the ability to produce and control flames. Magician's Red launches a projectile of flames, pinning Jotaro against the back wall of the jail cell. Holy worries about what her father is doing to her son, while the guards are confused as to what's going on as they don't see any flames. Having no other choice, Jotaro's spirit confronts Magician's Red, fully manifesting itself.",
]

icl_inputs = [fill_in_instruction(icl_input, main_characters_jojo) for icl_input in icl_inputs]

icl_outputs = [
    '''[Jotaro beats up three armed men and a trained boxer] (Jotaro)
[Jotaro is detained in prison] (Jotaro)
[Holy insists Jotaro leave his cell] (Other)
[Jotaro refuses to leave his cell] (Jotaro)
[Jotaro claims he is possessed by an evil spirit] (Jotaro)
[prison mates confirm Jotaro’s claim, desperate to get out of the shared cell] (Other)''',
    '''[guards notice Jotaro drinking a can of beer, reading Weekly Shōnen Jump, and listening to the radio within his cell] (Other)
[Jotaro drinks a can of beer] (Jotaro)
[Jotaro reads Weekly Shōnen Jump] (Jotaro)
[Jotaro listens to the radio] (Jotaro)
[Jotaro claims the evil spirit brings things to him] (Jotaro)''',
    '''[A group of fishermen uncover an extremely heavy 100-year-old coffin from the water near the Canary Islands in 1983] (Other)
[The fishermen believe the coffin to be a large treasure chest] (Other)''',
    '''[Joseph arrives in Japan and is greeted by Holy] (Joseph)
[Holy greets Joseph] (Other)
[Joseph brings along a mysterious partner] (Joseph)
[Joseph, Holy, and the partner head to the prison] (Joseph)
[prison guards stumble upon Jotaro’s cell filled with various objects such as books, jackets, a guitar, and a computer] (Other)
[Joseph confronts Jotaro and orders him to come out so they can go home] (Joseph)
[Jotaro refuses to leave due to his "evil spirit"] (Jotaro)
[Jotaro tears the little finger off Joseph’s prosthetic hand] (Jotaro)
[Joseph is left surprised, realizing he didn’t even notice Jotaro snatch the finger] (Joseph)''',
    '''[Joseph reveals his partner to be a fortune teller named Muhammad Avdol, whom he met three years ago in Egypt] (Joseph)
[Avdol reveals his spirit, Magician’s Red, a humanoid with a bird-like head and the ability to produce and control flames] (Avdol)
[Magician’s Red launches a projectile of flames, pinning Jotaro against the back wall of the jail cell] (Avdol)
[Holy worries about what her father is doing to her son] (Other)
[guards are confused as they don’t see any flames] (Other)
[Jotaro’s spirit confronts Magician’s Red, fully manifesting itself] (Jotaro)''',
]

icl_turns = []

for icl_input, icl_output in zip(icl_inputs, icl_outputs):
    icl_turns.append({"role": "assistant", "content": icl_input})
    icl_turns.append({"role": "user", "content": icl_output})

paragraphs = raw_storyline.split("\n\n")

paragraph_per_block = 3

blocks = ["\n\n".join(paragraphs[idx:idx+paragraph_per_block]) for idx in range(0, len(paragraphs), paragraph_per_block)]

action_series = []

for block in tqdm(blocks[:10], desc="Parsing into action series...", leave=leave_bar):
    prompt = fill_in_instruction(block, main_characters)
    messages=[*icl_turns, {"role": "user", "content": prompt}]
    scene = client.responses.create(
        model=engine,
        temperature=1e-8,
        input=messages,
    ).output_text
    actions = parse_scene_to_actions(scene)
    action_series.extend(actions)

all_actions = []
pairs = []
last_character = []
for item in tqdm(action_series, desc="Extracting target actions...", leave=leave_bar):
    all_actions.append(item["action"])
    if "character" in item:
        item["characters"] = [item["character"]]
    if profiling_character in item["characters"]:
        scene = "\n".join(all_actions[-1-10:-1])
        pairs.append({**item, "scene": scene, "last_character": last_character})
    last_character = item["characters"]

other_characters = [_character for _character in main_characters if _character != profiling_character]

topic2cdt = {}
rel_topic2cdt = {}

for attribute in topics:
    goal_topic = f"{profiling_character}'s {attribute}"
    cdt_tree = CDT_Node(profiling_character, goal_topic, pairs)
    topic2cdt[goal_topic] = cdt_tree

for other_character in other_characters:
    goal_topic = f"{profiling_character}'s interaction with {other_character}"
    relation_pairs = [data for data in pairs if other_character in data["last_character"]]
    
    if len(relation_pairs) >= 16:
        cdt_tree = CDT_Node(profiling_character, goal_topic, relation_pairs)
        rel_topic2cdt[goal_topic] = cdt_tree
    
cdts = {"topic2cdt": topic2cdt, "rel_topic2cdt": rel_topic2cdt}

content = f"{character_wiki_name}"
topic2cdt = cdts["topic2cdt"]
rel_topic2cdt = cdts["rel_topic2cdt"]

for topic in tqdm(topic2cdt, desc="Wikifying attribute information..."):
    cdt = topic2cdt[topic]
    increment = wikify(character_wiki_name, content, cdt.verbalize(), topic, note, lang)
    content = "\n\n".join([content, increment])

for rel_topic in tqdm(rel_topic2cdt, desc="Wikifying interactive information..."):
    cdt = rel_topic2cdt[rel_topic]
    increment = wikify(character_wiki_name, content, cdt.verbalize(), rel_topic, note, lang)
    content = "\n\n".join([content, increment])

with open(output_path, "w") as writer:
    writer.write(content)
