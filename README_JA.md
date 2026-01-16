# Codified Decision Tree (CDT)

**与えられたストーリーラインから、深く検証可能で構造化されたキャラクター行動を抽出するアルゴリズム。**

<img width="1451" height="774" alt="image" src="https://github.com/user-attachments/assets/d12431f8-c91f-4551-92ee-213fc75e97c6" />

<img height="96" alt="KomeijiForce_Logo" src="https://github.com/user-attachments/assets/3b931cd1-8ce9-4e89-8852-f20d288cad1d" /> - 幻想を現に

このリポジトリには次のものが含まれます：

* アルゴリズム実装：キャラクターの scene–action ペアに基づいて **Codified Decision Tree（CDT）** を構築するコード
* ベンチマーク：CDT 駆動のロールプレイの性能評価
* 自動プロファイリング：CDT を読みやすい wiki 形式のテキストに変換するスクリプト

## 使い方

* **初期化**

まずプロジェクトのルートディレクトリに `constant.py` を作り、OpenAI と HuggingFace のトークンを記述します：

```python
openai_key = "..."
hf_token = "..."
```

環境は以下のような構成を推奨します：

```
torch: 2.7.1+cu126
transformers: 4.55.0
sentence_transformers: 5.1.0
sklearn: 1.7.1
openai: 2.14.0
```

* **CDT の構築**

  <img width="1842" height="862" alt="main_fig_v2_cropped_cropped-1" src="https://github.com/user-attachments/assets/b686ce21-5b92-4987-9374-8197223e84bb" />

論文の実験で扱っているキャラクターについては、`build_cdt.sh` スクリプトを使うことで同じ CDT を再現できます：

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

あるいは、`codified_decision_tree.py` 内の `CDT_Node` クラスを直接使って、任意のキャラクターの CDT を構築することもできます：

```python
CDT_Node(character, goal_topic, pairs, built_statements, depth, established_statements, gate_path,
max_depth, threshold_accept, threshold_reject, threshold_filter)
```

各パラメータの意味は次のとおりです：

* `character`：対象となるキャラクター名

* `goal_topic`：この CDT でフォーカスしたい目標（トピック／側面）

* `pairs`：CDT の学習データ。形式は `[{"scene": "...", "action": "..."}, ...]` のリストで、`character` がそれぞれの `scene` 内で `action` を取ります

* `built_statements`：ノード成長用の内部状態。`None` のままにしておきます

* `depth`：深さに基づく停止条件に使う内部状態。`1` のままにしておきます

* `established_statements`：多様性（diversification）のために使用。`[]` のままにしておきます

* `gate_path`：多様性（diversification）のために使用。`[]` のままにしておきます

* `depth`：深さに基づく停止条件。推奨値は `3`

* `threshold_accept`：ステートメント（statement）を「採択」する際の精度を制御する閾値

* `threshold_reject`：仮説（hypothesis）を「棄却」する際の精度を制御する閾値

* `threshold_filter`：ゲート（gate）を「通過」させるかどうかを決めるフィルタ強度の閾値

* `device_id`：このアルゴリズムを実行する GPU の ID

* **Grounding（根拠取得）**

`CDT_Node`（例：`cdt_tree`）を構築したら、`cdt_tree.traverse(scene)` を呼び出すことで、入力した `scene` に対応する CDT 上の“根拠となるステートメント（grounding statements）”を取得できます。

* **CDT のベンチマーク**

ベンチマークは `run_benchmark.sh` によって実行され、次の 2 つのベンチマークデータセットを対象としています：
[Fine-grained Fandom Benchmark](https://huggingface.co/datasets/KomeijiForce/Fine_Grained_Fandom_Benchmark_Action_Sequences) と
[Bandori Conversational Benchmark](https://huggingface.co/datasets/KomeijiForce/Bandori_Conversational_Benchmark_Action_Sequences)。

これらのリンクには、16 本のストーリーラインにおける主要キャラクターの行動シーケンスが含まれており、`load_ar_pairs` 関数によって前処理され、訓練用とテスト用に分割されます。

```python
python run_benchmark.py \
  --character "Kasumi" \
  --method "cdt_package" \
  --engine "gpt-4.1" \
  --eval_engine "gpt-4.1" \
  --generator_path "meta-llama/Llama-3.1-8B-Instruct" \
  --device_id 1
```

* **Wikification（wiki 形式への変換）**

CDT を人間が読みやすい wiki 形式のキャラクタープロファイルに変換するためのサンプル Notebook を用意しています：
[Wikification.ipynb](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/Wikification.ipynb)

この Notebook は、以下のようなパラメータを入力として受け取ります（例）：

```
character = "戸山香澄"
cdt_id = "Kasumi"
lang = "Japanese"
content = f"{character}"
note = '''
# Notes
Kasumi -> 戸山香澄
Arisa -> 市ヶ谷有咲
Rimi -> 牛込りみ
Tae -> 花園たえ
Saaya -> 山吹沙綾
'''
```

* `character`：wiki ページで表示に使いたいキャラクター名
* `cdt_id`：CDT 構築時に使ったキャラクター ID
* `lang`：wiki ページを生成したい言語
* `content`：このセクションを書き始める前に存在する文章（前付けの内容）
* `note`：wikification のための補足的な指示・対応表など

wikification の結果は次のような形になります：

```
戸山香澄

- 香澄のアイデンティティ（Kasumi's identity） -

戸山香澄は、常にバンドや仲間との「一体感」や「みんなで過ごす時間」の大切さを強く意識しているキャラクターとして描かれる。グループの存在そのものが彼女の自己認識と深く結びついており、「みんなでいること」「一緒にやること」が香澄にとって自分らしさの核となっている。

驚きや新しい情報に対しては感情表現が豊かで、驚嘆や感嘆を交えたリアクションを取りやすく、その反応の大きさ自体が彼女の存在感や明るさを象徴している。また、ライブやイベントなど、グループでの活動を前にすると、期待や高揚感を素直に言葉にし、場の空気を盛り上げる「ムードメーカー」としての側面が強い。

仲間との距離感においては、特に感情が高ぶった場面で、友人たちに対する愛情や親しみをストレートに示す傾向があり、言葉や態度を通して「一緒にいたい」「大好き」といった気持ちを表現する。とりわけ、市ヶ谷有咲・牛込りみ・花園たえ・山吹沙綾といったメンバーに対しては、冗談めかしたり甘えたりしながらも、強い絆と依存にも近い安心感をにじませる。

一方で、自分の演奏や役割に対するフィードバックや批判を受けた際には、感情が表に出やすく、不安や戸惑いを見せることも多い。そのようなとき、香澄は周囲に肯定や励ましを求め、仲間からの「大丈夫」「一緒にやろう」という言葉によって、自分の価値や立ち位置を再確認していく。これは、彼女の自己評価が「自分ひとり」ではなく「みんなとの関係性」によって支えられていることを示している。

グループ内で個々の役割や貢献が話題になったとき、香澄はしばしば「みんなでひとつ」「全員が大事」といった形で、バンドの一体性やメンバーそれぞれの存在意義を言葉にする。ときに茶化すような、あるいは感情のこもった言い回しで特定のメンバーとの絆を強調しながら、自分自身も「みんなを引っ張る存在」「始まりを作った人」としてのアイデンティティを再確認する。

バンドの結成理由や、グループとしての特別さが語られる場面では、香澄は自分たちの「らしさ」や共有してきた経験を誇らしげに語り、グループの名前や象徴、共通の思い出を引き合いに出して、「ここが自分の居場所だ」という感覚を強く打ち出す。バンドの目標や伝統、存在意義が揺らいだときには、理屈よりも感情と勢いで「私たちはこうありたい」「これが私たちだよ」と再宣言し、グループの方向性と結束を取り戻そうとする。

このように戸山香澄のアイデンティティは、「バンドの一員である自分」「みんなをつなぐ自分」という自己像と不可分であり、仲間との絆・共有体験・グループの名前や象徴を通じて、自分がここにいる意味を何度も言葉にして確かめ続ける姿が特徴となっている。

- 香澄の性格（Kasumi's personality） -

戸山香澄は、感情表現が非常にストレートで、嬉しさ・楽しさ・驚き・不安といった心の動きをそのまま言葉や態度に乗せて表すタイプである。驚いたときや新しいことを知ったときには、大きなリアクションや感嘆を交えた反応を見せ、その場の空気を一気に明るくする存在として描かれる。

常に「みんなでやること」「一緒にいること」を大切にしており、物事に取り組む際も自然とグループでの関わり方を選びやすい。バンド活動やイベント、遊びの計画など、仲間と共有できる目標や予定があるときには、目を輝かせて賛同し、誰よりもわかりやすくテンションを上げて場を盛り上げるムードメーカー的な性格が強い。

困難やトラブルに直面したときには、落ち込んだ気持ちを隠さずに口にしつつも、「じゃあこうしてみよう！」と代わりの案を出したり、「もう一回やってみよう」と前向きな姿勢を取り戻そうとする傾向がある。特に、自分の演奏や行動に対する指摘・からかい・批判を受けた場面では、最初に戸惑いや不安を見せながらも、最終的には冗談を交えた返しや軽いノリで受け止め、空気を重くしないように振る舞うことが多い。

また、仲間が不安になっているときや落ち込んでいるときには、香澄自身の明るさと勢いで励まそうとする。大げさな言い回しやちょっとしたおどけた態度を交えながら、「一緒にやろう」「大丈夫だよ」と背中を押し、相手の気持ちを軽くしようとする姿がよく見られる。特に、市ヶ谷有咲・牛込りみ・花園たえ・山吹沙綾といった近しいメンバーに対しては、甘えや冗談を織り交ぜたスキンシップや言動が多く、親しみと信頼を前面に出した関わり方をする。

グループ内の雰囲気や関係性が変化したときには、その変化に敏感に反応し、驚きや戸惑いを大きく表現しながらも、場の空気を変えるために新しい提案をしたり、笑いを誘うような言動で空気を和らげようとする。こうした自発的で目立つリアクションや、感情に正直な振る舞いは、香澄の明るさと存在感を象徴する要素となっている。

- 香澄の能力（Kasumi's ability） -
...
- 香澄の対人関係（Kasumi's relationship） -
...
- 牛込りみとの関わり（Kasumi's interaction with Rimi） -
...
- 香澄の花園たえとの関わり（Kasumi's interaction with Tae） -
...
- 香澄の山吹沙綾との関わり（Kasumi's interaction with Saaya） -
...
- 香澄の市ヶ谷有咲との関わり（Kasumi's interaction with Arisa） -
...
```

完全な wikification 結果は[ここ](https://github.com/KomeijiForce/Codified_Decision_Tree/blob/main/profiles/%E6%88%B8%E5%B1%B1%E9%A6%99%E6%BE%84.wikified.profile.2.txt)で確認できます：

## ベンチマーク結果

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
