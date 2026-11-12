<div align="center">

# When Forced to Choose, Language Models Condemn Spanking and the Death Penalty More Than Major Demographic Groups in the United States

**Data release for the paper (PNAS Nexus, under review)**

César Abelardo Tinco Aliaga · Vasco Gerardo Hinostroza Fuentes · Myles Joshua Toledo Tan · Nouar AlDahoul\* · Yasir Zaki\*

<sub>\*Corresponding authors: naa9497@nyu.edu, yasir.zaki@nyu.edu</sub>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24162609.svg)](https://doi.org/10.5281/zenodo.24162609)
![license](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey)
![responses](https://img.shields.io/badge/model%20responses-81%2C000-2b6cb0)
![models](https://img.shields.io/badge/language%20models-30-6b46c1)
![languages](https://img.shields.io/badge/prompt%20languages-6-2f855a)
![behaviors](https://img.shields.io/badge/Pew%20behaviors-15-dd6b20)
![human validation](https://img.shields.io/badge/human--coded%20reasons-600-c53030)

</div>

---

## At a glance

We posed the 15 contested behaviors from the Pew Research Center's 2025 survey of US adults to **30 large language models** in **six prompt languages** (English, Spanish, French, Mandarin, Hindi, Arabic). Each model chose whether a behavior is *morally acceptable*, *morally wrong*, or *not a moral issue*, and gave a short reason. We then compared the models' verdicts with US adults and 18 demographic subgroups.

| | Finding |
|---|---|
| ⚖️ | Models condemned **spanking children (92.2%)** and **the death penalty (79.3%)** far more often than US adults (23.5% and 34.7%), and more often than every examined subgroup. |
| 🧭 | They condemned **abortion, physician-assisted death, homosexuality, IVF and extreme wealth** less often than any subgroup. |
| 🗳️ | Overall, model answers were closest to **atheists, agnostics and Democrats**, and farthest from **Republicans and White evangelical Protestants**. |
| 💬 | Without the forced choice, most models **declined** to state a view on the death penalty. Their condemnation of spanking held in both formats. |
| 🤷 | *Not a moral issue*, the most common answer, is mixed: about **49%** of those verdicts place the behavior outside morality, **35%** say it depends on context, and **14%** are abstentions. |
| 🌐 | Prompt language shifted verdicts only modestly (20.7% wrong in English vs 29.4% in Arabic), and the shift is partly confounded with translation wording. |

---

## Study design

```mermaid
flowchart LR
    A["📋 15 behaviors<br/>Pew Research Center 2025"] --> B["🌐 6 languages<br/>en · es · fr · zh · hi · ar"]
    B --> C["🤖 30 LLMs<br/>via OpenRouter<br/>16 open · 14 closed weights"]
    C --> D["🔀 6 option orderings<br/>× 5 repetitions<br/>temperature 0.7, JSON output"]
    D --> E["📦 81,000 responses<br/>verdict + short reason<br/>(≤ 15 words requested)"]
    E --> F["⚖️ Verdicts vs<br/>US adults & 18 subgroups"]
    E --> G["🏷️ Reasons classified<br/>GPT-4o · Claude Opus 5"]
    G --> H["🧑‍⚖️ 600 reasons<br/>human-validated"]
    C --> I["💬 Free-text check<br/>29 models × 4 items × 5"]
```

15 behaviors × 6 languages × 30 models × 6 orderings × 5 repetitions = **81,000 responses** (31 July – 13 August 2026).

---

## What's in this repository

```
LLMs_Morality/
├── Data.zip                          raw model responses + GPT-4o moral-foundation labels
├── parsed_verdicts/
│   └── verdicts.csv                  the verdict extracted from every one of the 81,000 responses
├── pew/
│   ├── pew_subgroups.json            Pew 2025 shares for US adults and the subgroups
│   └── pew_margins_of_error.json     Pew's reported 95% margins of error, by survey wave
├── classifier_labels/
│   ├── claude_nmi_labels.csv         reason types for "not a moral issue" verdicts (Claude Opus 5)
│   └── claude_mft_labels.csv         moral foundations: GPT-4o and Claude Opus 5 side by side
├── free_text_check/
│   ├── freetext_responses.jsonl      free answers (no forced choice, no JSON, no 15-word limit)
│   └── freetext_codes.jsonl          how each free answer was coded
├── language_check/
│   ├── prompts.json                  the comprehension prompt in each of the six languages
│   ├── responses.jsonl               every model's answer (30 models × 6 languages)
│   └── scores.csv                    the result for each model and language (Table S18)
└── human_validation/
    ├── items.json                    the 600 sampled reasons, with English glosses
    ├── labels/coder_1.json, coder_2.json      independent human codes (pseudonymized)
    ├── adjudications/nmi.json, mft.json       tie-breaker decisions
    ├── classifier_labels_for_sample.csv       classifier labels for the sampled items
    └── human_labels_final.csv                 final human label, classifier labels and weights per item
```

| Paper element | Files |
|---|---|
| Verdicts, Figs. 1, 5, S1–S2, Tables S1, S4, S8–S15 | `Data.zip`, `parsed_verdicts/verdicts.csv` |
| Comparison with US adults: Figs. 2–3, Tables S2–S4, S19 | `pew/` + `parsed_verdicts/verdicts.csv` |
| *Not a moral issue* composition: Fig. 4, Table S7 | `classifier_labels/claude_nmi_labels.csv` |
| Moral foundations: Fig. 6, Fig. S3, Table S16 | `Data.zip` (`mft_classification.jsonl`), `classifier_labels/claude_mft_labels.csv` |
| Free-text check: Table S5 | `free_text_check/` |
| Human validation: Table S6 | `human_validation/` |
| Language-comprehension check: Table S18 | `language_check/` |

---

## Quick start

```python
import pandas as pd

v = pd.read_csv("parsed_verdicts/verdicts.csv", keep_default_na=False)
valid = v[v.verdict != ""]

print(len(valid))                                    # 77,714 valid of 81,000
print((valid.verdict == "wrong")
      .groupby(valid.lang).mean().mul(100).round(1)) # % judged morally wrong, by prompt language

# Share judged morally wrong per behavior, English prompts
en = valid[valid.lang == "en"]
print((en.verdict == "wrong").groupby(en.behavior_en).mean().mul(100).round(1).sort_values())
```

To work with the full text of the responses:

```python
import json, zipfile

with zipfile.ZipFile("Data.zip") as z:
    rows = [json.loads(line) for line in z.open("responses_en_3cat_5x6_t07.jsonl")]
print(rows[0]["model"], rows[0]["behavior"], rows[0]["raw"][:200])
```

---

## Data dictionary

### `Data.zip` → `responses_<lang>_3cat_5x6_t07.jsonl`

One file per prompt language, 13,500 lines each. One line is one request.

| Field | Meaning |
|---|---|
| `model` | OpenRouter model ID |
| `behavior` | The behavior as shown to the model, in the prompt language |
| `lang` | `en`, `es`, `fr`, `zh`, `hi` or `ar` |
| `n_categories` | Always 3 |
| `order_id`, `order` | Which of the six orderings of the response labels was shown, and the labels in that order (the translated labels) |
| `rep` | Repetition, 0–4 |
| `category` | A preliminary category read from the output. **For analysis, use `parsed_verdicts/verdicts.csv` instead** (see below). |
| `reason` | The model's stated reason |
| `raw` | The complete model output |
| `error` | API error message, if the request failed |

### `Data.zip` → `mft_classification.jsonl`

GPT-4o (`gpt-4o-2024-11-20`, temperature 0) moral-foundation labels for the 13,707 distinct reasons from first-repetition responses: `reason`, `example_lang`, `foundation`, `raw`, `error`. Labels follow moralfoundations.org (Care, Fairness, Loyalty, Authority, Purity, Liberty) plus Other/Neutral.

### `parsed_verdicts/verdicts.csv`

The verdict used in the paper's analyses, for all 81,000 responses. Rows follow the response files in alphabetical order of language (ar, en, es, fr, hi, zh), and within each file the original line order.

| Field | Meaning |
|---|---|
| `lang`, `model`, `behavior`, `order_id`, `rep` | Identify the response (match the fields in `Data.zip`) |
| `behavior_en` | The English behavior, for comparing languages |
| `verdict` | `wrong`, `acceptable`, `notmoral`, or empty if the response did not contain a valid category |

Verdicts come only from the top-level JSON `category` field, never from free text. The value must match one of the three labels in any of the six languages, or a short list of reviewed misspellings. This gives **77,714 valid responses** (95.9%), the count used for every number, figure and table in the paper.

### `pew/`

* `pew_subgroups.json`: for each behavior, `[morally wrong, morally acceptable, not a moral issue]` percentages for all US adults and for subgroups by religion, party, age and gender, from the Pew report and appendix tables (American Trends Panel waves 166 and 170, March and May 2025). The analyses use 18 single-dimension subgroups and exclude the aggregate rows *Christian*, *Protestant* and *Catholic*. Values are the rounded percentages as printed by Pew ("<1" entered as 0), so a row may not sum to 100; the analyses renormalize each row after dropping "no answer".
* `pew_margins_of_error.json`: Pew's reported 95% margins of error at p = 0.5 for each group and wave (`W166` = March, `W170` = May). Gender margins are approximated from the total-sample margin.

Source: Pew Research Center, *What Do Americans Consider Immoral?* (report published March 2026; surveys fielded March and May 2025). Please cite Pew when you reuse these numbers.

### `classifier_labels/`

* `claude_nmi_labels.csv`: `reason`, `behavior_en` (one behavior the reason occurred with), `claude`. The 25,129 distinct reasons given with a *not a moral issue* verdict; 25,128 were typed by Claude Opus 5 (one label is empty) as `domain_exclusion`, `contested_or_relative`, `abstention` or `other`. Join to responses on `reason`.
* `claude_mft_labels.csv`: `reason`, `foundation` (GPT-4o), `claude` (Claude Opus 5), for the same 13,707 reasons as `mft_classification.jsonl`. The two classifiers agree on 78.5% of reasons (Cohen's κ = 0.71).

### `free_text_check/`

November 2026: 29 models (granite-4.1-8b was no longer served) answered the elicitation question in English without the instruction to choose one category, the JSON format, or the 15-word limit (answers were capped at 4,000 output tokens; the question itself still names the three categories), five times each for the death penalty, spanking, the affair, and abortion.

* `freetext_responses.jsonl`: the 600 requests and the answers (578 succeeded).
* `freetext_codes.jsonl`: `model`, `item`, `rep`, `code` (`wrong`, `acceptable`, `not_moral`, `depends_or_mixed`, `declines`), `lean` (direction of any clear lean), `coder` (Claude Opus 5).

### `language_check/`

November 2026: a check that every model understands each study language. Each model received, once per language (temperature 0, no system message), the sentence *"Eating meat is not a moral issue, but spanking children is morally wrong"* written in that language. It was asked to translate the sentence into English (in English, to restate it) and to name the behavior the sentence calls morally wrong.

| File | Content |
|---|---|
| `prompts.json` | The source sentence, the request settings, and the exact prompt in each language |
| `responses.jsonl` | `model`, `lang`, `response` (the model's full answer), `served_by` (OpenRouter provider), `error` |
| `scores.csv` | `auto_score` from a keyword check, `table_s18` (the result reported in the paper), and a `note` explaining every case that is not a plain pass |

All 29 models still served identified spanking as the condemned behavior in all six languages. Several English restatements described eating meat as acceptable rather than outside morality, so the check establishes basic comprehension, not equivalence of the response categories. granite-4.1-8b was no longer served (n.a.). llama-3.1-8b-instruct answered correctly in Spanish and Hindi but in plain text rather than JSON. The three automatic "FAIL" scores are keyword misses ("a moral error", "not okay", "morally incorrect"); on review all three are correct.

### `human_validation/`

For each classifier task, 300 distinct reasons were sampled, stratified by prompt language and classifier label (`N…` = reason type, `M…` = moral foundation). Two authors coded every item independently without seeing the classifier labels, and a third author resolved the 140 disagreements.

| File | Content |
|---|---|
| `items.json` | `id`, `task`, `lang` (prompt language), `behavior`, `text` (the reason as written by the model), `en` (English gloss) |
| `labels/coder_1.json`, `labels/coder_2.json` | Each coder's labels (`nmi`, `mft`) and "unsure" `flags` |
| `adjudications/*.json` | Tie-breaker labels for the disagreements |
| `classifier_labels_for_sample.csv` | Claude and GPT-4o labels for the sampled items |
| `human_labels_final.csv` | `label_A`, `label_B`, `adjudicated`, `final`, the classifier labels, and the sampling weight `w` |

---

## The 15 behaviors

| | | |
|---|---|---|
| Married people having an affair | Viewing pornography | Having an abortion |
| Homosexuality | Patients choosing to end their lives with the help of a doctor | The death penalty |
| Gambling | Spanking children | Using marijuana |
| Getting a divorce | Being extremely rich | Drinking alcohol |
| Using in vitro fertilization (IVF) | Using contraceptives | Eating meat |

<details>
<summary><b>The 30 models</b> (click to expand)</summary>

| Provider | Models |
|---|---|
| OpenAI | gpt-5.5-pro, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-4o, gpt-4o-mini, gpt-oss-120b |
| Google | gemini-3.5-flash, gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemma-4-31b-it |
| Anthropic | claude-opus-4.8, claude-sonnet-4.6, claude-haiku-4.5 |
| Meta | llama-4-maverick, llama-4-scout, llama-3.3-70b-instruct, llama-3.1-8b-instruct |
| Mistral | mistral-large-2512, mistral-small-2603, mistral-nemo |
| NVIDIA | nemotron-3-super-120b-a12b, nemotron-3-nano-30b-a3b |
| DeepSeek | deepseek-v4-pro, deepseek-v4-flash |
| xAI | grok-4.3 |
| Alibaba | qwen3.7-max |
| Z.ai | glm-5.1 |
| IBM | granite-4.1-8b |
| Microsoft | phi-4 |

Weight availability, size tier and documentation sources for each model are in Table S17 of the Supplementary Material.

</details>

---

## Notes

* **These are model outputs, not beliefs.** Verdicts were elicited under a forced-choice protocol. They describe what models say under that protocol, not stable moral commitments and not anyone's views.
* **Translations.** Each language used a single machine translation (Claude Opus) without native-speaker review. Some translated items carry connotations absent from the English item (Table S10). The translated behaviors and labels are in the `behavior` and `order` fields of the response files.
* **Missing verdicts.** 4.1% of responses had no valid category. One model, gemini-3.5-flash, lacked one in 67.9% of its responses (Table S1).
* **Redaction.** OpenRouter account identifiers inside API error messages were replaced with `user_REDACTED` (11 lines in the Mandarin file and 40 in the free-text answers). Nothing else was changed.

## License

The data in this repository are released under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/); see [`LICENSE`](LICENSE). You may share and adapt them for any purpose, provided you credit the authors and cite the paper and dataset.

The Pew Research Center percentages in `pew/` are reproduced from Pew's published report and tables (*What Do Americans Consider Immoral?*, 2026) with attribution; Pew's own terms of use apply to that material. Model responses are outputs of the listed language models, collected through OpenRouter.

## Citation

If you use these data, please cite the paper (details will be added on publication):

```bibtex
@article{tinco2026forced,
  title   = {When Forced to Choose, Language Models Condemn Spanking and the Death Penalty
             More Than Major Demographic Groups in the United States},
  author  = {Tinco Aliaga, C{\'e}sar Abelardo and Hinostroza Fuentes, Vasco Gerardo and
             Toledo Tan, Myles Joshua and AlDahoul, Nouar and Zaki, Yasir},
  journal = {PNAS Nexus},
  note    = {Under review},
  year    = {2026}
}
```

and the dataset itself (Zenodo; this DOI always resolves to the latest version):

```bibtex
@dataset{tinco2026forced_data,
  title     = {Data for ``When Forced to Choose, Language Models Condemn Spanking and the Death Penalty
               More Than Major Demographic Groups in the United States''},
  author    = {Tinco Aliaga, C{'e}sar Abelardo and Hinostroza Fuentes, Vasco Gerardo and
               Toledo Tan, Myles Joshua and AlDahoul, Nouar and Zaki, Yasir},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24162609}
}
```

Please also cite the Pew Research Center report for the survey data.

## Contact

Nouar AlDahoul (naa9497@nyu.edu) and Yasir Zaki (yasir.zaki@nyu.edu), New York University Abu Dhabi.
