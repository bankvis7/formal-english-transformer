# Informal → Formal English Transformer

A fine-tuned **T5-small** model that rewrites casual, informal English into polished, professional prose. This text style transfer system helps students, professionals, and non-native speakers convert everyday language into formal writing suitable for emails, reports, and documents.

**Team:** Korn Visaltanachoti (Bank), Pingpan Krutdumrongchai (Ping), Natcha Soranathavornkul (Baipor)

---

## Overview

This project addresses a common real-world problem: people often think in casual language but need to write formally. This Transformer-based system automatically converts informal sentences into professional tone while preserving meaning.

**Example transformations:**

| Informal | Formal |
|----------|--------|
| *"Gonna skip the meeting, can't deal with it rn."* | *"I will not be attending the meeting at this time."* |
| *"Tbh, I dunno if this plan is gonna work out."* | *"I am not sure if this plan is going to work out."* |
| *"Just got a sick new phone, it's lit."* | *"I have recently purchased a new phone."* |

The model uses Google's **T5 (Text-to-Text Transfer Transformer)** architecture, which treats every NLP task as a text-to-text problem. By prepending the task prefix `"transfer informal to formal:"` to each input, T5 learns to perform style transfer as a sequence-to-sequence generation task.

---

## Architecture

### T5-small Specifications

| Parameter | Value |
|-----------|-------|
| Model | `t5-small` (Google) |
| Parameters | 60 million |
| Architecture | Encoder-Decoder Transformer |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 per layer |
| Embedding dimension | 512 |
| Feed-forward dimension | 2048 |
| Vocabulary | 32,128 tokens (SentencePiece) |
| Max sequence length | 512 tokens |
| Pretraining corpus | C4 (750 GB) |
| License | Apache 2.0 |

### How It Works

1. **Encoder** processes the informal sentence with task prefix `"transfer informal to formal: gonna skip the meeting"`
2. **Multi-head self-attention** (8 heads) allows each token to attend to all other input tokens
3. **Encoder output** produces contextualized representations capturing full sentence meaning
4. **Decoder** generates formal text one token at a time using:
   - Masked self-attention (preventing look-ahead)
   - Cross-attention to encoder output (accessing source sentence)
   - Feed-forward networks
5. **Beam search** (4 beams) finds high-quality output sequences

**Key components:**
- **Tokenization:** SentencePiece subword tokenizer breaks text into meaningful pieces (vocabulary size 32,128)
- **Positional encoding:** Relative positional bias tells the model token order
- **Attention mechanism:** Scaled dot-product attention with Query, Key, Value projections
- **Residual connections:** Add & LayerNorm after each sub-layer prevents vanishing gradients

---

## Dataset

### Why We Built a Custom Dataset

Existing formality datasets were investigated but proved unsuitable:
- **GYAFC** (Grammarly's Yahoo Answers Formality Corpus): 100K+ pairs but requires license agreement, not publicly downloadable
- **portex/multilingual-formality-transfer**: No English data (9 languages: Portuguese, German, French, Italian, Spanish, Turkish, Norwegian, Czech, Hungarian)

We built a custom dataset from two sources:

### Source 1: BigSalmon2 Seed Data (91 pairs)
Public GitHub repository with informal/formal pairs from `data.txt`. Raw data was entirely lowercase with inconsistent formatting.

### Source 2: LLM-Generated Synthetic Data (~274 pairs)
Used **Groq API** with Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`) to generate additional pairs in batches of 10 until reaching ~500 total.

**Generation prompt guidelines:**
- Informal sentences sound like casual spoken/texted English
- Formal sentences convey exact same meaning in professional tone
- Keep under 30 words each
- Cover diverse topics: food, travel, work, technology, school
- No topic repetition across pairs

### Final Dataset Statistics

| Split | Examples |
|-------|----------|
| Training | 325 (90%) |
| Validation | 37 (10%) |
| **Total** | **365** (after dedup) |

### Example Samples

| Informal | Formal |
|----------|--------|
| Need to finish this project by Friday. | I have a deadline to complete this project by Friday. |
| I'm skipping school today, it's boring. | I have decided not to attend school today, as I find it unengaging. |
| Freedom of the press is important to holding accountable bad political officials. | The freedom of the press plays a crucial role in ensuring that political officials are held to account. |
| Spanish is an awesome language that is spoken in Mexico. | Spanish is an excellent language that is widely spoken in Mexico. |

### Data Preprocessing Pipeline

Three-step cleaning applied to both sources:

1. **Whitespace normalization:** Collapse tabs/newlines/multiple spaces → single space (regex)
2. **Truecasing:** Restore proper capitalization for names, cities, sentence starts using `truecase` library (BigSalmon2 data was all lowercase)
3. **Closing punctuation:** Add period if sentence doesn't end with `.`, `!`, or `?`

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Downloaded BigSalmon2 repo
│   └── processed/
│       ├── seed_pairs.csv          # 91 cleaned pairs from BigSalmon2
│       └── informal_formal.csv     # 365 total pairs (seed + synthetic)
├── LoadDataset.ipynb               # Step 1: Download & clean seed data
├── GenerateData.ipynb              # Step 2: Expand dataset via Groq API
└── TrainingProcessedData.ipynb     # Step 3: Tokenize, train, and evaluate
```

---

## Pipeline

### Step 1 — Load Dataset (`LoadDataset.ipynb`)

Downloads [BigSalmon2/InformalToFormalDataset](https://github.com/BigSalmon2/InformalToFormalDataset) from GitHub and extracts 91 pairs from `data.txt`. Each pair labeled as:
- `"Informal English:"` → informal sentence
- `"Translated into the style of Abraham Lincoln:"` → formal version

Applies preprocessing pipeline and outputs `data/processed/seed_pairs.csv`.

### Step 2 — Generate Data (`GenerateData.ipynb`)

Uses Groq API (Llama 4 Scout) to synthetically generate ~409 additional pairs:
- 41 batches of 10 pairs each
- 2-second delay between calls (free tier rate limits)
- JSON array response format
- Same preprocessing pipeline as seed data

Outputs `data/processed/informal_formal.csv`.

> **Requires Groq API key.** Get one free at https://console.groq.com/ and set `GROQ_API_KEY` in the notebook.

### Step 3 — Train (`TrainingProcessedData.ipynb`)

Combines both CSVs, drops duplicates → 365 unique pairs. Adds task prefix `"transfer informal to formal: "` to all inputs. Fine-tunes T5-small using Hugging Face `Trainer` API.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `t5-small` |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Loss function | Cross Entropy |
| Max token length | 128 |
| Decoding | Beam search (4 beams) |
| Train / Val split | 325 / 37 (90% / 10%) |
| Hardware | Google Colab (GPU) |
| Training time | ~5.75 seconds |
| Throughput | 282 samples/sec, 35.6 steps/sec |

---

## Results

### Validation Loss (5 Epochs)

| Epoch | Validation Loss |
|-------|-----------------|
| 1 | 2.3210 |
| 2 | 2.1737 |
| 3 | 2.1094 |
| 4 | 2.0731 |
| 5 | **2.0661** |

**Average training loss:** 2.2465

Validation loss decreased steadily without signs of overfitting. Small gap between training and validation loss indicates good generalization.

### Example Outputs

| Informal Input | Model Output | Quality |
|----------------|--------------|---------|
| Tbh, I dunno if this plan is gonna work out. | I am not sure if this plan is going to work out. | Good |
| We're way behind schedule and stuff's getting messy. | We are currently experiencing a significant delay in scheduling and the situation is experiencing considerable difficulty. | Good (verbose) |
| Just got a sick new phone, it's lit. | I have recently purchased a new phone that has been upgraded. | Meaning shift |
| Gonna skip the meeting, can't deal with it rn. | Gonna skip the meeting, and I am unable to deal with it. | Kept "Gonna" |
| That presentation was lowkey amazing, props to the team. | That presentation was a pleasure, and I highly recommend it to the team. | Meaning shift |
| She's super smart and totally crushed the interview. | She is extremely intelligent and has completely crushed the interview. | Kept "crushed" |

**What worked well:**
- Expands contractions (`we're` → `we are`, `can't` → `unable to`)
- Removes some slang (`Tbh` → omitted, `dunno` → `not sure`)
- Replaces `gonna` with `going to` (most cases)
- More formal vocabulary and sentence structure

**What needs improvement:**
- Occasionally keeps informal words (`Gonna`, `crushed`, `ASAP`)
- Sometimes changes meaning (`lit` → `upgraded`, `props to` → `recommend`)
- Can sound awkward or overly verbose
- Limited by small dataset (365 examples)

---

## Installation

```bash
pip install transformers[torch] datasets sentencepiece accelerate groq pandas truecase nltk
```

---

## Usage

After training, use the `formalize()` function:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained("./formal_transformer_checkpoints/<best-checkpoint>")

def formalize(text):
    input_text = "transfer informal to formal: " + text
    inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
print(formalize("Tbh, I dunno if this plan is gonna work out."))
# → "I am not sure if this plan is going to work out."
```

---

## Discussion

### What Worked Well

- **T5 text-to-text framework** made task definition simple via prefix
- **Pretrained on C4** (750 GB) gave strong English foundation
- **Fast training** (~6 seconds on Colab GPU)
- **Hugging Face Trainer API** simplified implementation
- Model learned core patterns: contraction expansion, slang removal, formal vocabulary

### Limitations

1. **Small dataset (365 examples)** — T5 paper used millions of examples during pretraining. A larger dataset would significantly improve results.

2. **No automatic evaluation metrics** — Only loss values and manual inspection. Should add BLEU, ROUGE, or formality classifiers.

3. **Incomplete slang coverage** — Training data didn't include enough examples for all informal expressions (`Gonna`, `ASAP`, `crushed` kept in output).

4. **Semantic drift** — Some outputs change meaning (`lit` → `upgraded`, `props` → `recommend`) rather than just changing tone.

5. **Sequence length limit (128 tokens)** — Works for single sentences but not paragraphs.

6. **Limited training (5 epochs)** — More epochs with learning rate scheduling could help, but risks overfitting on small dataset.

### Future Improvements

- Expand dataset to 10K+ high-quality pairs
- Add automatic evaluation metrics (BLEU, formality classifier)
- Try larger models (T5-base: 220M params, T5-large: 770M params)
- Implement back-translation for data augmentation
- Add curriculum learning (easy → hard examples)
- Use LoRA or adapter layers for parameter-efficient fine-tuning
- Support longer sequences for paragraph-level formalization

---

## Team Contributions

**Korn Visaltanachoti (Bank)** — Data Preprocessing  
Collected BigSalmon2 seed dataset, built preprocessing pipeline (whitespace normalization, truecasing, punctuation), wrote `LoadDataset.ipynb` and `GenerateData.ipynb`, generated synthetic pairs via Groq API.

**Pingpan Krutdumrongchai (Ping)** — Model Training  
Set up Hugging Face training pipeline, wrote `TrainingProcessedData.ipynb`, configured hyperparameters, implemented tokenization and label masking, ran training on Colab, implemented beam search inference.

**Natcha Soranathavornkul (Baipor)** — Research & Documentation  
Researched Transformer architectures, selected T5-small, studied T5 paper and documentation, wrote project report, created architecture diagrams.

---

## References

- BigSalmon. (n.d.). *InformalToFormalDataset*. GitHub. https://github.com/BigSalmon2/InformalToFormalDataset
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
- Rao, S., & Tetreault, J. (2018). Dear sir or madam, may I introduce the GYAFC dataset: Corpus, benchmarks and metrics for formality style transfer. *Proceedings of NAACL-HLT 2018*, 129-140.
- Chen, Q. (2020). T5: A detailed explanation. *Medium (Analytics Vidhya)*. https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51
- Hugging Face. (2024). *google-t5/t5-small*. https://huggingface.co/google-t5/t5-small

---

## License

This project uses T5-small under the Apache 2.0 license. Dataset combines public BigSalmon2 data with synthetic generations.