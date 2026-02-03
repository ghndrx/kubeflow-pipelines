# Healthcare ML Use Cases & Datasets

Curated list of similar healthcare/biomedical use cases with publicly available datasets for training on RunPod.

---

## 🔥 Priority 1: Ready to Train

### 1. Adverse Drug Event Classification
**Dataset:** `Lots-of-LoRAs/task1495_adverse_drug_event_classification`
- **Task:** Classify text for presence of adverse drug events
- **Size:** ~10K samples
- **Labels:** Binary (adverse event / no adverse event)
- **Use Case:** Pharmacovigilance, FDA reporting automation
- **Model:** Bio_ClinicalBERT

```python
from datasets import load_dataset
ds = load_dataset("Lots-of-LoRAs/task1495_adverse_drug_event_classification")
```

### 2. PubMed Multi-Label Classification (MeSH)
**Dataset:** `owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH`
- **Task:** Assign MeSH medical subject headings to research articles
- **Size:** ~50K articles
- **Labels:** Multi-label (medical topics)
- **Use Case:** Literature categorization, research discovery
- **Model:** PubMedBERT

```python
from datasets import load_dataset
ds = load_dataset("owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH")
```

### 3. Symptom-to-Disease Prediction
**Dataset:** `shanover/disease_symptoms_prec_full`
- **Task:** Predict disease from symptom descriptions
- **Size:** Variable
- **Labels:** Disease categories
- **Use Case:** Triage, symptom checker apps
- **Model:** Bio_ClinicalBERT

```python
from datasets import load_dataset
ds = load_dataset("shanover/disease_symptoms_prec_full")
```

### 4. Medical Triage Classification
**Dataset:** `shubham212/Medical_Triage_Classification`
- **Task:** Classify urgency level of medical cases
- **Size:** ~500 downloads (popular)
- **Labels:** Triage levels (Emergency, Urgent, Standard)
- **Use Case:** ER automation, telemedicine routing
- **Model:** Bio_ClinicalBERT

---

## 📚 Priority 2: QA & Reasoning

### 5. MedMCQA - Medical Exam Questions
**Dataset:** `openlifescienceai/medmcqa` (24K downloads!)
- **Task:** Answer medical entrance exam questions
- **Size:** 194K MCQs covering 2.4K healthcare topics
- **Labels:** Multiple choice (A/B/C/D)
- **Use Case:** Medical education, knowledge testing
- **Model:** Llama-3 or Gemma (LLM fine-tuning)

```python
from datasets import load_dataset
ds = load_dataset("openlifescienceai/medmcqa")
```

### 6. PubMedQA - Research Question Answering
**Dataset:** `qiaojin/PubMedQA` (18K downloads!)
- **Task:** Answer yes/no/maybe questions from abstracts
- **Size:** 274K samples
- **Labels:** yes / no / maybe
- **Use Case:** Evidence-based medicine, literature review
- **Model:** PubMedBERT or Bio_ClinicalBERT

```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA")
```

---

## 🧬 Priority 3: Specialized NLP

### 7. Medical Abbreviation Disambiguation (MeDAL)
**Dataset:** `McGill-NLP/medal`
- **Task:** Disambiguate medical abbreviations in context
- **Size:** 14GB → curated to 4GB
- **Labels:** Abbreviation meanings
- **Use Case:** Clinical note processing, EHR parsing
- **Model:** Bio_ClinicalBERT

### 8. BioInstruct - Instruction Following
**Dataset:** `bio-nlp-umass/bioinstruct`
- **Task:** Instruction-tuned biomedical tasks
- **Size:** 25K instructions
- **Labels:** Various biomedical tasks
- **Use Case:** General biomedical assistant
- **Model:** Llama-3 or Mistral (LoRA fine-tuning)

---

## 🛠️ Implementation Roadmap

### Week 1: Adverse Drug Events
1. Download ADE dataset
2. Add to handler.py as new training mode
3. Train classifier → S3
4. Build inference endpoint

### Week 2: PubMed Classification
1. Download PubMed MeSH dataset
2. Multi-label classification head
3. Train → S3
4. Literature search API

### Week 3: Medical QA
1. Download MedMCQA
2. LLM fine-tuning with LoRA
3. Deploy QA endpoint

### Week 4: Symptom Checker
1. Symptom-disease dataset
2. Train classifier
3. Build symptom input → disease prediction API

---

## 📊 Dataset Comparison

| Dataset | Size | Task | Difficulty | Business Value |
|---------|------|------|------------|----------------|
| DDI (current) | 176K | Classification | Medium | ⭐⭐⭐⭐⭐ |
| Adverse Events | 10K | Binary | Easy | ⭐⭐⭐⭐⭐ |
| PubMed MeSH | 50K | Multi-label | Medium | ⭐⭐⭐⭐ |
| MedMCQA | 194K | MCQ | Hard | ⭐⭐⭐⭐ |
| PubMedQA | 274K | Yes/No/Maybe | Medium | ⭐⭐⭐⭐ |
| Symptom→Disease | Varies | Classification | Easy | ⭐⭐⭐⭐⭐ |
| Triage | ~5K | Classification | Easy | ⭐⭐⭐⭐⭐ |

---

## 🔗 Additional Resources

- **MIMIC-III/IV:** ICU clinical data (requires PhysioNet access)
- **n2c2 Challenges:** Clinical NLP shared tasks
- **i2b2:** De-identified clinical records
- **ChemProt:** Chemical-protein interactions
- **BC5CDR:** Chemical-disease relations

---

*Generated: 2026-02-03*
