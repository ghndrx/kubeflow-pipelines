# Healthcare ML Use Cases & Datasets

Curated list of healthcare/biomedical use cases with publicly available datasets.

---

## Implemented

### 1. Drug-Drug Interaction (DDI) Classification
- **Dataset:** DrugBank (bundled)
- **Task:** Classify interaction severity
- **Size:** 176K samples
- **Labels:** Minor, Moderate, Major, Contraindicated
- **Status:** Production ready

### 2. Adverse Drug Event Detection
- **Dataset:** `ade-benchmark-corpus/ade_corpus_v2`
- **Task:** Binary classification for ADE presence
- **Size:** 30K samples
- **Labels:** ADE / No ADE
- **Status:** Production ready

### 3. Symptom-to-Disease Prediction
- **Dataset:** `shanover/disease_symptoms_prec_full`
- **Task:** Predict disease from symptoms
- **Size:** ~5K samples
- **Labels:** 41 disease categories
- **Status:** Production ready

### 4. Medical Triage Classification
- **Dataset:** `shubham212/Medical_Triage_Classification`
- **Task:** Classify urgency level
- **Labels:** Emergency, Urgent, Standard, Non-urgent
- **Status:** Production ready (needs more training data)

---

## Future Candidates

### PubMed Multi-Label Classification (MeSH)
- **Dataset:** `owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH`
- **Task:** Assign MeSH subject headings to articles
- **Size:** 50K articles
- **Use Case:** Literature categorization

### MedMCQA - Medical Exam QA
- **Dataset:** `openlifescienceai/medmcqa`
- **Task:** Answer medical entrance exam questions
- **Size:** 194K MCQs
- **Use Case:** Medical education, knowledge testing

### PubMedQA - Research Question Answering
- **Dataset:** `qiaojin/PubMedQA`
- **Task:** Yes/No/Maybe from abstracts
- **Size:** 274K samples
- **Use Case:** Evidence-based medicine

### Medical Abbreviation Disambiguation
- **Dataset:** `McGill-NLP/medal`
- **Task:** Disambiguate abbreviations in context
- **Size:** 4GB curated
- **Use Case:** Clinical note processing

### BioInstruct
- **Dataset:** `bio-nlp-umass/bioinstruct`
- **Task:** Instruction-tuned biomedical tasks
- **Size:** 25K instructions
- **Use Case:** General biomedical assistant

---

## Dataset Comparison

| Dataset | Size | Task | Complexity |
|---------|------|------|------------|
| DDI (DrugBank) | 176K | 4-class | Medium |
| ADE Corpus | 30K | Binary | Low |
| PubMed MeSH | 50K | Multi-label | High |
| MedMCQA | 194K | MCQ | High |
| PubMedQA | 274K | 3-class | Medium |
| Symptom-Disease | 5K | 41-class | Medium |
| Triage | 5K | 4-class | Low |

---

## Additional Resources

- **MIMIC-III/IV:** ICU clinical data (PhysioNet access required)
- **n2c2 Challenges:** Clinical NLP shared tasks
- **i2b2:** De-identified clinical records
- **ChemProt:** Chemical-protein interactions
- **BC5CDR:** Chemical-disease relations
