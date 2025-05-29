# 📩 Customer Support Ticket Classification & Entity Extraction

## 🎯 Objective
Develop a machine learning pipeline that:
- Classifies customer support tickets by **issue type** and **urgency level**
- Extracts key entities from ticket text such as **product names**, **dates**, and **complaint keywords**

---

## 📁 Dataset
Provided Excel File: `ai_dev_assignment_tickets_complex_1000.xlsx`

### Columns:
- `ticket_id`
- `ticket_text`
- `issue_type` (target)
- `urgency_level` (target: Low, Medium, High)
- `product` (ground truth for entity extraction)

---

## ⚙️ Pipeline Components

### 1. Data Preprocessing
- Lowercasing, punctuation & special character removal
- Tokenization, stopword removal, lemmatization
- Handling missing values

### 2. Feature Engineering
- TF-IDF vectors
- Ticket text length
- Sentiment score

### 3. Multi-Task ML Models
- **Issue Type Classifier** (Multiclass)
- **Urgency Level Classifier** (Multiclass)
- Models used: Logistic Regression, Random Forest, SVM

### 4. Entity Extraction
- Product name (matched from list)
- Dates (regex-based)
- Complaint keywords (e.g., "broken", "late", "error")

### 5. Integration Function
```python
def process_ticket(ticket_text: str) -> dict:
    return {
        "issue_type": ...,
        "urgency_level": ...,
        "entities": {
            "product": ...,
            "date": ...,
            "complaint_keywords": [...]
        }
    }

### 6. Optional Gradio Interface
- Input raw ticket text
- View predictions + extracted entities

---

## 📈 Evaluation
- Accuracy & F1-Score
- Confusion matrix
- Manual entity evaluation

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
# or
streamlit run app_gradio.py  # For Gradio interface
