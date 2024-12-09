# Fine-Tuning DistilBERT for Sentiment Analysis - IMDB Dataset of 50K Movie Reviews Challenge

This repository demonstrates the process of fine-tuning **DistilBERT**, a lightweight version of **BERT**, for **sentiment analysis** on the **IMDB dataset of 50,000 movie reviews**. This project is designed to showcase the power of **large language models** like **DistilBERT** in natural language processing tasks.

## Project Overview

In this project, we use **DistilBERT** for classifying movie reviews into **positive** and **negative** sentiments. The process involves data preparation, model fine-tuning, and evaluation. 

### Key Steps:
1. **Data Preparation**:
    - The IMDB dataset is cleaned, and sentiment labels are converted into binary format (positive = 1, negative = 0).
    - The reviews are tokenized using **DistilBERT**'s tokenizer.
  
2. **Model Training**:
    - **DistilBERT** is fine-tuned using the **train** split of the dataset. The model is trained on a GPU to speed up the process.
    - We apply techniques like **learning rate scheduling** and **gradient accumulation** to optimize training.

3. **Model Evaluation**:
    - After training, the model is evaluated using the **validation** split. The accuracy and performance of the model are measured.

4. **Optimization**:
    - The model's training process uses **AdamW optimizer**, with a **linear learning rate decay**.
    - **Batching** and **GPU acceleration** techniques are employed to speed up training.

## Getting Started

To run the code on your machine:

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/fine-tuning-distilbert-sentiment-analysis.git
cd fine-tuning-distilbert-sentiment-analysis
```
### 2. Install dependencies:
```
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook for model training and evaluation:
```
jupyter notebook fine_tune_distilbert.ipynb 
```
## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.5+
- scikit-learn
- tqdm

## Contributions

Feel free to open an issue or submit a pull request for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### How to Use the README:
- Replace `yourusername` with your actual GitHub username or the username for your repository.
- Make sure the Python script filenames and paths match the ones used in your project.
