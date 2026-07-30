# SymptomSense AI

An AI-powered symptom analysis project that uses natural language processing and machine learning to map user-described symptoms and patterns to likely intent categories and generate relevant responses.

> **Project status:** Development / academic project

## 📌 Overview

**SymptomSense AI** is designed to process natural-language symptom descriptions and classify them into predefined intent categories. The project includes data preparation, exploratory data analysis (EDA), model training, and a deployment application.

The workflow is structured as:

1. Prepare and augment the original intent dataset.
2. Process the dataset into machine-learning-ready CSV files.
3. Perform exploratory data analysis and generate visualizations.
4. Train the intent classification model.
5. Use the trained model in the deployment application.

## ✨ Features

* Natural-language symptom and user-input processing
* Intent classification using a trained machine-learning model
* Dataset augmentation and preprocessing pipeline
* Exploratory data analysis (EDA)
* Visualization of intent and pattern distributions
* Deployment-ready application structure
* Modular organization of data processing, model training, and application code

## 🗂️ Project Structure

```text
SYMPTOSENSE AI/
│
├── Deployment/
│   └── app.py
│
├── EDA/
│   ├── EDA.ipynb
│   └── Plots/
│       ├── NUMBER OF INTENTS PER CATEGORY TYPE.png
│       ├── NUMBER OF PATTERNS PER INTENT.png
│       ├── NUMBERS OF INTENTS PER CATEGORY TYPE.png
│       ├── PATTERN LENGTH DISTRIBUTION.png
│       ├── PATTERN TO RESPONSE RATIO PER INTENT.png
│       ├── RESPONSE LENGTH DISTRIBUTION.png
│       └── TOP 20 MOST FREQUENT WORDS IN PATTERNS.png
│
├── Model Training/
│   └── train_model.py
│
├── Original Datasets/
│   ├── augment_intents.py
|   ├── intents_backup.json
│   ├── intents.json
│   └── process_data.py
│
├── Processed Datasets/
│   ├── processed_data_columns.csv
|   ├── processed_data_eda.csv
|   └── processed_data.csv
│
├── .gitignore
├── LICENSE
└── README.md
```

> Generated datasets, model files, and EDA plot images may be excluded from version control through `.gitignore`.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas
- Joblib
- Streamlit
- TF-IDF (Natural Language Processing)
- Logistic Regression
- Jupyter Notebook
- Git & GitHub

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yuvaraj-Dey-2006/SymptoSense-AI.git
cd "SYMPTOSENSE AI"
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the project's Python scripts and notebooks according to your environment.

## 📥 Dataset

This project uses the **Medical Assistance Dataset** available on Kaggle.

**Dataset:** https://www.kaggle.com/datasets/balajikartheek/medical-assistance-dataset

### Download Instructions

1. Download the dataset from Kaggle.
2. Extract the downloaded archive.
3. Copy the `intents.json` file into:

```text
Original Datasets/
```

4. Create a copy of `intents.json` and name it `intents_backup.json` and store it in the same directory.

The folder should look like:

```text
Original Datasets/
├── augment_intents.py
├── process_data.py
├── intents.json
└── intents_backup.json
```

> **Note:** The dataset is not included in this repository because it is ignored by Git. You must download it manually before running the project.

## 🚀 Running the Project

### Step 1: Prepare the dataset

Use the scripts in `Original Datasets/` to augment and process the source intent data.

```bash
python "Original Datasets/augment_intents.py"
python "Original Datasets/process_data.py"
```

The processed data can then be used for analysis and model training.

### Step 2: Perform EDA

Open the notebook:

```text
EDA/EDA.ipynb
```

Run the notebook to inspect the dataset and generate visualizations such as:

* Number of intents by category
* Number of patterns per intent
* Pattern length distribution
* Response length distribution
* Pattern-to-response ratios
* Most frequent words in patterns

### Step 3: Train the model

Run the training script:

```bash
python "Model Training/train_model.py"
```

The trained model artifacts should be generated according to the configuration in the training script.

### Step 4: Run the deployment application

Start the application using:

```bash
cd Deployment
Streamlit run app.py
```

The exact command and interface depend on the framework used in `app.py`.

## 📊 Data Pipeline

The project follows a simple data pipeline:

```text
Dataset (intents.json)
        │
        ▼
Data Augmentation
        │
        ▼
Data Processing
        │
        ▼
Processed CSV Dataset
        │
        ├────────► Exploratory Data Analysis
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Logistic Regression Training
        │
        ▼
Saved Model Artifacts (.pkl)
        │
        ▼
Deployment Application
```

## 📈 Exploratory Data Analysis

The EDA stage helps understand the structure and distribution of the dataset before model training. The generated plots provide insights into:

* Intent balance and category distribution
* Frequency of training patterns
* Length of text patterns and responses
* Vocabulary distribution
* Relationship between patterns and responses

## 🧠 Model Training

The model training pipeline is implemented in:

```text
Model Training/train_model.py
```

The training script:

- Loads the processed dataset
- Converts user symptom patterns into TF-IDF feature vectors
- Trains a Logistic Regression classifier
- Maps each intent tag to its possible responses
- Saves the trained artifacts using Joblib

Generated model files:

```text
Model/
├── classifier.pkl
├── vectorizer.pkl
└── tag_to_responses.pkl
```

## 🔒 Git & Ignored Files

The following generated or local files are excluded from version control:

```gitignore
Original Datasets/*.json
Processed Datasets/*.csv
Model/*.pkl
EDA/Plots/*.png
```

These files are automatically generated or downloaded locally and are not required to be stored in the repository.

## ⚠️ Disclaimer

SymptomSense AI is an educational and technical project intended for informational and research purposes. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Users should consult qualified healthcare professionals for medical concerns.

## 📄 License

This project is distributed under the license included in the [`LICENSE`](LICENSE) file.

## 👨‍💻 Author

**Yuvaraj Dey**

GitHub: [Yuvaraj-Dey-2006](https://github.com/Yuvaraj-Dey-2006)

## 👥 Collaborators

This project is maintained by the following collaborator(s):

### Current Collaborators

| Name | Role | GitHub |
|------|------|--------|
| **Yuvaraj Dey** | Model Training & Deployment | [Yuvaraj-Dey-2006](https://github.com/Yuvaraj-Dey-2006) |
| **Surmistha Datta** | Data Cleaning & Engineering | [Surmistha-Datta](https://github.com/Surmistha-Datta) |
| **Sumit Das** | Data Analysis | [dassumit2607-web](https://github.com/dassumit2607-web) |
| **Shubham Roy** | Data Analysis and PPT/Report | [royshubo067-dotcom](https://github.com/royshubo067-dotcom) |
| **Swastika Sarkar** | Data Fetching & Cleaning | [swastika104](https://github.com/swastika104) |

### How to Contribute

1. Fork the repository.
2. Clone your fork.
3. Create a new feature branch.

```bash
git checkout -b feature/your-feature-name
```

4. Make your changes and commit them.

```bash
git commit -m "Add your feature"
```

5. Push your branch.

```bash
git push origin feature/your-feature-name
```

6. Open a Pull Request describing your changes.

We appreciate all contributions, including:
- 🐞 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- ⚡ Performance optimizations
- 🧪 Tests and code quality improvements

---

⭐ If you find this project useful, consider giving the repository a star.
