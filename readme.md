````markdown
# SDGE Gas Department Data Cleaning & EDA

This repository focuses on **data cleaning** and **exploratory data analysis (EDA)** for the **San Diego Gas & Electric (SDGE) Gas Department**.  
It processes multiple raw operational tables (e.g., `TASKS`, `ASSIGNMENTS`, `ENGINEERS`, etc.), merges them into a unified dataset, and generates visual and statistical insights to support further analysis on scheduling effectiveness, work duration, and crew efficiency.

---

## Requirements:
   requires gas analysis data store in `data/click/` 
   This project is tested on **Anaconda > 24.0** (with Python 3.12).  
   Other version of Anaconda should work without issues.

## 1. Installation

Create a clean Python environment (Python **3.12**) and install all dependencies:

```bash
conda create -n sdge python=3.12
conda activate sdge
pip install -r requirements.txt
````

---

## 2. Load and Merge Tables

Make sure all raw data files are stored under the folder:

```
data/click/
```

To **merge and save** the cleaned result:

```bash
python merge_table.py
```

The cleaned and merged result will be saved to:

```
data/cleaned_result/
```

To **merge, save, and generate an HTML report**:

```bash
python merge_table.py --report
```

An automatic **HTML report** (e.g., `Merge_tables_overview.html`) will be generated summarizing the merged tables.

---

## 3. Run Notebooks

Open and run any notebook (files ending with `.ipynb`) to reproduce the **EDA results**, visualizations, and further analyses.

---

### Example Directory Structure

```
.
├── Merge_tables_overview.html
├── brain_eda.ipynb
├── data
│   ├── cleaned_result
│   │   └── ...
│   └── click
│       └── ...
├── docs
├── george_eda.ipynb
├── merge_table.py
├── README.md
├── requirements.txt
├── stephaine_eda.ipynb
├── tina_eda.ipynb
└── util
    ├── __init__.py
    └── build_report.py
```

```
```


