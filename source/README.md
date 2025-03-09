## **README: Project Usage Guide**


### **1. Setup Instructions**

#### **Step 1: Install Dependencies**
Use the provided `requirements.txt` file to set up the Python environment:
```bash
pip install -r requirements.txt
```

Ensure you have:
- Python 3.10 or above
- CUDA-compatible GPU (if available) for efficient embedding generation and fine-tuning.

#### **Step 2: Configure the Data Folder**
Before running the script, you need to set up the `data` directory with the required datasets:
1. Download the dataset archive from the following link:  
   [Our data](https://drive.google.com/file/d/1VsGPA8iqbg7Ue_bliV9nmbIn9XGTWGA6/view?usp=sharing)
2. Extract the downloaded `.tar.gz` file into the `data` folder:
   ```bash
   mkdir -p ../data
   tar -xvzf data.tar.gz -C ../data
   ```

After extraction, the `data` folder should have the following structure:
```
data/
├── weights
├── training_Q_A.json
├── dev_Q_A.json
├── training_dev_technotes.json
└── other necessary files...
```

#### **Step 3: Verify Data and Weights Paths**
Ensure the paths specified in the script match your folder structure:
- **Data Path**: `../data`
- **Weights Path**: `../data/weights`

If needed, customize these paths using the `--data_path` and `--weights_path` arguments.


### **2. Running Experiments**

#### **Basic Command**
To execute the pipeline, run:
```bash
python run_experiment.py --question_title "Your Question Here"
```

#### **Command-Line Arguments**

| Argument                | Default Value             | Description                                                                                     |
|-------------------------|---------------------------|-------------------------------------------------------------------------------------------------|
| `--data_path`           | `../data`                | Path to the directory containing datasets and preprocessed files.                              |
| `--weights_path`        | `../data/weights`        | Path to the directory containing pre-trained embeddings and fine-tuned model weights.          |
| `--embedding_model_name`| `all-mpnet-base-v2`      | Name of the sentence transformer model for generating embeddings.                              |
| `--method`              | `embedding`             | Retrieval method: `embedding` (default), `bm25`, or `chunk`.                                   |
| `--filter_threshold`    | `0.7`                    | Cosine similarity threshold for filtering retrieved documents in embedding-based retrieval.     |
| `--top_k`               | `10`                     | Number of top documents to retrieve.                                                          |
| `--question_title`      | **Required**             | The query for which relevant documents and answers are to be generated.                        |
| `--Llm_model_name`      | `deepseek-chat`          | Name of the LLM model to use for generating answers.                                           |
| `--url`                 | `https://api.deepseek.com/v1` | URL of the LLM API.                                                                             |
| `--token`               | **Required** (example provided) | Authentication token for accessing the LLM API.                                                |

### **3. Customization**

You can customize the behavior of the pipeline using the following options:

1. **Data Path**:
   - Use `--data_path` to specify the directory containing the datasets and preprocessed files.  
   - Similarly, `--weights_path` points to the directory with pre-trained embeddings and fine-tuned model weights.

2. **Embedding Model**:
   - By default, the embedding model is `all-mpnet-base-v2`. You can replace it with other sentence-transformer-compatible models by changing the `--embedding_model_name` parameter.

3. **Retrieval Method**:
   - Use the `--method` argument to switch between retrieval strategies:
     - `embedding`: Embedding-based retrieval (default).
     - `bm25`: BM25-based lexical retrieval.
     - `chunk`: Chunk-based retrieval for long documents.

4. **Adjusting Precision and Recall**:
   - Modify the similarity threshold (`--filter_threshold`) to balance retrieval quality and precision.  
   - Increase or decrease the number of retrieved documents (`--top_k`) to adapt to specific requirements.

5. **LLM Configuration**:
   - Use `--Llm_model_name` to specify the LLM used for answer generation (e.g., `deepseek-chat`).
   - Configure the API endpoint URL with `--url`.
   - Provide the required API authentication token via the `--token` argument.


### **4. Example Usage**

1. **Default Configuration**:
   ```bash
   python run_experiment.py --question_title "Help with Security Bulletin: Vulnerabilities in IBM Dojo Toolkit affect IBM Image Construction and Composition Tool (CVE-2014-8917)"
   ```

2. **Using BM25 for Retrieval**:
   ```bash
   python run_experiment.py --method bm25 --question_title "Help with Security Bulletin: Vulnerabilities in IBM Dojo Toolkit affect IBM Image Construction and Composition Tool (CVE-2014-8917)",
   ```


3. **Adjusting Precision and Recall**:
   ```bash
   python run_experiment.py --filter_threshold 0.8 --top_k 5 --question_title "Help with Security Bulletin: Vulnerabilities in IBM Dojo Toolkit affect IBM Image Construction and Composition Tool (CVE-2014-8917)",
   ```

4. **Using a Custom LLM API**:
   ```bash
   python run_experiment.py --Llm_model_name "custom-llm" --url "https://custom-llm-api.com/v1" --token "your-api-token" --question_title "Help with Security Bulletin: Vulnerabilities in IBM Dojo Toolkit affect IBM Image Construction and Composition Tool (CVE-2014-8917)",
   ```


### **5. Additional Scripts**
- `retrieve.py`: Handles retrieval logic (BM25 and embedding-based).
- `embedding.py`: Generates embeddings for long texts and aggregates them for efficient retrieval.
- `bm25.py`: Implements BM25-based preprocessing and ranking.
- `fine_tune.py`: Fine-tunes the QA model for extractive tasks.
- `util.py`: Contains utility functions for preprocessing and data loading.
