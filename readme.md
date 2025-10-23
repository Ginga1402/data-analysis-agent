# 📚 Reasonlytics: AI-Powered Data Analysis Agent

## Transform Your Data into Actionable Insights with Natural Language


## 🎯 Project Description

Reasonlytics is an intelligent data analysis agent that bridges the gap between raw data and meaningful insights. Built on LangGraph and powered by local LLMs via Ollama, Reasonlytics enables users to interact with their datasets using natural language queries and receive comprehensive analysis with human-readable explanations.

### What Makes Reasonlytics Special

**Reasonlytics combines the power of multiple AI agents working in harmony to deliver a seamless data analysis experience:**

- 🧠 Intelligent Query Understanding: Automatically classifies whether you want visualizations or data analysis

- 🐍 Dynamic Code Generation: Creates optimized pandas code tailored to your specific dataset and query

- ⚡ Safe Code Execution: Runs analysis in a secure, isolated environment

- 📊 Smart Visualization: Generates charts and plots when requested

- 💡 Contextual Reasoning: Provides clear, business-friendly explanations of results

- 🔍 Instant Dataset Insights: Automatically analyzes new datasets and suggests exploration questions


### 🏗️ Technical Architecture

The system follows a **modular, agent-based architecture** powered by **LangGraph**, **open-source LLMs**, and **Pandas** for structured, explainable data analysis workflows.

## LangGraph Agent Pipeline
- Orchestrates the **multi-step reasoning flow** using `MessagesState`  
- Executes **8 interconnected nodes** — covering data ingestion, insight generation, query classification, code synthesis, execution, and result explanation  
- Employs **custom @tool decorators** for modular tool execution (e.g., DataFrameSummaryTool, DataInsightAgent, CodeExecutionTool)  
- Deterministic routing ensures reproducible outputs (`DATA_INPUT → INSIGHT → QUERY → CODE_GEN → EXECUTION → EXPLANATION`)

## LLM Integration
- **Model:** Qwen2.5-Coder-7B-Instruct-Q4_K_M (configurable open-source LLM)  
- **Inference Engine:** Ollama / vLLM for local and GPU-accelerated deployments  
- **Prompt Templates:** Modular prompt blocks for summary, query classification, and code generation  
- **Output Parsing:** Structured text extraction with safety filtering for Python and visualization code  

## Data Handling Layer
- **Data Engine:** Pandas DataFrame (loaded from CSV, Excel, or SQL sources)  
- **Schema Summary:** Auto-generated dataset overview including types, missing values, and size metadata  
- **Validation:** Pre-execution code checks ensure only safe read-only operations (no file I/O, no external writes)  

## Visualization & Explanation Layer
- **Renderer:** Matplotlib / Seaborn for chart generation  
- **Result Display:** Inline rendering of visual and textual insights  
- **Reasoning Layer:** Generates natural-language explanations summarizing trends, outliers, and actionable insights  

## Observability & Extensibility
- Integrated logging for every node step with timestamped traces  
- Easily extensible to support other LLMs (Mistral, Gemma2, Llama3)  
- Can integrate with external data APIs or cloud storage connectors  



### ✨ Key Features

- **🗣️ Natural Language Interface**: Ask data-driven questions like “Show me sales trends by region” or “What’s the correlation between price and sales?” — no coding required. The agent understands intent and translates your query into executable Python or SQL automatically.
  
- **🤖 Automated Insights & Reasoning**: On every dataset upload or query, the agent instantly summarizes key patterns and relationships. It also provides clear explanations of results (e.g., “North region leads with 35% of total sales”) using an integrated Reasoning LLM.
  
- **📊 Multi-Modal Data Analysis**: Seamlessly handles both data analytics and visualization requests. The system dynamically decides whether to return a table, chart, or statistical summary based on the query context.
  
- **💡 Code Transparency & Safe Execution**: Every output comes with the generated pandas/matplotlib code for verification and learning. Code execution is sandboxed to ensure complete safety and prevent unauthorized operations.
  
- **🔒 Local, Private, and Configurable**: Runs entirely on your own infrastructure using Ollama and LangGraph, ensuring full data privacy. Supports multiple open-source LLMs like Qwen2.5, CodeGemma, Llama 3, and Mistral, with easy configuration for different workflows.

## 📁 Project Structure
```
📦 llm-data-analyst-agent-langgraph-ollama
│
├── configuration.py        # Environment setup and configurations
├── FastAPI.py              # API layer for backend integration
├── compile_agent.py        # Core LangGraph workflow
├── streamlit_app.py         # Streamlit frontend for user interaction
├── agent_tools.py            # Core agent and tools logic
└── README.md
└── License

```

### 💡 Use Cases

- **📈 Sales Performance Analysis**: “Show me total revenue by product category for the last quarter.”

The agent automatically aggregates the data, generates a bar chart, and explains key insights — such as which regions or products drive the highest revenue. 
- **🏪 Retail Demand Forecasting**: “Visualize weekly sales trends for top 5 products.”

The agent produces time-series plots, highlights seasonal patterns, and provides a reasoning summary to support inventory or marketing decisions.  
- **👩‍💼 HR Analytics Dashboard**: “What’s the average salary by department?” or “Plot employee attrition by age group.”

The agent creates pandas aggregations and visual insights to help HR teams identify trends and optimize workforce planning.  
- **💰 Financial Data Insights**: “Compare average returns across investment portfolios” or “Show me expense distribution by category.”

It generates precise visual summaries and explains financial performance differences in natural language.
  
- **🧠 Exploratory Data Analysis (EDA) Assistant**: “Give me a quick summary and possible questions to explore.”

The agent detects schema, missing values  


---

## 🧭 Demo Sample Images

**Streamlit Interface**

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/c36c2265-1288-497c-b36c-f46671d43ea9" />


<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/2afae2e9-8179-4765-a6ca-a8b45bd0f894" />


---

## 🛠️ Installation Instructions

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM recommended

### Step 1: Clone Repository
```bash
git clone https://github.com/Ginga1402/llm-data-analyst-agent-langgraph-ollama.git
cd llm-data-analyst-agent-langgraph-ollama
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```


### Step 3: Set Up Ollama (LLM Backend)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull codegemma:7b-instruct-v1.1-q4_K_S
```

### Step 4: Configure Paths
Update the paths in `configuration.py` to match your system:
```python
model_name = "qwen2.5-coder:7b-instruct-q4_K_M"
```

## 📖 Usage

### Starting the Application

1. **Start the FastAPI Server**:
```bash
python FastAPI.py
```
The API will be available at `http://localhost:8000`

2. **Launch the Streamlit Interface**:
```bash
streamlit run streamlit_app.py
```
The web interface will open at `http://localhost:8501`

### ⚙️ **Workflow Graph:**  
 
<img width="727" height="1080" alt="Image" src="https://github.com/user-attachments/assets/c56e1075-37e6-4f57-9dfa-488c7a65189d" />

---

### 🧱 Technologies Used

| Technology | Description | Link |
|------------|-------------|------|
| **LangChain** | Framework for building LLM-driven applications and chains | [LangChain](https://python.langchain.com) |
| **LangGraph** | State-based agent orchestration for complex LLM workflows | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **Ollama** | Local LLM inference engine for privacy-focused AI | [Ollama](https://ollama.ai) |
| **Mistral 7B (Q4_K_M)** | Quantized instruction-tuned model for query classification | [Mistral AI](https://mistral.ai) |
| **Qwen2.5-Coder 7B (Q4_K_M)** | Specialized code generation model for pandas operations | [Qwen Models](https://github.com/QwenLM/Qwen2.5-Coder) |
| **Qwen2.5 7B (Q4_K_M)** | General-purpose reasoning model for data insights | [Qwen Models](https://github.com/QwenLM/Qwen2.5) |
| **Pandas** | Data manipulation and analysis library for Python | [Pandas](https://pandas.pydata.org) |
| **Matplotlib** | Comprehensive plotting library for data visualization | [Matplotlib](https://matplotlib.org) |
| **Streamlit** | Web framework for building interactive data applications | [Streamlit](https://streamlit.io) |
| **PyTorch** | Deep learning framework with CUDA support | [PyTorch](https://pytorch.org) |
| **NumPy** | Fundamental package for scientific computing | [NumPy](https://numpy.org) |
| **FastAPI** | High-performance API framework for Python | [FastAPI](https://fastapi.tiangolo.com) |
| **Pydantic** | Data validation using Python type annotations | [pydantic.dev](https://pydantic.dev/) |

## 🤝 Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If you find Reasonlytics useful, please consider giving it a star ⭐ on GitHub!
