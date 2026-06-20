# ⚡ Multi-Agent LLM Research System

A powerful **multi-agent AI research pipeline** built using **LangChain + Mistral AI + Streamlit**.

This system automatically:

* Breaks a topic into sub-questions
* Searches the web
* Extracts key insights
* Identifies research gaps
* Generates a professional report
* Critically evaluates the output

---

## 🚀 Features

### 🧠 1. Question Splitter Agent

* Converts a broad topic into focused sub-questions
* Ensures deep and structured research

### 🌐 2. Search Agent

* Uses Tavily API for real-time web search
* Handles multiple queries using `multi_search`
* Retrieves relevant and recent data

### 📖 3. Reader Agent

* Scrapes content from URLs
* Extracts:

  * Key facts
  * Important data
  * Arguments & insights
* Removes noise and duplication

### 🧠 4. Planner Agent

* Identifies missing research gaps
* Suggests follow-up queries for deeper analysis

### ✍️ 5. Writer Agent

* Generates a **professional research report** in Markdown
* Structured output:

  * Key Insights
  * Trends (2026)
  * Challenges
  * Opportunities
  * Detailed Analysis
  * Conclusion
  * Sources

### 🧪 6. Critic Agent

* Evaluates report quality (PhD-level review)
* Provides:

  * Score
  * Strengths
  * Weaknesses
  * Improvement suggestions

---

## 🖥️ Demo UI (Streamlit)

* Clean interface
* Step-by-step pipeline execution
* Expandable sections
* Download final report as `.md`

---

## ⚙️ Tech Stack

* **LangChain**
* **Mistral AI (mistral-small-2506)**
* **Tavily Search API**
* **BeautifulSoup (Web Scraping)**
* **Streamlit (Frontend UI)**
* **Python 3.11**

---

## 📂 Project Structure

```
Multi_agent/
│
├── pipeline.py        # Main execution pipeline
├── agents.py          # All agent definitions
├── tools.py           # Web search & scraping tools
├── app.py             # Streamlit UI
├── .env               # API keys
└── README.md
```

---

## 🔑 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kartikmarwari/Multi_agent-LLM.git
cd Multi_agent-LLM
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

Create a `.env` file:

```
TAVILY_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📥 Output Example

The system generates:

* Structured research report
* Clean Markdown output
* Downloadable file

---

## 🎯 Use Cases

* Research automation
* Market analysis
* AI trend reports
* Academic exploration
* Content generation

---

## 🔥 Future Improvements

* Streaming responses (real-time output)
* Memory-based reasoning
* Multi-step autonomous research loops
* Deployment (Streamlit Cloud / Vercel)

---

## 👨‍💻 Author

**Kartik Marwari**

---

## ⭐ Support

If you like this project:

* Star ⭐ the repo
* Share it
* Contribute improvements

---

## ⚠️ Disclaimer

This tool uses AI-generated content. Always verify critical information from original sources.
