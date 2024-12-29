# 📈 Investment Portfolio Risk Assessment AI Agent
<a href="https://investment-portfolio-ai-agent.streamlit.app/"><img src="https://img.shields.io/badge/deployment-website-blue" alt="Website"/></a>

## 🌟 Project Overview

An AI-powered investment analysis tool that leverages simple ReAct AI agent flow framework and financial analysis techniques to provide comprehensive portfolio insights. This intelligent agent helps investors make data-driven decisions by offering deep portfolio risk assessment, stock profiling, and personalized recommendations.

## 🏗️ Key Features
- **Stock Risk Profiling**: Detailed risk assessment for individual stocks, including volatility, market cap, and return potential.
- **Portfolio Diversification Analysis**: Breakdown of sector allocation and risk levels within a given portfolio. 
- **Expected Return Calculation**: Estimation of annual returns based on historical stock performance.
- **Intelligent Portfolio Adjustments**: Personalized recommendations aligned with user's risk tolerance.
- **Interactive Web Interface**: User-friendly Streamlit application for easy interaction with the AI agent.

## 👨‍🏫 How ReAct Agent works

The Investment Portfolio Analysis AI Agent is built on the ReAct (Reasoning and Action) framework, which combines the strength of large language models with a structured approach to problem-solving. Here's the workflow:

1. **Thought**: The agent analyzes the user's query and formulates a plan of action.
2. **Action**: Based on the thought, the agent selects and executes an appropriate tool or API call.
3. **Observation**: The agent observes and interprets the results of the action.
4. **Repeat**: This cycle continues until the agent has gathered enough information to provide a comprehensive answer.

### Extension to Tool Calling

The ReAct framework can be extended to incorporate specialized tools:

- **Tool Definition**: Each tool (e.g., stock risk profiler, portfolio analyzer) is defined with clear inputs and outputs.
- **Tool Selection**: The agent learns to choose the most appropriate tool based on the current context and user query.
- **Tool Execution**: The selected tool is called with the necessary parameters.
- **Result Integration**: The agent incorporates tool outputs into its reasoning process for the final response.

This extension allows the agent to leverage specific financial analysis functions while maintaining a flexible, language-model-driven interaction flow.

## 🚀 Technologies Used

- **Backend**: Python
- **AI Model**: Groq API
- **Frontend**: Streamlit
- **Key Libraries**:
  - Groq
  - python-dotenv
  - Logging
  - JSON processing
  - Dataclasses

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Groq API Key

### Setup Steps
1. Clone the repository
```bash
git clone https://github.com/yourusername/investment-portfolio-ai.git
cd investment-portfolio-ai
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
- Create a `.env` file
- Add your Groq API key: `GROQ_API_KEY=your_groq_api_key`
- Optionally you can add file paths to stock risk json file and system prompt

`STOCK_DATA_PATH=your_stock_data_json_file_path`
`SYS_PROMPT_DATA_PATH=your_system_prompt_data_path`

## 🖥️ Usage

### Streamlit Web Application
```bash
streamlit run streamlit_app.py
```

### Example Queries
- "Get risk profile for NVDA stock"
- "Analyze a portfolio with 40% AAPL, 30% GOOGL, 30% SPY"
- "Calculate expected return for a portfolio"
- "Recommend adjustments for a portfolio with specific risk tolerance"

## 📊 How It Works

The AI agent uses a ReAct (Reasoning and Acting) framework to:
1. Understand the user's investment query
2. Choose appropriate financial analysis tools
3. Generate data-driven insights and recommendations

### Key Components
- **FinancialTools**: Comprehensive analysis methods
- **Agent**: Intelligent interaction and tool selection
- **Streamlit Interface**: User-friendly web application

## 🛡️ Risk Disclaimer

*Investment involves risks. This tool provides insights and should not be considered definitive financial advice. Always consult with a financial professional before making investment decisions.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 💌Acknowledgments

- Groq for providing the API used in this project.
- Streamlit for the excellent web app framework.
- Contributors and maintainers of the open-source libraries used in this project.

## 👋 Connect with Me
Feel free to reach out for collaboration, questions, or potential opportunities! I'm always excited to discuss 
- 🧠 Intelligent AI Agents
- 📊 Risk Analysis & Financial Technology
- 🤖 Machine Learning Solutions
- 🚀 AI-Driven Decision Support Systems


[<img src="https://github.com/shiv-rna/Airflow-Basics/blob/e4ea0578dc2f664532a17755fe21534a9bd33e51/docs/linkedin.png" alt="Linkedin" width="50"/>](https://www.linkedin.com/in/sr099/) [<img src="https://github.com/shiv-rna/Airflow-Basics/blob/e4ea0578dc2f664532a17755fe21534a9bd33e51/docs/twitterx.png" alt="TwitterX" width="50"/>](https://twitter.com/wtfisshivang)
