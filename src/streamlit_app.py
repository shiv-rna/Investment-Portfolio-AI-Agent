"""
Investment Portfolio Analysis Streamlit Application

This module provides a web-based interface for the Investment Portfolio Risk Assessment AI,
allowing users to interact with the financial analysis tools and receive investment insights.

The application uses Streamlit to create an interactive dashboard where users can:
1. Input their Groq API key for AI interactions
2. Select from predefined investment queries or enter custom queries
3. Receive detailed portfolio analysis, risk assessments, and recommendations

Dependencies:
- streamlit
- invest_portfolio_risk_react_ai_agent (custom module)
- os
- sys
- logging
- re

Usage:
Run this script to launch the Streamlit application:
    streamlit run streamlit_app.py

Author: Shivang Rana
Created: 11-27-2024
Version: 1.0.0
"""
import os
import sys
import logging
import re
from typing import Dict, Any
import json
from pathlib import Path

import streamlit as st

# Add the directory containing your original script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import everything from the original script
from invest_portfolio_risk_react_ai_agent import (
    agent_loop, 
    load_system_prompt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stock_data(): 
    """
    Load stock data from the JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing stock data
    
    Returns:
        dict: Dictionary of stock information
    """
    try:
        # Get the current file's directory
        current_dir = Path(__file__).parent
        
        # Construct path to JSON file
        json_path = current_dir / 'stock-risk-profile-json.json'
        
        with json_path.open('r') as file:
            stock_data = json.load(file)
            return stock_data['stocks']
    except FileNotFoundError:
        st.error(f"Stock data file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {file_path}")
        return {}

        
def display_sidebar(stocks) -> str:
    """
    Create a sidebar with API key input.
    Add a stock list display to the Streamlit sidebar.

    Args:
        stocks (dict): Dictionary of stock information

    Returns:
        str: The entered Groq API key.
    """
    st.sidebar.title("Investment Portfolio Risk Assessment")

    # Groq API Key input
    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input(
        "Enter Groq API Key", 
        type="password", 
        help="Your Groq API key for AI interactions"
    )

    # Sidebar section for supported stocks
    st.sidebar.header("🏛 Supported Stocks")
    
    # Group stocks by sector
    sectors = {}
    for ticker, stock_info in stocks.items():
        sector = stock_info['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((ticker, stock_info['name']))
    
    # Create an expander for each sector
    for sector, stock_list in sorted(sectors.items()):
        with st.sidebar.expander(f"{sector} Stocks"):
            stock_display = "\n".join([f"{ticker}: {name}" for ticker, name in sorted(stock_list)])
            st.text(stock_display)
    
    # Total stocks and sectors summary
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Supported Stocks", len(stocks))
    st.sidebar.metric("Total Sectors", len(sectors))


    return api_key

def capture_output(max_iterations: int, system_prompt: str, query: str, api_key: str) -> Dict[str, str]:
    """
    Capture and return the output from agent_loop.
    
    Args:
        max_iterations (int): Maximum number of iterations for the agent loop.
        system_prompt (str): The system prompt for the agent.
        query (str): The user's investment query.
        api_key (str): The Groq API key.

    Returns:
        Dict[str, str]: A dictionary containing the full output and final answer.
    """
    # Temporarily redirect stdout to capture print statements
    import io
    import sys

    # Create a string buffer to capture output
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    try:
        # Set the Groq API key in environment
        os.environ['GROQ_API_KEY'] = api_key

        # Call agent_loop
        agent_loop(max_iterations, system_prompt, query)

        # Get the captured output
        full_output = output_buffer.getvalue()

        # Extract the Answer section
        answer_match = re.search(r'(?:\*\*Answer\*\*|Answer):(.*)', full_output, re.DOTALL | re.IGNORECASE)
        if answer_match:
            final_answer = answer_match.string[answer_match.start():].lstrip('Answer:').strip()
        else:
            final_answer = "No specific answer found."
        
        return {
            'full_output': full_output,
            'final_answer': final_answer
        }

    except Exception as e:
        logger.error(f"Error in capture_output: {str(e)}")
        return {
            'full_output': f"Error: {str(e)}",
            'final_answer': f"Error processing query: {str(e)}"
        }
    finally:
        # Restore stdout
        sys.stdout = original_stdout

def main():
    """
    Main function to run the Streamlit application for Investment Portfolio Analysis.

    This function sets up the Streamlit page configuration, displays the sidebar,
    handles user input (API key and investment query), and manages the interaction
    with the AI agent for generating investment insights.

    The function performs the following tasks:
    1. Sets up the Streamlit page configuration
    2. Displays the sidebar with tool capabilities and API key input
    3. Loads the system prompt for the AI agent
    4. Provides a text input for user queries
    5. Processes the user query and displays the AI-generated insights
    """
    st.set_page_config(
        page_title="Investment Portfolio AI Assistant", 
        page_icon="📈", 
        layout="wide"
    )

    st.title("🤖 Investment Portfolio Risk Assessment ReAct AI Agent")

    stocks = load_stock_data()

    api_key = display_sidebar(stocks)
    if not api_key:
        st.warning("Please enter your Groq API Key to continue.")
        return

    system_prompt = load_system_prompt()

    st.markdown("### Example Queries")
    st.markdown("Select a predefined query or enter your own below:")

    example_queries = [
        "Get risk profile for NVDA stock",
        "Analyze a portfolio with 40% AAPL, 30% GOOGL, 30% SPY",
        "Calculate expected return for a portfolio with 50% VTI, 30% QQQ, 20% BAC",
        "Recommend adjustments for a portfolio with 50% MRNA, 30% JNJ, 20% SPY with low risk tolerance",
        "Analyze and optimize a tech-heavy portfolio with 30% AAPL, 25% MSFT, 25% GOOGL, 20% NVDA for medium risk tolerance, suggesting sector diversification"
    ]

    selected_query = st.selectbox("Choose an example query:", [""] + example_queries)

    query = st.text_input(
        "Enter your investment query or use the selected example above:", 
        value=selected_query,
        placeholder="E.g., Get risk profile for NVDA, Analyze portfolio with 40% AAPL, 30% GOOGL, 30% SPY"
    )

    if st.button("Get Investment Insights🔍"):
        if not query:
            st.error("Please select an example query or enter your own")
            return

        with st.spinner('Generating investment insights ⚙️...'):
            try:
                # Capture and display output
                output = capture_output(
                    max_iterations=5,
                    system_prompt=system_prompt,
                    query=query,
                    api_key=api_key
                )

                st.markdown("### Key Insights 🎯")
                st.text_area("Response", output['final_answer'], height=300)


                st.markdown("### Detailed Analysis")
                st.code(output['full_output'], language='text')

            except Exception as e:
                logger.error(f"Error in main function: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
    
    if st.button("Get Tool Capabilities🔨"):
        st.markdown("""
        ### Tool Capabilities 🚀

        This AI-powered investment tool provides:

        1. **Stock Risk Profile**
            - Detailed risk assessment for individual stocks
            - Key metrics like volatility, market cap, and return potential

        2. **Portfolio Diversification Analysis**
            - Sector and risk level breakdown
            - Allocation insights

        3. **Portfolio Return Calculation**
            - Estimated annual return calculation
            - Based on historical stock performance

        4. **Portfolio Adjustment Recommendations**
            - Personalized recommendations
            - Aligned with your risk tolerance
        """)


if __name__ == "__main__":
    main()
