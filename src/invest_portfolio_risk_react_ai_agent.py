"""
Investment Portfolio Risk Assessment Tool

A sophisticated financial analysis module for comprehensive
portfolio risk assessment and intelligent recommendations.

Dependencies:
- groq
- python-dotenv
- typing
- dataclasses

Author: Shivang Rana
Created: 11-25-2024
Version: 1.0.0
"""

import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

import dotenv
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('investment_portfolio.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

@dataclass
class StockData:
    """
    Represents detailed information about a stock.

    Attributes:
        name (str): Full name of the company.
        sector (str): Industrial sector of the company.
        beta (float): Stock's volatility relative to the market.
        market_cap (float): Total market capitalization.
        volatility_index (float): Measure of stock price fluctuations.
        estimated_risk (str): Qualitative risk assessment.
        avg_annual_return (float): Historical average annual return.
    """
    name: str
    sector: str
    beta: float
    market_cap: float
    volatility_index: float
    estimated_risk: str
    avg_annual_return: float

class FinancialTools:
    """
    Comprehensive financial analysis toolkit for investment portfolios

    Provides methods for portfolio diversification, risk assessment,
    return calculations, and intelligent recommendations.
    """

    def __init__(self):
        """
        Initialize FinancialTools by loading stock data.
        """
        self.STOCK_DATA = self.load_stock_data()

    def load_stock_data(self, file_path: str = 'stock-risk-profile-json.json') -> Dict[str, StockData]:
        """
        Load stock data from JSON file and convert to StockData objects.

        Args:
            file_path (str, optional): Path to stock risk profile JSON.
                Defaults to 'stock_risk_profile.json'.
        
        Returns:
            Dict[str, StockData]: Mapping of stock symbols to StockData instances.
        """
        file_path = os.getenv("STOCK_DATA_PATH", file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)['stocks']
            return {
                symbol: StockData(**data) for symbol, data in raw_data.items()
            }
        except FileNotFoundError:
            logger.error(f"Stock data file not found at {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding stock data JSON: {e}")
        return {}

    def analyze_portfolio_diversification(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze portfolio diversification across sectors and risk levels.

        Args:
            portfolio (Dict[str, float]): Stock allocations with percentage weights.

        Returns:
            Dict[str, Any]: Comprehensive diversification analysis.
        """
        sectors = {}
        risk_levels = {}
        total_allocation = sum(portfolio.values())

        for stock, allocation in portfolio.items():
            stock_info = self.STOCK_DATA.get(stock, {})
            if stock_info:
                sector = stock_info.sector
                risk_level = stock_info.estimated_risk
            else:
                sector = 'Unknown'
                risk_level = 'Unknown'

            sectors[sector] = sectors.get(sector, 0) + allocation
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + allocation
        
        return {
            "total_allocation": total_allocation,
            "sector_breakdown": sectors,
            "risk_level_breakdown": risk_levels
        }
    
    def calculate_expected_portfolio_return(self, portfolio: Dict[str, float]) -> float:
        """
        Calculate expected annual return of the portfolio.

        Args:
            portfolio (Dict[str, float]): Stock allocations with percentage weights.

        Returns:
            float: Expected annual return percentage
        """
        total_return = 0
        for stock, allocation in portfolio.items():
            stock_info = self.STOCK_DATA.get(stock, {})
            if stock_info:
                avg_return = stock_info.avg_annual_return
            else:
                avg_return = 0
            total_return += avg_return * (allocation/ 100)
        return total_return * 100

    def recommend_portfolio_adjustments(self, portfolio: Dict[str, float], risk_tolerance: str) -> List[str]:
        """
        Recommend portfolio adjustments based on risk tolerance.

        Args:
            portfolio (Dict[str, float]): Current portfolio allocation.
            risk_tolerance (str): Risk tolerance level.
        
        Returns:
            List[str]: Recommended portfolio adjustments.
        """
        current_analysis = self.analyze_portfolio_diversification(portfolio)
        recommendations = []

        # Risk Mapping based on real-world risk levels
        risk_mapping = {
            "low": ["SPY", "VTI", "JNJ", "PFE", "JPM"], # Conservative, stable stocks
            "medium": ["AAPL", "GOOGL", "MSFT", "QQQ", "BAC", "XOM"], # Established, moderate growth stocks
            "high": ["NVDA", "MRNA", "COIN"], # High Growth, high volatility stocks
        }

        # Detailed recommendation logic
        current_sectors = current_analysis['sector_breakdown']
        current_risk_levels = current_analysis['risk_level_breakdown']

        # Low-risk tolerance recommendations
        if risk_tolerance == "low":
            if current_risk_levels.get('High', 0) > 20:
                recommendations.append("Significantly reduce exposure to high-risk stocks")
            if any(stock in risk_mapping['high'] for stock in portfolio):
                recommendations.append("Remove high-volatility stocks from the portfolio")
            recommendations.append("Consider increasing allocation to stable sectors like Index Funds and Healthcare")

        # Medium-risk tolerance recommendations
        elif risk_tolerance == "medium":
            if current_risk_levels.get('High', 0) > 30:
                recommendations.append("Rebalance portfolio to reduce high-risk stock exposure")
            if sum(1 for stock in portfolio if stock in risk_mapping['high']) > 2:
                recommendations.append("Limit the number of high-risk stocks to maintain portfolio stability")
            recommendations.append("Maintain a balanced mix of technology, financial, and index fund stocks")

        # High-risk tolerance recommendations
        elif risk_tolerance == "high":
            if current_risk_levels.get('Low', 0) > 40:
                recommendations.append("Consider increasing exposure to high-growth stocks")
            recommendations.append("Explore opportunities in emerging technologies and speculative sectors")
            recommendations.append("Be prepared for higher volatility and potential higher returns")

        # Sector diversification check
        if len(current_sectors) < 3:
            recommendations.append("Improve portfolio diversification by adding stocks from different sectors")

        return recommendations

    def get_stock_risk_profile(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Get detailed risk profile for a specific stock.

        Args:
            stock_symbol (str): Stock symbol to retrieve risk profile for.

        Returns:
            Dict[str, Any]: Comprehensive stock risk profile.
        """
        stock_data = self.STOCK_DATA.get(stock_symbol, {})
        if not stock_data:
            return {"Error": f"Stock symbol {stock_symbol} not found in dataset."}
        
        risk_explanations = {
            "Very High": "Extremely volatile with potential for significant gains or losses.",
            "High": "Significant price volatility and potential for substantial price swings.",
            "Medium": "Moderate price fluctuations with balanced growth potential.",
            "Medium-Low": "Relatively stable with some price variations.",
            "Low": "Minimal price volatility, typically large, established companies with consistent performance."
        }
        
        return {
            "Name": stock_data.name,
            "Sector": stock_data.sector,
            "Risk Level": stock_data.estimated_risk,
            "Beta": stock_data.beta,
            "Market Cap": f"${stock_data.market_cap / 1e9:.2f} Billion",
            "Volatility Index": stock_data.volatility_index,
            "Average Annual Return": f"{stock_data.avg_annual_return * 100:.2f}%",
            "Risk Explanation": risk_explanations.get(stock_data.estimated_risk, 'N/A')
        }

class Agent:
    """
    Investment portfolio analysis agent with interactive capabilities.
    """
    def __init__(self, client, system, financial_tools):
        """
        Initialize the agent with Groq client and system prompt.

        Args:
            client: Groq API client
            system (str): System prompt for guiding agent behavior
        """
        self.client = client
        self.system = system
        self.messages = []
        self.financial_tools = financial_tools
        
        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})
    
    def __call__(self, message=""):
        """
        Execute agent interaction.

        Args:
            message (str, optional): User message. Defaults to "".

        Returns:
            str: Agent's response
        """
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        return result
    
    def execute(self):
        """
        Execute Groq API call to generate agent response.

        Returns:
            str: Generated response from the language model
        """
        try:
            completion = self.client.chat.completions.create(
                messages=self.messages,
                model="llama3-70b-8192",
            )
            result = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            logger.error(f"Error during Groq API call: {e}")
            return "Error: Unable to process your agent execute request at this time"


def load_system_prompt(file_path: str = 'system_prompt.txt') -> str:
    """
    Load system prompt from a text file

    Args:
        file_path (str): File path to system prompt txt file
    
    Returns:
        str: system_prompt
    """
    file_path = os.getenv("SYS_PROMPT_DATA_PATH", file_path)
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("System prompt file not found. Using default.")
        return """You are an Investment Portfolio Analysis Agent..."""

def execute_tool_action(chosen_tool, args_str, financial_tools):
    """
    Execute a financial tool action based on the specified tool and arguments.

    Args:
        chosen_tool (str): Name of the financial tool function.
        args_str (str): Arguments for the tool function.
        financial_tools (FinancialTools): Financial tools helper.

    Returns:
        str: Output of the tool execution.
    """
    try: # Handle different tool requirements
        # Tool: get_stock_risk_profile
        if chosen_tool =="get_stock_risk_profile":
            stock_symbol = args_str.strip('"\'') # Remove surrounding quotes if present
            result_tool = financial_tools.get_stock_risk_profile(stock_symbol)
            return json.dumps(result_tool, indent=2)

        # Tool: recommend_portfolio_adjustments
        elif chosen_tool == "recommend_portfolio_adjustments":
            # Expect args_str in the format: '{"AAPL": 50, "GOOGL": 30, "SPY": 20}, low'
            # Split portfolio and risk tolerance
            portfolio_str, risk_tolerance = [arg.strip() for arg in args_str.split(',')]
            # Convert single quotes to double quotes
            # Parse portfolio JSON
            portfolio = json.loads(portfolio_str.replace("'", '"'))
            result_tool = financial_tools.recommend_portfolio_adjustments(portfolio, risk_tolerance)
            return "\n".join(result_tool)

        # Tool: analyze_portfolio_diversification
        elif chosen_tool == "analyze_portfolio_diversification":
            # Expect args_str in JSON format: '{"AAPL": 40, "GOOGL": 30, "SPY": 30}'
            portfolio = json.loads(args_str.replace("'", '"'))  # Convert single quotes to double quotes
            result_tool = financial_tools.analyze_portfolio_diversification(portfolio)
            return json.dumps(result_tool, indent=2)

        # Tool: calculate_expected_portfolio_return
        elif chosen_tool == "calculate_expected_portfolio_return":
            # Expect args_str in JSON format: '{"AAPL": 50, "GOOGL": 30, "SPY": 20}'
            portfolio = json.loads(args_str.replace("'", '"'))  # Convert single quotes to double quotes
            result_tool = financial_tools.calculate_expected_portfolio_return(portfolio)
            return f"Expected Portfolio Return: {result_tool:.2f}%"

        # Handle unsupported tool names
        else:
            return f"Error: Tool '{chosen_tool}' is not implemented or unrecognized."

    except json.JSONDecodeError as e:
        return f"Error parsing arguments: {e}"
    except KeyError as e:
        return f"Missing required key in arguments: {e}"
    except Exception as e:
        logger.error(f"Error processing tool action '{chosen_tool}': {e}")
        return f"Error executing tool '{chosen_tool}': {str(e)}"


def agent_loop(max_iterations, system_prompt, query):
    """
    Execute agent interaction loop for portfolio analysis.

    Args:
        max_iterations (int): Maximum number of interaction iterations
        system_prompt (str): Initial system prompt for agent guidance
        query (str): Initial user query

    Returns:
        None: Prints analysis results
    """
    # Initialize Groq client
    api_key=os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY is not set in environment variables.")
        return

    client = Groq(api_key=api_key)
    financial_tools = FinancialTools()
    agent = Agent(client, system_prompt, financial_tools)
    
    tools = [
        'analyze_portfolio_diversification', 
        'calculate_expected_portfolio_return', 
        'recommend_portfolio_adjustments',
        'get_stock_risk_profile'
    ]
    
    next_prompt = query
    
    for _ in range(max_iterations):
        result = agent(next_prompt)
        print(result)

        # Check if the response contains 'Answer' and break the loop
        if "Answer" in result:
            break
        
        # Process any requested tool actions
        if "PAUSE" in result and "Action" in result:
            action_match = re.search(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            if action_match:
                chosen_tool, args_str = action_match.groups()
                observation = execute_tool_action(chosen_tool, args_str, financial_tools)
                next_prompt = f"Observation: {observation}"             
                print(next_prompt)
                continue

def main():
    """
    Main execution function demonstrating investment analysis capabilities.
    """
    
    # Initialize financial tools
    financial_tools = FinancialTools()
    print("Loading system prompt")
    system_prompt = load_system_prompt('system_prompt.txt')
    print("Initiating Agent loop")
    # agent_loop(max_iterations=5, system_prompt=system_prompt, query="Get risk profile for NVDA stock")
    # agent_loop(max_iterations=5, system_prompt=system_prompt, query="Recommend adjustments for a portfolio with 50% MRNA, 30% JNJ, 20% SPY with low risk tolerance")
    agent_loop(max_iterations=5, system_prompt=system_prompt, query="I want to analyze a portfolio with 40% AAPL, 30% GOOGL, 30% SPY")

if __name__ == "__main__":
    main()