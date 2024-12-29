"""
Investment Portfolio Risk Assessment Tool

A sophisticated financial analysis module for comprehensive
portfolio risk assessment and intelligent recommendations.

Dependencies:
- groq
- python-dotenv
- typing
- dataclasses
- pathlib

Author: Shivang Rana
Created: 11-27-2024
Version: 1.0.0
"""

import os
import json
import re
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

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
        self.loaded_stock_data = self.load_stock_data()

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

        # Calculate portfolio beta and volatility
        portfolio_beta = 0
        portfolio_volatility = 0

        for stock, allocation in portfolio.items():
            stock_info = self.loaded_stock_data.get(stock, {})
            if stock_info:
                sector = stock_info.sector
                risk_level = stock_info.estimated_risk
                weight = allocation/ total_allocation

                sectors[sector] = sectors.get(sector, 0) + allocation
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + allocation

                # Weighted beta and volatility
                portfolio_beta += stock_info.beta * weight
                portfolio_volatility += stock_info.volatility_index * weight
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        sector_hhi = sum((v/total_allocation)**2 for v in sectors.values())

        hhi_thresholds = {
            "highly_diversified": 0.15,
            "moderately_concentrated": 0.25,
        }

        hhi_interpretation = (
            "Highly Diversified" if sector_hhi < hhi_thresholds["highly_diversified"]
            else "Moderately Concentrated" if sector_hhi <= hhi_thresholds["moderately_concentrated"]
            else "Highly Concentrated"
        )

        return {
            "total_allocation": total_allocation,
            "sector_breakdown": sectors,
            "risk_level_breakdown": risk_levels,
            "portfolio_metrics":{
                "portfolio_beta": round(portfolio_beta, 2),
                "portfolio_volatility": round(portfolio_volatility, 2),
                "sector_concentration_hhi": round(sector_hhi, 2),
                "hhi_interpretation": hhi_interpretation,
            }
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
            stock_info = self.loaded_stock_data.get(stock, {})
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
        if not portfolio or sum(portfolio.values()) == 0:
            return ["Portfolio is empty or allocations sum to zero. Please review the input."]

        current_analysis = self.analyze_portfolio_diversification(portfolio)
        recommendations = []

        # Define target allocations based on risk tolerance
        target_allocations = {
            "low": {"Low": 60, "Medium":30, "High": 10},
            "medium": {"Low": 30, "Medium":50, "High": 20},
            "high": {"Low": 10, "Medium":40, "High": 50},
        }
        target_risk = target_allocations.get(risk_tolerance, {})
        if not target_risk:
            return [f"Invalid risk tolerance: {risk_tolerance}. Choose from 'low', 'medium', or 'high'."]

        # Portfolio metrics analysis
        portfolio_metrics = current_analysis["portfolio_metrics"]
        sector_concentration = portfolio_metrics["sector_concentration_hhi"]

        # For sector threshold allocations
        sector_benchmarks = {"Technology": 28, "Healthcare": 15, "Energy": 8, "Financials": 12} # Based on S&P 500
        risk_tolerance_thresholds = {"low": 20, "medium": 30, "high": 40}
        sector_volatility = {"Technology": 0.25, "Healthcare": 0.15, "Energy": 0.30, "Financials": 0.20}
        threshold = risk_tolerance_thresholds.get(risk_tolerance, 30)

        # Concentration Warnings
        if sector_concentration > 0.45:
            recommendations.append(f"High sector concentration detected (HHI: {sector_concentration:.2f}). Consider diversifying across sectors")
            # Specific sector suggestions
            for sector, allocation in current_analysis["sector_breakdown"].items():
                sector_pct = (allocation / current_analysis["total_allocation"]) * 100
                benchmark_weight = sector_benchmarks.get(sector, 10)
                volatility = sector_volatility.get(sector, 0.2)
                
                # Final adaptive threshold
                adaptive_threshold = min(
                    threshold,
                    benchmark_weight * 1.2,
                    40 - (volatility * 100)  # Adjust for volatility
                )
 
                if sector_pct > adaptive_threshold:
                    recommendations.append(f"Reduce exposure to {sector} (currently {sector_pct:.1f}%, target ≤ {adaptive_threshold:.1f}%).")
                    
        # Risk allocation analysis
        current_risk_levels = current_analysis["risk_level_breakdown"]
        for risk_level, target_pct in target_risk.items():
            current_pct = (current_risk_levels.get(risk_level, 0) / current_analysis["total_allocation"]) * 100
            if abs(current_pct - target_pct) > 15:
                if current_pct > target_pct:
                    recommendations.append(f"Reduce {risk_level} - risk allocation from {current_pct:.1f}% to near {target_pct}%.")
                else:
                    recommendations.append(f"Increase {risk_level} - risk allocation from {current_pct:.1f}% to near {target_pct}%.")

        # Beta threshold recommendations
        portfolio_beta = portfolio_metrics["portfolio_beta"]
        beta_threshold = {"low": 1.0, "medium": 1.2, "high":1.5}
        if portfolio_beta > beta_threshold[risk_tolerance]:
            recommendations.append(f" Portfolio beta ({portfolio_beta:.2f}) exceeds target for {risk_tolerance} risk tolerance. Reduce high-beta stocks")       

        return recommendations

    def get_stock_risk_profile(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Get detailed risk profile for a specific stock.

        Args:
            stock_symbol (str): Stock symbol to retrieve risk profile for.

        Returns:
            Dict[str, Any]: Comprehensive stock risk profile with advanced metrics.
        """
        stock_data = self.loaded_stock_data.get(stock_symbol, {})
        if not stock_data:
            similar_stocks = [symbol for symbol in self.loaded_stock_data.keys() if stock_data.lower in symbol.lower]
            return {
                "Error": f"Stock symbol {stock_symbol} not found in dataset.",
                "Suggestions": similar_stocks[:3] if similar_stocks else "No similar stocks found."
            }

        risk_explanation = (
        f"{'High' if stock_data.volatility_index > 0.35 else 'Moderate' if stock_data.volatility_index > 0.2 else 'Low'} "
        f"volatility stock with an estimated risk level of {stock_data.estimated_risk}. Suitable for "
        f"{'aggressive' if stock_data.volatility_index > 0.35 else 'moderate' if stock_data.volatility_index > 0.2 else 'conservative'} investors."
    )

        relative_market_size = "Large Cap" if stock_data.market_cap > 200_000_000_000 else \
                           "Mid Cap" if stock_data.market_cap > 50_000_000_000 else "Small Cap"

        risk_reward_ratio = round(stock_data.avg_annual_return / stock_data.volatility_index, 2) if stock_data.volatility_index else "N/A"
        beta_interpretation = (
            "Highly Volatile" if stock_data.beta > 1.5 else
            "Moderately Volatile" if stock_data.beta > 1 else
            "Stable"
        )

        return {
            "Basic Information": {
                "Name": stock_data.name,
                "Sector": stock_data.sector,
                "Market Cap": f"${stock_data.market_cap / 1e9:.2f} Billion",
                "Market Cap Category": relative_market_size
            },
            "Risk Metrics": {
                "Risk Level": stock_data.estimated_risk,
                "Beta": stock_data.beta,
                "Beta Interpretation": beta_interpretation,
                "Volatility Index": stock_data.volatility_index,
                "Risk/Reward Ratio": risk_reward_ratio
            },
            "Performance Metrics": {
                "Average Annual Return": f"{stock_data.avg_annual_return * 100:.2f}%",
                "Market Performance": "Above Market" if stock_data.beta > 1 else "Below Market",
            },
            "Risk Analysis": {
                "Risk Explanation": risk_explanation
            }
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
            financial_tools (FinancialTools): Financial tools helper
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


def load_system_prompt(file_name: str = 'system_prompt_v1.txt') -> str:
    """
    Load system prompt from a text file

    Args:
        file_name (str): File path to system prompt txt file
    
    Returns:
        str: system_prompt
    """
    try:
        # Get the current file's directory
        current_dir = Path(__file__).parent
        prompt_path = current_dir / file_name
        
        return prompt_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        logger.warning("System prompt file not found. Using default.")
        return """You are an Investment Portfolio Analysis Agent..."""
    except Exception as e:
        logger.error(f"Error loading system prompt: {str(e)}")
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

    next_prompt = query
    for iteration in range(max_iterations):
        result = agent(next_prompt)
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}:")
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
    system_prompt = load_system_prompt('system_prompt_v1.txt')
    agent_loop(max_iterations=5, system_prompt=system_prompt, query="I want to analyze a portfolio with 40% AAPL, 30% GOOGL, 30% SPY")

if __name__ == "__main__":
    main()
