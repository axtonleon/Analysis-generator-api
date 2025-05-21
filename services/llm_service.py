# services/llm_service.py
import logging
from typing import List, Tuple, Optional, Any
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

logger = logging.getLogger(__name__)

# --- LLM Model Initialization (Gemini) ---
model = None
try:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)
    logger.info("Google Gemini Pro model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Google Gemini model: {e}")

def generate_analysis_ideas(columns: List[str], df_head_sample: str, max_ideas: int = 5) -> List[Tuple[str, str]]:
    """Suggests analytical questions based on dataframe columns and sample."""
    if not model:
        logger.warning("LLM model not initialized. Cannot generate analysis ideas.")
        return []
    prompt_template = PromptTemplate.from_template(
        """
        You are a data analyst AI. Your task is to suggest interesting analytical questions about a dataset.
        The dataset has the following columns: {columns}
        Here's a small sample of the data (first few rows):
        {df_head_sample}
        Based on this, suggest up to {max_ideas} distinct and insightful analytical questions a user might ask.
        For each suggestion, provide a concise title for the analysis and the question itself.
        Format EACH suggestion strictly as:
        Title: [A short, descriptive title for this analysis]
        Question: [The analytical question itself, phrased naturally]
        ---
        (Ensure a '---' separator between suggestions if providing multiple)
        Example:
        Title: Customer Age Distribution
        Question: What is the distribution of customer ages?
        ---
        Focus on questions that can likely be answered using Pandas operations and visualized. Avoid overly complex or vague questions.
        """
    )
    prompt_str = prompt_template.format(columns=", ".join(columns), df_head_sample=df_head_sample, max_ideas=max_ideas)
    ideas = []
    try:
        response = model.invoke(prompt_str)
        content = response.content.strip()
        logger.info(f"\nLLM Raw Response for Analysis Ideas:\n{content}\n")

        suggestions = content.split('---')
        for suggestion_block in suggestions:
            suggestion_block = suggestion_block.strip()
            if not suggestion_block: continue
            title_match = re.search(r"Title:\s*(.+)", suggestion_block, re.IGNORECASE | re.DOTALL)
            question_match = re.search(r"Question:\s*(.+)", suggestion_block, re.IGNORECASE | re.DOTALL)
            if title_match and question_match:
                title = title_match.group(1).strip()
                question = question_match.group(1).strip()
                if title and question: ideas.append((title, question))
            else: logger.warning(f"Could not parse title/question from block: '{suggestion_block[:100]}...'")
        return ideas[:max_ideas]
    except Exception as e:
        logger.error(f"Error generating analysis ideas with Gemini: {e}")
        return []

def text_to_pandas_query(question: str, columns: List[str]) -> Optional[str]:
    """Converts a natural language question into a Pandas query string."""
    if not model:
        logger.warning("LLM model not initialized. Cannot generate pandas query.")
        return None
    prompt_template = PromptTemplate.from_template(
        """
        You are a data analysis expert. Given an input question, generate a Pandas-compatible query to answer it.

        Available columns: {columns}

        Question: {question}

        Guidelines:
        1. Use Pandas syntax (e.g., `df.groupby()` or `df.query()`).
        2. Return data (e.g., a DataFrame or Series) instead of a plot.
        3. Always include relevant columns in the output.

        Provide only the final Pandas query as plain text without any formatting.
    """
    )
    prompt_str = prompt_template.format(columns=", ".join(columns), question=question)
    try:
        response = model.invoke(prompt_str)
        query_text = response.content.strip()
        # Clean up potential markdown code blocks often returned by LLMs
        if query_text.startswith("```python"): query_text = query_text[len("```python"):].strip()
        elif query_text.startswith("```"): query_text = query_text[len("```"):].strip()
        if query_text.endswith("```"): query_text = query_text[:-len("```")].strip()

        # Additional check: if the LLM includes "df = ..." or "import pandas as pd"
        lines = query_text.splitlines()
        cleaned_lines = [line for line in lines if not line.strip().startswith("import pandas as pd")]
        query_text = "\n".join(cleaned_lines).strip()
        if "df =" in query_text:
             # Simple heuristic: take everything after the last "df =" assignment if multiple lines
             # This is fragile but common in LLM outputs. A proper parser would be better.
             query_text = query_text.split("df =")[-1].strip()

        return query_text
    except Exception as e:
        logger.error(f"Error generating pandas query with Gemini: {e}")
        return None

def recommend_visualization(question: str, result: Any) -> Optional[str]:
    """Recommends a visualization type based on the question and query results."""
    if not model:
        logger.warning("LLM model not initialized. Cannot recommend visualization.")
        return "Recommended Visualization: none\nReason: LLM model not available."

    result_summary = ""
    if isinstance(result, pd.DataFrame): result_summary = result.head().to_string()
    elif isinstance(result, pd.Series): result_summary = result.head().to_string()
    else: result_summary = str(result)
    result_summary = result_summary[:500] # Limit length for prompt

    prompt_template = PromptTemplate.from_template(
        """
        You are an AI assistant specializing in recommending appropriate data visualizations for customer and address analytics. Based on the user's question and query results, suggest the most suitable type of graph or chart to visualize the data.

        Available chart types and their ideal use cases:

        - Bar Graphs (for 3+ categories):
          * Compare distributions across multiple categories (e.g., customer counts by region).
          * Analyze demographics (e.g., age groups, monthly registration counts).
          * Identify patterns in categorical data with 3 or more distinct categories.

        - Grouped Bar Graphs (for comparing multiple categories across groups):
          * Compare multiple categories across different groups (e.g., sales by product category across regions).
          * Use when the data has multiple columns for different categories or groups.

        - Horizontal Bar Graphs (for 2-3 categories or large value disparities):
          * Compare binary or small category sets (e.g., gender distribution).
          * Highlight significant differences between 2-3 categories.

        - Line Graphs (for time series data):
          * Show trends over time (e.g., registration growth by month).
          * Analyze any metric with a time-based X-axis (e.g., create_timestamp).
          Note: The X-axis must represent time.

        - Pie Charts (for proportions, 3-7 categories):
          * Visualize distribution percentages (e.g., market share or demographic proportions).
          Note: The total must sum to 100%.

        - Scatter Plots (for numeric relationships):
          * Identify relationships between two numeric variables (e.g., age vs registration count).
          * Analyze non-categorical data points.
          Note: Both axes must be numeric.

        Special Cases:
        1. Geographic Data:
           * If the result contains latitude and longitude → No chart (recommend a map visualization).
           * For questions about addresses or locations → No chart (recommend a map visualization).

        2. Tabular Data (Raw Results):
           * For individual customer records → No chart (tabular display only).
           * For non-aggregated raw data → No chart (tabular display only).

        Question: {question}
        Query Results Summary (First few rows/items):
        {result_summary}

        Recommended Visualization: [Chart type or "none"]
        Reason: [Brief explanation for your recommendation]
        """
    )
    prompt_str = prompt_template.format(question=question, result_summary=result_summary)
    try:
        response = model.invoke(prompt_str)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error recommending visualization with Gemini: {e}")
        return "Recommended Visualization: none\nReason: Error during recommendation."


def explain_results_in_natural_language(question: str, result: Any) -> Optional[str]:
    """Explains query results in natural language."""
    if not model:
        logger.warning("LLM model not initialized. Cannot generate explanation.")
        return None

    result_summary = ""
    if isinstance(result, pd.DataFrame): result_summary = result.head(10).to_string()
    elif isinstance(result, pd.Series): result_summary = result.head(10).to_string()
    else: result_summary = str(result)
    result_summary = result_summary[:1000] # Limit length for prompt

    prompt_template = PromptTemplate.from_template(
        """
        You are a data analysis expert. Given a user's question and the corresponding query results (or a summary of them), explain the results in simple, natural language.
        Be concise and clear. If the results are empty or an error occurred, state that.
        Question: {question}
        Query Results (summary):
        {result_summary}
        Explanation:
        """
    )
    prompt_str = prompt_template.format(question=question, result_summary=result_summary)
    try:
        response = model.invoke(prompt_str)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating natural language explanation with Gemini: {e}")
        return None