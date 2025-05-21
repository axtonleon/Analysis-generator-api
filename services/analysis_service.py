# services/analysis_service.py
import pandas as pd
import numpy as np
import json
from typing import Optional, Any, Tuple, List, Dict
import re

import logging
logger = logging.getLogger(__name__)
from helpers import deep_convert_numpy_to_python
# Import functions/state access from other services
from . import llm_service
from . import visualization_service
from . import data_service
# Import the new helper function from data_service
from .data_service import _to_json_serializable # Import helper


import schemas # Import schemas for response structure

def execute_query(query: str, df: pd.DataFrame) -> Tuple[Optional[Any], Optional[str]]:
    """Executes a Pandas query string against a DataFrame."""
    if df is None: return None, "DataFrame not loaded."
    try:
        # Using a copy for safety
        allowed_builtins = {
            'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'range': range,
            'min': min, 'max': max, 'abs': abs, 'round': round, 'sum': sum,
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'isinstance': isinstance, 'issubclass': issubclass,
            'Exception': Exception, 'TypeError': TypeError, 'ValueError': ValueError,
        }
        limited_globals = {'pd': pd, 'df': df, '__builtins__': allowed_builtins, 'np': np}
        limited_locals = {'df': df}


        logger.info(f"Executing query (LLM generated): {query}")
        result = eval(query, limited_globals, limited_locals)
        logger.info(f"LLM generated query execution successful. Result type: {type(result)}")

        # The conversion to JSON serializable format now happens in the caller
        # when the result is assigned to the QueryResult.data field.
        return result, None

    except Exception as e:
        logger.error(f"Error executing query '{query}': {e}")
        error_message = str(e)

        # Check for specific aggregation errors related to dtype
        if "agg function failed" in error_message and "dtype->object" in error_message:
            match = re.search(r"\['([^']+)']\s*\.agg\(", query)
            column_name = match.group(1) if match else "a column"
            return None, (f"Error calculating statistics (e.g., mean, std) on {column_name}. "
                         f"The data type of this column seems to be text or mixed types, "
                         f"which is not suitable for this calculation. You may need to clean or convert this column to numeric.")
        elif "could not convert string to float" in error_message or "unsupported operand type" in error_message:
            return None, (f"Error during numeric calculation based on the query. "
                         f"This might be caused by attempting numeric operations on non-numeric data. "
                         f"Check the data types of the columns used in the calculation.")
        elif "name 'df' is not defined" in error_message:
             return None, "Query execution failed: Could not find the DataFrame (variable 'df')."
        elif "SyntaxError" in error_message:
             return None, f"Syntax error in query: {e}"
        else:
             # Generic error
            return None, f"Error executing query: {e}"


# NEW FUNCTION: Execute raw user-provided Pandas code
def execute_raw_pandas_query(query: str) -> schemas.QueryResult:
    """
    Executes a raw Pandas query string provided by the user.
    Returns a QueryResult schema object with data converted to JSON serializable format.
    """
    df = data_service.get_current_df() # Get the current DataFrame
    # Initialize QueryResult with the query string for context
    query_result = schemas.QueryResult(raw_result_str=f"Attempting to execute: {query[:200]}...")

    if df is None:
        query_result.error = "No data file has been uploaded and processed yet."
        return query_result

    raw_result = None # Variable to hold the result before JSON conversion
    query_error = None # Variable to hold potential error message

    try:
        # Using a copy for safety
        allowed_builtins = {
            'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'range': range,
            'min': min, 'max': max, 'abs': abs, 'round': round, 'sum': sum, # Common math/conversion
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted, # Iteration/functional
            'isinstance': isinstance, 'issubclass': issubclass, # Type checking (potentially risky, but useful)
            'Exception': Exception, 'TypeError': TypeError, 'ValueError': ValueError, # Basic error types
            # Add other minimal builtins if needed for common pandas operations, e.g.,
            # 'print': print # If you want print output (caution: could be large)
        }

        limited_globals = {'pd': pd, 'df': df, '__builtins__': allowed_builtins, 'np': np}
        limited_locals = {'df': df}


        logger.info(f"Executing raw query: {query}")
        raw_result = eval(query, limited_globals, limited_locals) # Get the raw result
        logger.info(f"Raw query execution successful. Result type: {type(raw_result)}")

        # --- Use the new helper to convert raw result to JSON serializable structure ---
        query_result.data = _to_json_serializable(raw_result)

        # Set is_dataframe/is_series flags based on the *original* raw result type
        query_result.is_dataframe = isinstance(raw_result, pd.DataFrame)
        query_result.is_series = isinstance(raw_result, pd.Series)

        query_result.raw_result_str = str(raw_result)[:1000] # Store a string preview of original result

    except Exception as e:
        logger.error(f"Error executing raw query: {e}")
        query_error = str(e)

        # Check for specific aggregation errors related to dtype
        if "agg function failed" in query_error and "dtype->object" in query_error:
            match = re.search(r"\['([^']+)']\s*\.agg\(", query)
            column_name = match.group(1) if match else "a column"
            query_result.error = (f"Error calculating statistics (e.g., mean, std) on {column_name}. "
                                 f"The data type of this column seems to be text or mixed types, "
                                 f"which is not suitable for this calculation. You may need to clean or convert this column to numeric.")
        elif "could not convert string to float" in query_error or "unsupported operand type" in query_error:
             query_result.error = (f"Error during numeric calculation based on the query. "
                                  f"This might be caused by attempting numeric operations on non-numeric data. "
                                  f"Check the data types of the columns used in the calculation.")
        elif "name 'df' is not defined" in query_error:
             query_result.error = "Query execution failed: Could not find the DataFrame (variable 'df')."
        elif "SyntaxError" in query_error:
             query_result.error = f"Syntax error in query: {e}"
        else:
             query_result.error = f"Error executing query: {e}"

        query_result.raw_result_str = f"Error: {query_result.error}" # Update raw result str on error

    return query_result


async def process_user_question(question: str) -> schemas.ProcessedQuestionResponse:
    """
    Processes a single user question against the currently loaded data,
    generating query results, visualization JSON, and explanation.
    Does NOT save images.
    """
    df = data_service.get_current_df() # Get the current DataFrame
    if df is None:
        logger.error("process_user_question called without a loaded DataFrame.")
        return schemas.ProcessedQuestionResponse(
             original_question=question,
             errors=["No data file has been uploaded and processed yet."]
        )

    columns = df.columns.tolist()
    response_payload = schemas.ProcessedQuestionResponse(
        original_question=question,
        query_result=schemas.QueryResult(),
        visualization_recommendation=schemas.VisualizationRecommendation(),
        visualization_data=schemas.VisualizationData(),
        natural_language_explanation=schemas.NaturalLanguageExplanation()
    )

    # 1. Interpret question into query
    interpreted_query = llm_service.text_to_pandas_query(question, columns)
    response_payload.interpreted_query = interpreted_query
    if not interpreted_query:
        response_payload.errors.append("Could not interpret question into a query.")
        response_payload.query_result.error = "Failed to generate query."
        response_payload.query_result.raw_result_str = "Failed to generate query."
        return response_payload
    logger.info(f"Interpreted Query: {interpreted_query}")

    # 2. Execute query (using the execute_query helper, which uses limited eval)
    raw_result_data, query_error = execute_query(interpreted_query, df)
    if query_error:
        response_payload.errors.append(f"Query execution error: {query_error}")
        response_payload.query_result.error = query_error
        response_payload.query_result.raw_result_str = f"Error: {query_error}"
        # Note: We still proceed to recommend viz/explain even on query error,
        # but the subsequent steps should handle the fact that raw_result_data is None
        # and query_error is set.
    else:
        # --- Use the new helper to convert raw result to JSON serializable structure ---
        response_payload.query_result.data = _to_json_serializable(raw_result_data)

        # Set is_dataframe/is_series flags based on the *original* raw result type
        response_payload.query_result.is_dataframe = isinstance(raw_result_data, pd.DataFrame)
        response_payload.query_result.is_series = isinstance(raw_result_data, pd.Series)

        response_payload.query_result.raw_result_str = str(raw_result_data)[:1000] # Still capture string preview of original result


    # 3. Recommend visualization (Pass potential error or None result to LLM)
    # Pass the *original* raw result data to the LLM for recommendation,
    # as the LLM prompt is designed to summarize the raw pandas result structure (head).
    viz_recommendation_str = llm_service.recommend_visualization(question, raw_result_data)
    response_payload.visualization_recommendation.chart_type_description = viz_recommendation_str
    extracted_chart_type = visualization_service.extract_chart_type(viz_recommendation_str)
    response_payload.visualization_recommendation.chart_type_extracted = extracted_chart_type
    if not viz_recommendation_str : response_payload.errors.append("Could not get visualization recommendation.")
    logger.info(f"Viz Recommendation: {extracted_chart_type}")

    # 4. Generate visualization JSON (Only if NO query error AND data is available AND viz is recommended)
    if not query_error and raw_result_data is not None: # Use original raw result for viz prep
        if extracted_chart_type and extracted_chart_type != "none":
             # Pass the *original* raw_result_data (DF/Series/etc.) to the transformation function
             transformed_viz_data, transform_error = visualization_service.transform_data_for_visualization(raw_result_data, viz_recommendation_str)
             if transform_error:
                 response_payload.errors.append(f"Data transformation for viz error: {transform_error}")
                 response_payload.visualization_data.error = transform_error
             elif transformed_viz_data:
                 plotly_json_str, render_error = visualization_service.generate_plotly_json(transformed_viz_data, viz_recommendation_str)
                 plotly_json_str = deep_convert_numpy_to_python(plotly_json_str)
                 
                 if render_error:
                     response_payload.errors.append(f"Visualization rendering error: {render_error}")
                     response_payload.visualization_data.error = render_error
                 else:
                     response_payload.visualization_data.plotly_json = plotly_json_str
                     logger.info("Plotly JSON generated for user question.")
             else:
                 error_msg = "Visualization data transformation resulted in no data or unsuitable data."
                 response_payload.errors.append(error_msg)
                 response_payload.visualization_data.error = error_msg
        elif extracted_chart_type == "none":
           response_payload.visualization_data.error = "No visualization recommended."
        elif not extracted_chart_type and viz_recommendation_str:
            response_payload.visualization_data.error = "Could not extract a specific chart type from recommendation."
    elif not query_error and raw_result_data is None: # Data is None, but no query error -> likely empty result
         response_payload.visualization_data.error = "Query returned no data for visualization."
    elif query_error: # Visualization skipped due to query error
         response_payload.visualization_data.error = "Visualization skipped due to query error."


    # 5. Generate natural language explanation (Only if NO query error)
    if not query_error:
        # Pass the original raw result data (or None) to the explanation LLM
        explanation = llm_service.explain_results_in_natural_language(question, raw_result_data)
        if explanation: response_payload.natural_language_explanation.explanation = explanation
        else:
            response_payload.errors.append("Could not generate natural language explanation.")
            response_payload.natural_language_explanation.error = "Failed to generate explanation."
    else: # Explanation skipped due to query error
        response_payload.natural_language_explanation.error = "Explanation skipped due to query error."
        response_payload.errors.append("Explanation skipped due to query error.")


    return response_payload