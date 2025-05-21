# services/data_service.py
import pandas as pd
import numpy as np # Import numpy
import io
from fastapi import UploadFile, HTTPException
from typing import Optional, Tuple, Any, List, Dict
import logging
import json
import re
from helpers import deep_convert_numpy_to_python
# Import functions/state access from other services
from . import llm_service
from . import visualization_service # Used for extract_chart_type etc.

import schemas # Import schemas for type hinting and structure

logger = logging.getLogger(__name__)

# --- Global State (for simplicity, NOT production-ready) ---
# This state is managed within the data_service
current_df: Optional[pd.DataFrame] = None
current_df_columns: List[str] = []
current_filename: Optional[str] = None # Store filename for context
# --- End Global State ---

# --- Helper functions to access global DataFrame state ---
def get_current_df_columns() -> List[str]:
    """Returns the columns of the currently loaded DataFrame."""
    global current_df_columns
    return current_df_columns

def get_current_df_preview() -> Optional[List[Dict[str, Any]]]:
    """Returns the head (preview) of the currently loaded DataFrame as a list of dicts."""
    global current_df
    if current_df is not None:
        # Ensure that NaN values are converted to None for JSON serialization
        return current_df.head().where(pd.notnull(current_df.head()), None).to_dict(orient='records')
    return None

def get_current_df() -> Optional[pd.DataFrame]:
    """Returns the currently loaded DataFrame."""
    global current_df
    return current_df

def get_current_filename() -> Optional[str]:
    """Returns the filename of the currently loaded DataFrame."""
    global current_filename
    return current_filename

# NEW HELPER FUNCTION: Convert query results to JSON serializable structure

def _to_json_serializable(result: Any) -> Any:
    """
    Converts various Pandas/NumPy result types into standard Python types
    suitable for JSON serialization. Handles DataFrames, Series, NumPy arrays,
    and scalar NumPy types. Converts pandas nulls (NaN, NaT) to Python None.
    """
    def _convert_scalar(x):
        if x is None:
            return None
        if isinstance(x, np.integer):  # NumPy integer
            return int(x)
        if isinstance(x, np.floating):  # NumPy float
            return float(x)
        if isinstance(x, np.bool_):  # NumPy bool
            return bool(x)
        if isinstance(x, (pd.Timestamp, np.datetime64)):  # date/time
            return str(x)
        return x

    if result is None:
        return None

    # DataFrame: replace NaNs with None, convert each cell, then to records
    if isinstance(result, pd.DataFrame):
        df = result.where(pd.notnull(result), None)
        return df.applymap(_convert_scalar).to_dict(orient="records")

    # Series: replace NaNs with None, convert each value, then to dict
    if isinstance(result, pd.Series):
        series = result.where(pd.notnull(result), None)
        processed = series.map(_convert_scalar)
        try:
            return {str(k): v for k, v in processed.items()}
        except Exception:
            return {str(k): processed.loc[k] for k in processed.index}

    # NumPy array: convert via Series, then to list
    if isinstance(result, np.ndarray):
        series = pd.Series(result).where(pd.notnull(result), None)
        processed = series.map(_convert_scalar)
        return processed.tolist()

    # NumPy scalar types
    if isinstance(result, np.integer):
        return int(result)
    if isinstance(result, np.floating):
        return float(result)
    if isinstance(result, np.bool_):
        return bool(result)
    if isinstance(result, (pd.Timestamp, np.datetime64)):
        return str(result)

    # Pandas nulls
    if pd.isna(result):
        return None

    # Fallback: if already JSON serializable, return; else convert to string
    try:
        json.dumps(result)
        return result
    except (TypeError, OverflowError):
        logger.warning(f"Result of type {type(result)} is not JSON serializable, converting to string.")
        return str(result)


# --- Query Execution Helper (Used in Initial Analysis) ---
def execute_query_local(query: str, df: pd.DataFrame) -> Tuple[Optional[Any], Optional[str]]:
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


        logger.info(f"Executing initial analysis query: {query}")
        result = eval(query, limited_globals, limited_locals)
        logger.info(f"Initial analysis query execution successful. Result type: {type(result)}")

        # The conversion to JSON serializable format now happens *after* this function returns,
        # when the result is assigned to item.query_result.data.
        return result, None

    except Exception as e:
        logger.error(f"Error executing query '{query}' during initial analysis: {e}")
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


async def load_and_initially_analyze_data(file: UploadFile) -> Tuple[Optional[pd.DataFrame], Optional[str], List[schemas.InitialAnalysisItem]]:
    """
    Loads a file, performs initial automated analyses, and generates results
    including Plotly JSON (no images saved). Updates global DataFrame state.
    """
    global current_df, current_df_columns, current_filename
    initial_analyses_results: List[schemas.InitialAnalysisItem] = []
    current_df = None # Reset global state on new upload
    current_df_columns = []
    current_filename = None # Reset filename

    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            # Use keep_default_na=False and na_values=[] to prevent default NaN conversion for object columns initially
            # This helps the LLM see the original strings like 'N/A', '?', etc.
            # The _to_json_serializable helper will handle converting NaNs to None for display/serialization.
            # Note: This might affect automatic type inference for purely numeric columns with missing values.
            # A more sophisticated approach might be needed for production, potentially type hinting specific columns.
            # For now, let's prioritize seeing original values for LLM prompt context.
            # Alternative: Let pandas handle NA naturally, and focus on LLM prompt to be aware of NA representation.
            # Let's revert to default pandas NA handling as it's more standard. The _to_json_serializable should handle it.
            current_df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.xlsx'):
             current_df = pd.read_excel(io.BytesIO(content))
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file.", []

        if current_df is not None:
            current_df_columns = current_df.columns.tolist()
            current_filename = file.filename

            # --- Perform initial analyses ---
            df_head_sample = current_df.head().to_string()
            analysis_ideas = llm_service.generate_analysis_ideas(current_df_columns, df_head_sample, max_ideas=3)

            for i, (title, idea_question) in enumerate(analysis_ideas):
                if len(initial_analyses_results) >= 5: break

                logger.info(f"\n--- Processing initial analysis idea: {title} ---")
                logger.info(f"Suggested Question: {idea_question}")

                item = schemas.InitialAnalysisItem(
                    analysis_title=title,
                    suggested_question=idea_question,
                    query_result=schemas.QueryResult(),
                    visualization_recommendation=schemas.VisualizationRecommendation(),
                    visualization_data=schemas.VisualizationData(), # Will now only include plotly_json
                    natural_language_explanation=schemas.NaturalLanguageExplanation()
                )

                # Interpret question into query
                interpreted_query = llm_service.text_to_pandas_query(idea_question, current_df_columns)
                item.interpreted_query = interpreted_query
                if not interpreted_query:
                    item.errors.append("Failed to generate Pandas query for this idea.")
                    initial_analyses_results.append(item)
                    continue
                logger.info(f"Interpreted Query: {interpreted_query}")

                # Execute query (using the local helper)
                query_result_data, query_error = execute_query_local(interpreted_query, current_df)
                if query_error:
                    item.errors.append(f"Query execution error: {query_error}")
                    item.query_result.error = query_error
                    item.query_result.raw_result_str = f"Error: {query_error}" # Error message as raw result
                    # Continue to allow adding to results list, but skip viz/explanation steps below
                else:
                    # --- Use the new helper to convert result to JSON serializable structure ---
                    item.query_result.data = _to_json_serializable(query_result_data)

                    # Set is_dataframe/is_series flags based on the *original* result type
                    item.query_result.is_dataframe = isinstance(query_result_data, pd.DataFrame)
                    item.query_result.is_series = isinstance(query_result_data, pd.Series)

                    item.query_result.raw_result_str = str(query_result_data)[:1000] # Still capture string preview of original result
                    logger.info(f"Query Result (sample): {str(query_result_data)[:200]}")


                # Recommend visualization (Pass potential error or None result to LLM)
                # Pass the *processed JSON serializable* data to the LLM recommendation, as this is what the frontend will see
                viz_recommendation_str = llm_service.recommend_visualization(idea_question, item.query_result.data)
                item.visualization_recommendation.chart_type_description = viz_recommendation_str
                extracted_chart_type = visualization_service.extract_chart_type(viz_recommendation_str)
                item.visualization_recommendation.chart_type_extracted = extracted_chart_type
                if not viz_recommendation_str : item.errors.append("Could not get visualization recommendation.")
                logger.info(f"Viz Recommendation: {extracted_chart_type}")

                # Generate visualization JSON (Only if NO query error AND data is available AND viz is recommended)
                if not query_error and query_result_data is not None: # Use original result for viz prep
                    if extracted_chart_type and extracted_chart_type != "none":
                         # Pass the *original* query_result_data (DF/Series/etc.) to the transformation function
                         transformed_viz_data, transform_error = visualization_service.transform_data_for_visualization(query_result_data, viz_recommendation_str)
                         if transform_error:
                             item.errors.append(f"Data transformation for viz error: {transform_error}")
                             item.visualization_data.error = transform_error
                         elif transformed_viz_data:
                             plotly_json_str, render_error = visualization_service.generate_plotly_json(transformed_viz_data, viz_recommendation_str)
                             if render_error:
                                 item.errors.append(f"Visualization rendering error: {render_error}")
                                 item.visualization_data.error = render_error
                             else:
                                 item.visualization_data.plotly_json = plotly_json_str
                                 logger.info("Plotly JSON generated for initial analysis.")
                         else:
                             error_msg = "Visualization data transformation resulted in no data or unsuitable data."
                             item.errors.append(error_msg)
                             item.visualization_data.error = error_msg
                    elif extracted_chart_type == "none":
                       item.visualization_data.error = "No visualization recommended."
                    elif not extracted_chart_type and viz_recommendation_str:
                        item.visualization_data.error = "Could not extract a specific chart type from recommendation."
                elif not query_error and query_result_data is None: # Data is None, but no query error -> likely empty result
                     item.visualization_data.error = "Query returned no data for visualization."
                elif query_error: # Visualization skipped due to query error
                     item.visualization_data.error = "Visualization skipped due to query error."


                # Generate natural language explanation (Only if NO query error)
                if not query_error:
                    # Pass the original raw result data (or None) to the explanation LLM
                    explanation = llm_service.explain_results_in_natural_language(idea_question, query_result_data)
                    if explanation: item.natural_language_explanation.explanation = explanation
                    else:
                        item.errors.append("Could not generate natural language explanation.")
                        item.natural_language_explanation.error = "Failed to generate explanation."
                else: # Explanation skipped due to query error
                    item.natural_language_explanation.error = "Explanation skipped due to query error."
                    item.errors.append("Explanation skipped due to query error.")


                initial_analyses_results.append(item)

        return current_df, None, initial_analyses_results
    except Exception as e:
        logger.error(f"Error in load_and_initially_analyze_data: {e}")
        current_df = None
        current_df_columns = []
        current_filename = None
        return None, f"Error processing file and performing initial analysis: {e}", []
    
def get_detailed_column_info() -> schemas.ColumnDetailsResponse:
    """
    Generates detailed information for each column in the currently loaded DataFrame.
    """
    df = get_current_df()
    filename = get_current_filename()
    response = schemas.ColumnDetailsResponse(filename=filename)

    if df is None:
        response.error = "No data file has been uploaded and processed yet."
        return response

    response.total_rows = len(df)
    details_list: List[schemas.ColumnDetails] = []

    for col in df.columns:
        try:
            col_series = df[col]
            col_dtype = str(col_series.dtype)
            non_null_count = col_series.count() # .count() excludes NaNs
            unique_count = None
            basic_stats = None

            # Calculate unique_count (include NaNs)
            try:
                 unique_count_raw = col_series.nunique(dropna=False)
                 # Convert np.integer unique count to standard int, handle None if df is empty
                 if isinstance(unique_count_raw, np.integer):
                     unique_count = int(unique_count_raw)
                 elif unique_count_raw is not None: # Should be an int/float for non-empty series
                      unique_count = unique_count_raw
            except Exception as nunique_e:
                 logger.warning(f"Could not get unique count for column '{col}': {nunique_e}")
                 # Continue without unique_count


            # Calculate basic stats using .describe()
            try:
                stats_series = col_series.describe()
                stats_dict_raw = stats_series.to_dict() # Get as standard Python dict

                # --- Apply a robust conversion to ensure JSON serializability ---
                # Iterate through the raw stats dictionary and convert values
                basic_stats_converted: Dict[str, Any] = {}
                for k, v in stats_dict_raw.items():
                    # Use the _to_json_serializable helper for individual values
                    basic_stats_converted[str(k)] = _to_json_serializable(v)

                basic_stats = basic_stats_converted

            except Exception as describe_e:
                logger.warning(f"Could not get detailed stats for column '{col}': {describe_e}")
                basic_stats = {"error": f"Could not generate stats: {describe_e}"}


            details_list.append(schemas.ColumnDetails(
                name=col,
                dtype=col_dtype,
                non_null_count=non_null_count,
                unique_count=unique_count,
                basic_stats=basic_stats
            ))
        except Exception as e:
            logger.error(f"Error processing details for column '{col}': {e}")
            details_list.append(schemas.ColumnDetails(
                name=col,
                dtype="error",
                non_null_count=0,
                unique_count=None,
                basic_stats={"error": f"Could not process column details: {e}"} # Ensure basic_stats is a dict on error
            ))

    response.columns = details_list
    return response