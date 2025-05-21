# services/visualization_service.py
import pandas as pd
import plotly.express as px
import json
import re # For extracting chart type
from typing import Optional, Dict, Any, Tuple, List, Union
import schemas
import numpy as np
from . import data_service
from helpers import deep_convert_numpy_to_python

import logging
logger = logging.getLogger(__name__)


def extract_chart_type(chart_type_str: str) -> Optional[str]:
    """Extracts a standardized chart type string from the LLM recommendation."""
    if not chart_type_str: return None
    rec_line = next((line for line in chart_type_str.split('\n') if "Recommended Visualization:" in line), None)
    if not rec_line: return None
    chart_type_val = rec_line.split("Recommended Visualization:")[1].strip().lower()
    if "grouped bar" in chart_type_val: return "grouped_bar"
    if "horizontal bar" in chart_type_val: return "bar" # Group horizontal under generic bar for Plotly Express
    if "bar" in chart_type_val: return "bar"
    if "line" in chart_type_val: return "line"
    if "pie" in chart_type_val: return "pie"
    if "scatter" in chart_type_val: return "scatter"
    if any(keyword in chart_type_val for keyword in ["none", "map", "tabular"]): return "none"
    return None

def prepare_data_for_plotly(
    df: pd.DataFrame,
    chart_type: str,
    x_col_name: Optional[str] = None,
    y_col_name: Optional[str] = None,
    color_col_name: Optional[str] = None,
    size_col_name: Optional[str] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extracts specified columns from a DataFrame and formats them for Plotly Express.
    Handles different chart types and data types.
    """
    if df is None or df.empty:
        return None, "DataFrame is empty or not loaded."

    # Ensure specified columns exist
    all_cols = df.columns.tolist()
    required_cols = []
    if x_col_name: required_cols.append(x_col_name)
    if y_col_name: required_cols.append(y_col_name)
    if color_col_name: required_cols.append(color_col_name)
    if size_col_name: required_cols.append(size_col_name)

    for col_name in required_cols:
        if col_name not in all_cols:
            return None, f"Specified column '{col_name}' not found in the DataFrame."

    # Select only the relevant columns and handle potential non-serializable types and NaNs
    # Work on a copy to avoid modifying the global df
    df_viz = df[required_cols].copy()

    # Convert specific non-serializable types like Timestamps to strings
    df_viz = df_viz.applymap(lambda x: str(x) if isinstance(x, pd.Timestamp) else x, na_action='ignore')

    # Convert pandas nulls (NaN, NaT, None) to Python None for JSON serialization
    # This step is crucial *before* passing to Plotly.js
    df_viz = df_viz.where(pd.notnull(df_viz), None)

    data_for_plotly: Dict[str, Any] = {}
    error = None

    try:
        if chart_type == "grouped_bar":
             if not x_col_name or not y_col_name or not color_col_name:
                 return None, "Grouped bar chart requires X, Y (value), and Color (series) column names."

             # Ensure Y column is numeric, coerce errors to None and drop
             # FIX: Explicitly create Series before pd.to_numeric
             if y_col_name:
                try:
                    numeric_y_series = pd.to_numeric(pd.Series(df_viz[y_col_name]), errors='coerce')
                    # Drop rows where value became NaN, or where X/Color is also None
                    # Ensure we use the index from the filtered numeric series
                    valid_indices = numeric_y_series.dropna().index
                    if x_col_name: valid_indices = valid_indices.intersection(df_viz[x_col_name].dropna().index)
                    if color_col_name: valid_indices = valid_indices.intersection(df_viz[color_col_name].dropna().index)

                    df_viz = df_viz.loc[valid_indices].copy()
                    df_viz[y_col_name] = numeric_y_series.loc[valid_indices] # Update Y column with numeric values

                    if df_viz.empty:
                         return None, "Grouped bar chart data is empty after filtering for non-null numeric values."

                except Exception as num_e:
                     return None, f"Error ensuring numeric data for grouped bar Y-axis ('{y_col_name}'): {num_e}"


             data_for_plotly = {"data_frame": df_viz.to_dict(orient='list'), "x": x_col_name, "y": y_col_name, "color": color_col_name}
             data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])


        elif chart_type in ["bar", "line", "scatter"]:
            if not x_col_name or not y_col_name:
                return None, f"{chart_type.capitalize()} chart requires X and Y column names."

            # Ensure Y is numeric for scatter plots, coerce errors to None and drop
            if chart_type == "scatter":
                # FIX: Explicitly create Series before pd.to_numeric
                try:
                    numeric_y_series = pd.to_numeric(pd.Series(df_viz[y_col_name]), errors='coerce')
                    # Drop rows where Y became NaN after coercion, or where X is None
                    valid_indices = numeric_y_series.dropna().index.intersection(df_viz[x_col_name].dropna().index)

                    df_viz = df_viz.loc[valid_indices].copy() # Filter df_viz
                    df_viz[y_col_name] = numeric_y_series.loc[valid_indices] # Update Y column with numeric values

                    if df_viz.empty:
                        return None, "Scatter plot requires two columns with corresponding non-null numeric Y values."

                except Exception as num_e:
                    return None, f"Error ensuring numeric data for scatter plot Y-axis ('{y_col_name}'): {num_e}"

            # Ensure size column is numeric for scatter plots, coerce errors to None and drop
            if chart_type == "scatter" and size_col_name:
                 # FIX: Explicitly create Series before pd.to_numeric
                 try:
                     numeric_size_series = pd.to_numeric(pd.Series(df_viz[size_col_name]), errors='coerce')
                     # Drop rows where size became NaN after coercion, or where X/Y is None (already handled, but intersect again)
                     valid_indices = numeric_size_series.dropna().index
                     if x_col_name: valid_indices = valid_indices.intersection(df_viz[x_col_name].dropna().index) # Re-intersect with non-null X/Y
                     if y_col_name: valid_indices = valid_indices.intersection(df_viz[y_col_name].dropna().index)


                     df_viz = df_viz.loc[valid_indices].copy() # Filter df_viz based on valid size
                     df_viz[size_col_name] = numeric_size_series.loc[valid_indices] # Update size column with numeric values

                     if df_viz.empty: # If filtering size removed all data, but x/y were valid
                          return None, "Size column contains only non-numeric or null values after coercion."

                     data_for_plotly["size"] = size_col_name
                     data_for_plotly["data_frame"] = df_viz.to_dict(orient='list') # Update data_frame after filtering size
                     data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])

                 except Exception as num_e:
                    return None, f"Error ensuring numeric data for size encoding ('{size_col_name}'): {num_e}"


            # For bar/line, Y should ideally be numeric, coerce errors to None but don't necessarily drop rows unless X is also None
            if chart_type in ["bar", "line"]:
                 # FIX: Explicitly create Series before pd.to_numeric
                 try:
                     numeric_y_series = pd.to_numeric(pd.Series(df_viz[y_col_name]), errors='coerce')
                     # Keep rows where X is not null OR Y is not null after coercion
                     valid_indices = df_viz[x_col_name].dropna().index.union(numeric_y_series.dropna().index)

                     df_viz = df_viz.loc[valid_indices].copy()
                     # Update Y column with coerced numeric values (including None)
                     df_viz[y_col_name] = numeric_y_series.loc[valid_indices].where(pd.notnull(numeric_y_series.loc[valid_indices]), None)

                     if df_viz[y_col_name].isnull().sum() > len(df_viz) * 0.5:
                          logger.warning(f"Y-axis for {chart_type} ('{y_col_name}') seems mostly non-numeric or null after coercion. Proceeding anyway.")

                 except Exception as num_e:
                     logger.warning(f"Could not fully numeric-coerce Y-axis for {chart_type} ('{y_col_name}'): {num_e}. Proceeding with original data.")
                     # Fallback: use the original data (with Timestamp/NaN->str/None conversion applied earlier)
                     # No specific update needed here as df_viz already has the original data for these columns

            # Use column names directly for Plotly Express
            data_for_plotly["data_frame"] = df_viz.to_dict(orient='list') # Ensure data_frame is based on potentially filtered df_viz
            data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])
            data_for_plotly["x"] = x_col_name
            data_for_plotly["y"] = y_col_name
            if color_col_name:
                 if color_col_name not in df_viz.columns: # Should be caught earlier, but double check
                      return None, f"Color column '{color_col_name}' not found in the selected data after filtering."
                 data_for_plotly["color"] = color_col_name


        elif chart_type == "pie":
            # Pie charts typically need 'names' (categories) and 'values' (numeric)
            # Using x_col_name for names, y_col_name for values convention
            if not x_col_name or not y_col_name:
                return None, "Pie chart requires column names for Categories (Names) and Values."

            # Ensure Y (values) is numeric, coerce errors to None and drop
            # FIX: Explicitly create Series before pd.to_numeric
            try:
                numeric_values_series = pd.to_numeric(pd.Series(df_viz[y_col_name]), errors='coerce')
                # Drop rows where value became NaN or where name is also None
                valid_indices = numeric_values_series.dropna().index.intersection(df_viz[x_col_name].dropna().index)

                df_viz = df_viz.loc[valid_indices].copy() # Filter df_viz
                df_viz[y_col_name] = numeric_values_series.loc[valid_indices] # Update Y column with numeric values

                if df_viz.empty:
                     return None, "Pie chart requires non-null category names and numeric values."

                # For pie chart, Plotly Express takes dataframe and column names for 'names' and 'values'
                data_for_plotly = {"data_frame": df_viz.to_dict(orient='list'), "names": x_col_name, "values": y_col_name}
                data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])

            except Exception as num_e:
                 return None, f"Error ensuring numeric data for pie chart values ('{y_col_name}'): {num_e}"

        else:
            return None, f"Unsupported chart type '{chart_type}' for manual visualization generation."

        # Ensure the data_frame key is present if other keys are set (should be already)
        # if 'data_frame' not in data_for_plotly and not df_viz.empty:
        #      data_for_plotly['data_frame'] = df_viz.to_dict(orient='list')


    except Exception as e:
        logger.error(f"Error preparing data for Plotly for chart type {chart_type}: {e}")
        return None, f"Error preparing data for visualization: {e}"

    if not data_for_plotly or 'data_frame' not in data_for_plotly or pd.DataFrame(data_for_plotly['data_frame']).empty:
        return None, "Data preparation failed or resulted in no data after processing."
    

    return data_for_plotly, None

def transform_data_for_visualization(result: Any, chart_type_str: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Transforms query result data (DataFrame, Series, etc.) into a format suitable for Plotly Express.
    This is used by the LLM pipeline to infer structure.
    Returns a dictionary of data (e.g., lists for x, y, names, values, or a dict for df)
    and a potential error message.
    """
    extracted_chart_type = extract_chart_type(chart_type_str)
    if extracted_chart_type == "none" or extracted_chart_type is None:
        return None, "No visualization recommended or chart type not extractable."

    # Ensure result is a DataFrame or Series for consistent handling
    if isinstance(result, np.ndarray):
         # Convert raw numpy array to Series for consistent handling
         result = pd.Series(result)
         logger.debug("Converted NumPy array result to Series for transformation.")
    elif isinstance(result, list) or isinstance(result, dict):
         # Attempt to convert raw list/dict to DataFrame/Series
         try:
             temp_df = pd.DataFrame(result)
             if not temp_df.empty:
                 if len(temp_df.columns) == 1:
                     result = temp_df.iloc[:, 0] # Treat as Series
                     logger.debug("Converted raw list/dict result to Series for transformation.")
                 else:
                     result = temp_df # Treat as DataFrame
                     logger.debug("Converted raw list/dict result to DataFrame for transformation.")
             else:
                  return None, "Could not transform raw list/dict into a suitable DataFrame/Series for visualization."
         except Exception as e:
              logger.warning(f"Could not convert raw list/dict to DataFrame/Series for viz transformation: {e}")
              return None, f"Unsupported data type ({type(result)}) for visualization transformation."
    elif not isinstance(result, (pd.DataFrame, pd.Series, type(None))):
         # If it's still not a DF, Series, or None, it's likely scalar or unsupported
         return None, f"Unsupported data type ({type(result)}) for visualization transformation."


    if result is None or (isinstance(result, (pd.DataFrame, pd.Series)) and result.empty):
        return None, "Input data for visualization is empty."


    try:
        # Apply initial type conversion and null handling if it's a DataFrame or Series
        if isinstance(result, pd.DataFrame):
             # Convert specific non-serializable types like Timestamps to strings
             result = result.applymap(lambda x: str(x) if isinstance(x, pd.Timestamp) else x, na_action='ignore')
             # Convert pandas nulls (NaN, NaT, None) to Python None
             result = result.where(pd.notnull(result), None)
        elif isinstance(result, pd.Series):
             # Convert specific non-serializable types like Timestamps to strings
             result = result.applymap(lambda x: str(x) if isinstance(x, pd.Timestamp) else x, na_action='ignore')
             # Convert pandas nulls (NaN, NaT, None) to Python None
             result = result.where(pd.notnull(result), None)


        data_for_plotly: Dict[str, Any] = {}

        if isinstance(result, pd.DataFrame):
            # Logic similar to prepare_data_for_plotly, but inferring columns
            if extracted_chart_type == "grouped_bar":
                # Assumption for LLM pipeline: The query result for grouped bar is already "tall"
                # with at least 3 columns: category, series, value.
                # We'll map the first column to X, second to Color, third to Y (value).
                if len(result.columns) >= 3:
                    x_col = result.columns[0]
                    color_col = result.columns[1]
                    y_col = result.columns[2] # This is the VALUE column

                    # Ensure Y column is numeric for grouped bar, coerce errors to None and drop
                    try:
                        # FIX: Explicitly create Series before pd.to_numeric
                        numeric_y_series = pd.to_numeric(pd.Series(result[y_col]), errors='coerce')
                        # Drop rows where value became NaN, or where X/Color is also None
                        valid_indices = numeric_y_series.dropna().index
                        if x_col: valid_indices = valid_indices.intersection(result[x_col].dropna().index)
                        if color_col: valid_indices = valid_indices.intersection(result[color_col].dropna().index)

                        result_filtered = result.loc[valid_indices].copy()
                        result_filtered[y_col] = numeric_y_series.loc[valid_indices] # Update Y column with numeric values

                        if result_filtered.empty:
                             return None, "Grouped bar chart data is empty after filtering for non-null numeric values."

                        data_for_plotly = {"data_frame": result_filtered.to_dict(orient='list'), "x": x_col, "y": y_col, "color": color_col}
                        data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])

                    except Exception as num_e:
                         return None, f"Error ensuring numeric data for grouped bar Y-axis (inferred column '{y_col}'): {num_e}"

                else: return None, "DataFrame for grouped bar needs at least 3 columns (Category, Series, Value)."

            elif extracted_chart_type in ["bar", "line", "scatter", "pie"]:
                if len(result.columns) >= 2:
                     col0_data = result.iloc[:, 0]
                     col1_data = result.iloc[:, 1]
                     x_col_name = result.columns[0]
                     y_col_name = result.columns[1]

                     # For scatter and pie, ensure Y/values are primarily numeric, coerce errors to None and drop
                     if extracted_chart_type in ["scatter", "pie"]:
                         try:
                             # FIX: Explicitly create Series before pd.to_numeric
                             numeric_vals_series = pd.to_numeric(pd.Series(col1_data), errors='coerce')
                             # Drop entries where the numeric value is None after coercion, or where x/name is also None
                             valid_indices = numeric_vals_series.dropna().index.intersection(col0_data.dropna().index)

                             df_filtered = pd.DataFrame({x_col_name: col0_data.loc[valid_indices],
                                                         y_col_name: numeric_vals_series.loc[valid_indices]})

                             if df_filtered.empty:
                                  return None, f"{extracted_chart_type} chart requires two columns with corresponding non-null numeric values (Y/Values)."

                             if extracted_chart_type == "pie":
                                 # For pie, map the first column to names and second to values explicitly
                                 data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "names": x_col_name, "values": y_col_name}
                                 data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])
                             else: # Scatter
                                 data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                                 data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])


                         except Exception as num_e:
                              return None, f"Error ensuring numeric data for {extracted_chart_type} chart: {num_e}"

                     # For bar/line, Y should ideally be numeric, coerce errors to None but don't necessarily drop rows unless X is also None
                     if extracted_chart_type in ["bar", "line"]:
                          try:
                             # FIX: Explicitly create Series before pd.to_numeric
                             numeric_y_series = pd.to_numeric(pd.Series(col1_data), errors='coerce')
                             # Keep rows where X is not null OR Y is not null after coercion
                             valid_indices = col0_data.dropna().index.union(numeric_y_series.dropna().index)

                             df_processed = pd.DataFrame({x_col_name: col0_data.loc[valid_indices],
                                                          y_col_name: numeric_y_series.loc[valid_indices].where(pd.notnull(numeric_y_series.loc[valid_indices]), None)
                                                          })

                             if df_processed[y_col_name].isnull().sum() > len(df_processed) * 0.5:
                                  logger.warning(f"Y-axis for {extracted_chart_type} ('{y_col_name}') seems mostly non-numeric or null after coercion. Proceeding anyway.")

                             data_for_plotly = {"data_frame": df_processed.to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                             data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])


                          except Exception as num_e:
                             logger.warning(f"Could not fully numeric-coerce Y-axis for {extracted_chart_type} ('{y_col_name}'): {num_e}. Proceeding with original data.")
                             # Fallback: use the original data (with Timestamp/NaN->str/None conversion applied earlier)
                             data_for_plotly = {"data_frame": result[[x_col_name, y_col_name]].to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                             data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])


                # Added handling for 1 column DataFrame result (e.g., value_counts().to_frame())
                elif len(result.columns) == 1 and extracted_chart_type in ["bar", "pie"]:
                     # Assume the index is the category (names) and the single column is the value
                     col0_data = result.iloc[:, 0] # This is the value column
                     index_data = result.index # This is the category/name index

                     # Convert index and values to lists, handling Timestamp/NaN to str/None
                     names_list = [str(x) if isinstance(x, (pd.Timestamp, pd.NaT)) else (None if pd.isna(x) else x) for x in index_data.tolist()]
                     values_list = [str(x) if isinstance(x, (pd.Timestamp, pd.NaT)) else (None if pd.isna(x) else x) for x in col0_data.tolist()]

                     # For bar and pie, values must be numeric. Coerce errors to None and drop
                     try:
                          # FIX: Explicitly create Series before pd.to_numeric
                          numeric_values_series = pd.to_numeric(pd.Series(values_list), errors='coerce')
                          # Drop entries where value became None after coercion OR where name is None
                          valid_indices = numeric_values_series.dropna().index.intersection(pd.Series(names_list).dropna().index)

                          names_list_filtered = [names_list[i] for i in valid_indices]
                          values_list_filtered = numeric_values_series.loc[valid_indices].tolist() # numeric list


                          if not names_list_filtered or not values_list_filtered:
                               return None, f"{extracted_chart_type} chart from 1-column result requires non-null names (index) and numeric values."

                          df_filtered = pd.DataFrame({"names": names_list_filtered, "values": values_list_filtered})

                          if extracted_chart_type == "pie":
                              data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "names": "names", "values": "values"}
                              data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])
                          else: # bar
                              data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "x": "names", "y": "values"}
                              data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])# Use "names" for X, "values" for Y

                     except Exception as num_e:
                          return None, f"Error ensuring numeric data for {extracted_chart_type} from 1-column result: {num_e}"

                else: return None, f"Need at least 2 columns for {extracted_chart_type} chart (or 1 column Series result for bar/pie)."

        elif isinstance(result, pd.Series):
            if not result.empty:
                # Convert Series to DataFrame for easier handling, preserving index as a column
                result_df = result.reset_index()
                # Ensure data types are compatible after reset_index and before applying map/where
                result_df = result_df.applymap(lambda x: str(x) if isinstance(x, pd.Timestamp) else x, na_action='ignore')
                result_df = result_df.where(pd.notnull(result_df), None)

                # After reset_index, we expect columns 'index' (or similar) and '0' (the value)
                if len(result_df.columns) >= 2:
                    x_col_name = result_df.columns[0] # Index column
                    y_col_name = result_df.columns[1] # Value column


                    # For scatter and pie from Series, ensure Y (value) is numeric, coerce errors to None and drop
                    if extracted_chart_type in ["scatter", "pie"]:
                        try:
                             # FIX: Explicitly create Series before pd.to_numeric
                             numeric_vals_series = pd.to_numeric(pd.Series(result_df[y_col_name]), errors='coerce')
                             # Drop entries where value became None after coercion OR where x/name is None
                             valid_indices = numeric_vals_series.dropna().index.intersection(result_df[x_col_name].dropna().index)

                             df_filtered = result_df.loc[valid_indices].copy()
                             df_filtered[y_col_name] = numeric_vals_series.loc[valid_indices] # Update Y column


                             if df_filtered.empty:
                                  return None, f"{extracted_chart_type} from Series requires index (names) and numeric values with corresponding non-null entries."

                             if extracted_chart_type == "pie":
                                 data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "names": x_col_name, "values": y_col_name}
                                 data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])
                             else: # Scatter
                                 data_for_plotly = {"data_frame": df_filtered.to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                                 data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])

                        except Exception as num_e:
                             return None, f"Error ensuring numeric data for {extracted_chart_type} from Series: {num_e}"

                    elif extracted_chart_type in ["bar", "line"]:
                         try:
                             # FIX: Explicitly create Series before pd.to_numeric
                             numeric_y_series = pd.to_numeric(pd.Series(result_df[y_col_name]), errors='coerce')
                             # Keep rows where X is not null OR Y is not null after coercion
                             valid_indices = result_df[x_col_name].dropna().index.union(numeric_y_series.dropna().index)

                             df_processed = result_df.loc[valid_indices].copy()
                             df_processed[y_col_name] = numeric_y_series.loc[valid_indices].where(pd.notnull(numeric_y_series.loc[valid_indices]), None)


                             if df_processed[y_col_name].isnull().sum() > len(df_processed) * 0.5:
                                  logger.warning(f"Y-axis for {extracted_chart_type} from Series ('{y_col_name}') seems non-numeric. Proceeding anyway.")

                             data_for_plotly = {"data_frame": df_processed.to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                             data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])
                             

                         except Exception as num_e:
                             logger.warning(f"Could not fully numeric-coerce Y-axis for {extracted_chart_type} from Series ('{y_col_name}'): {num_e}. Proceeding with original Series data.")
                             # Fallback: use the original DataFrame converted from Series (with potential NaNs to None)
                             data_for_plotly = {"data_frame": result_df[[x_col_name, y_col_name]].to_dict(orient='list'), "x": x_col_name, "y": y_col_name}
                             data_for_plotly["data_frame"] = deep_convert_numpy_to_python(data_for_plotly["data_frame"])

                else:
                     return None, "Series needs both index and values to transform for chart."

            else: return None, "Input Series for visualization is empty."

        if not data_for_plotly: # Check if data_for_plotly dict was populated
             return None, f"Transformation for chart type '{extracted_chart_type}' did not produce valid data for Plotly."

        # Final check on data_frame content
        if 'data_frame' not in data_for_plotly or pd.DataFrame(data_for_plotly['data_frame']).empty:
             return None, "Transformed data structure is missing 'data_frame' or it is empty."


        return data_for_plotly, None
    except Exception as e:
        logger.error(f"Error transforming data for visualization: {e}")
        return None, f"Error transforming data for visualization: {e}"


def generate_plotly_json(data: Union[Dict, pd.DataFrame, pd.Series], chart_type_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generates a Plotly figure JSON string from transformed data or a DataFrame.
    Can accept the dictionary format from transform_data_for_visualization or prepare_data_for_plotly.
    Returns the JSON string and a potential error message.
    """
    # Allow passing data dictionary OR a DataFrame directly now?
    # No, let's keep it consistent and expect the dictionary format from the transformation steps.
    # The transform functions handle converting DataFrame/Series to the dict format needed by generate_plotly_json.

    # Extract chart type from the recommendation string first (this function is still used by LLM pipeline)
    extracted_chart_type = extract_chart_type(chart_type_str)

    # If called manually, chart_type_str might just be the chart type name.
    # Let's allow passing just the chart type string if needed,
    # but prioritize the extracted type if present.
    if extracted_chart_type is None:
        extracted_chart_type = chart_type_str.strip().lower()
        if 'grouped bar' in extracted_chart_type: extracted_chart_type = 'grouped_bar'
        elif 'horizontal bar' in extracted_chart_type: extracted_chart_type = 'bar'
        elif 'bar' in extracted_chart_type: extracted_chart_type = 'bar'
        elif 'line' in extracted_chart_type: extracted_chart_type = 'line'
        elif 'pie' in extracted_chart_type: extracted_chart_type = 'pie'
        elif 'scatter' in extracted_chart_type: extracted_chart_type = 'scatter'
        else: extracted_chart_type = 'none' # Treat as none if not recognized simple type

    if not data or extracted_chart_type == "none" or extracted_chart_type is None:
        return None, "No data or no viz recommended/specified."

    fig = None
    try:
        # Use the data_frame key from the dictionary for Plotly Express
        df_for_px = pd.DataFrame(data.get("data_frame", {}))
        if df_for_px.empty:
             # This check might be redundant if prepare_data_for_plotly/transform_data already checked
             # but good safeguard.
             return None, "Transformed data for visualization is empty."

        # Basic default labels - can be made smarter later
        x_label = data.get("x", "X-axis")
        y_label = data.get("y", "Y-axis")
        color_label = data.get("color", "Group")
        size_label = data.get("size", "Size")


        # Ensure specified columns exist in the dataframe passed to px
        px_cols_to_check = [data.get('x'), data.get('y'), data.get('color'), data.get('size'), data.get('names'), data.get('values')]
        for col_name in px_cols_to_check:
            if col_name is not None and col_name not in df_for_px.columns:
                return None, f"Column '{col_name}' specified for Plotly Express not found in the prepared data."


        if extracted_chart_type == "bar":
            # Use column names from the data dict
            fig = px.bar(df_for_px, x=data.get("x"), y=data.get("y"), color=data.get("color"),
                         labels={data.get("x"): x_label, data.get("y"): y_label, data.get("color"): color_label} if data.get("color") else {data.get("x"): x_label, data.get("y"): y_label})

        elif extracted_chart_type == "grouped_bar":
            # Expects data_frame with x, y, color columns
            fig = px.bar(df_for_px, x=data.get("x"), y=data.get("y"), color=data.get("color"), barmode="group",
                         labels={data.get("x"): x_label, data.get("y"): y_label, data.get("color"): color_label}) # Color label likely represents series name

        elif extracted_chart_type == "line":
             fig = px.line(df_for_px, x=data.get("x"), y=data.get("y"), color=data.get("color"),
                           labels={data.get("x"): x_label, data.get("y"): y_label, data.get("color"): color_label} if data.get("color") else {data.get("x"): x_label, data.get("y"): y_label})

        elif extracted_chart_type == "pie":
             # Use column names for 'names' and 'values'
             fig = px.pie(df_for_px, names=data.get("names"), values=data.get("values"), title="Distribution")

        elif extracted_chart_type == "scatter":
             fig = px.scatter(df_for_px, x=data.get("x"), y=data.get("y"), color=data.get("color"), size=data.get("size"),
                              labels={data.get("x"): x_label, data.get("y"): y_label, data.get("color"): color_label, data.get("size"): size_label},
                              size_max=60) # Cap max size for aesthetics

        else:
            return None, f"Unsupported chart type '{extracted_chart_type}' for rendering."

        if fig:
            # Convert Plotly figure to JSON string
            return fig.to_dict(), None
        return None, "Figure could not be generated by Plotly Express."
    except Exception as e:
        logger.error(f"Error rendering {extracted_chart_type} visualization: {e}")
        return None, f"Error rendering visualization: {e}"
    
    
    
def generate_plotly_json_from_specs(
    chart_type: str,
    x_col_name: Optional[str] = None,
    y_col_name: Optional[str] = None,
    color_col_name: Optional[str] = None,
    size_col_name: Optional[str] = None
) -> schemas.VisualizationData:
    """
    Generates Plotly JSON for a manual visualization request using specified columns
     from the currently loaded DataFrame.
    """
    df = data_service.get_current_df()
    viz_data = schemas.VisualizationData()

    if df is None:
        viz_data.error = "No data file has been uploaded and processed yet."
        return viz_data

    # 1. Prepare data using the specified column names
    prepared_data, prepare_error = prepare_data_for_plotly(
        df=df,
        chart_type=chart_type,
        x_col_name=x_col_name,
        y_col_name=y_col_name,
        color_col_name=color_col_name,
        size_col_name=size_col_name
    )

    if prepare_error:
        viz_data.error = prepare_error
        return viz_data

    if not prepared_data:
        viz_data.error = "Data preparation failed or resulted in no data."
        return viz_data

    # 2. Generate Plotly JSON using the prepared data
    # Pass the *requested* chart_type directly, not from LLM recommendation string
    plotly_json_str, render_error = generate_plotly_json(prepared_data, chart_type)
    plotly_json_str = deep_convert_numpy_to_python(plotly_json_str)

    if render_error:
        viz_data.error = render_error
        return viz_data

    if not plotly_json_str:
        viz_data.error = "Plotly JSON generation failed."
        return viz_data

    viz_data.plotly_json = plotly_json_str
    return viz_data