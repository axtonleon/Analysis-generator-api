# schemas.py
from pydantic import BaseModel, Field, model_validator # Import model_validator for V2
from typing import Optional, Any, Dict, List, Union, Literal # Import Literal

class QueryResult(BaseModel):
    data: Optional[Any] = None # This will hold the JSON-serializable processed data
    is_dataframe: bool = False
    is_series: bool = False
    raw_result_str: Optional[str] = None # Store a string preview/summary of the original result
    error: Optional[str] = None

class VisualizationRecommendation(BaseModel):
    chart_type_description: Optional[str] = None
    chart_type_extracted: Optional[str] = None
    error: Optional[str] = None

class NaturalLanguageExplanation(BaseModel):
    explanation: Optional[str] = None
    error: Optional[str] = None

class VisualizationData(BaseModel):
    # CORRECTED TYPE: Plotly JSON is returned as a STRING by fig.to_json()
    plotly_json: Optional[dict] = None
    error: Optional[str] = None

class InitialAnalysisItem(BaseModel):
    analysis_title: str = "Untitled Analysis"
    suggested_question: str
    interpreted_query: Optional[str] = None
    query_result: QueryResult
    visualization_recommendation: VisualizationRecommendation
    visualization_data: VisualizationData
    natural_language_explanation: NaturalLanguageExplanation
    errors: List[str] = Field(default_factory=list)

class FileUploadResponse(BaseModel):
    filename: str
    message: str
    columns: Optional[List[str]] = None
    preview: Optional[List[Dict[str, Any]]] = None
    initial_analyses: List[InitialAnalysisItem] = Field(default_factory=list)

class QuestionRequest(BaseModel):
    question: str

class ProcessedQuestionResponse(BaseModel):
    original_question: str
    interpreted_query: Optional[str] = None
    query_result: QueryResult
    visualization_recommendation: VisualizationRecommendation
    visualization_data: VisualizationData
    natural_language_explanation: NaturalLanguageExplanation
    errors: List[str] = Field(default_factory=list)

# Schema for raw query execution
class RawQueryRequest(BaseModel):
    query: str = Field(..., description="Raw Pandas query string to execute against the loaded DataFrame (variable name 'df').")

class RawQueryResponse(BaseModel):
    executed_query: str
    result: QueryResult


# SCHEMAS FOR COLUMN DETAILS
class ColumnDetails(BaseModel):
    name: str
    dtype: str # Pandas dtype string
    non_null_count: int
    unique_count: Optional[int] = None
    basic_stats: Optional[Dict[str, Any]] = None # Results from .describe(), values converted to JSON-serializable types
    error: Optional[str] = None # Added error field

class ColumnDetailsResponse(BaseModel):
    filename: Optional[str] = None # Add filename for context
    total_rows: Optional[int] = None # Add total rows
    columns: List[ColumnDetails] = Field(default_factory=list)
    error: Optional[str] = None


# SCHEMAS FOR MANUAL VISUALIZATION
class VizSpecsRequest(BaseModel):
    chart_type: str = Field(..., description="Desired chart type (e.g., 'bar', 'line', 'scatter', 'pie', 'grouped_bar').")
    x_column_name: Optional[str] = Field(None, description="Name of the column for the X-axis or categories.")
    y_column_name: Optional[str] = Field(None, description="Name of the column for the Y-axis or values.")
    color_column_name: Optional[str] = Field(None, description="Name of the column for color encoding (grouped bars, line colors, scatter colors).")
    size_column_name: Optional[str] = Field(None, description="Name of the column for size encoding (scatter plots).")


# NEW SCHEMAS: For Data Cleaning Operations
class DropNaOperation(BaseModel):
    type: Literal["drop_na"]
    subset: Optional[List[str]] = Field(None, description="Columns to consider for dropping rows/columns.")
    how: Optional[Literal["any", "all"]] = Field("any", description="Determine if row or column is removed when all or any NA is present.")
    axis: Optional[Literal["index", "columns"]] = Field("index", description="Determine if rows or columns are removed. 'index' (0) or 'columns' (1).")

class ImputeNaOperation(BaseModel):
    type: Literal["impute_na"]
    subset: Optional[List[str]] = Field(None, description="Columns to apply imputation to.")
    strategy: Literal["mean", "median", "mode", "constant", "ffill", "bfill"] = Field(..., description="Strategy for imputation.")
    value: Optional[Any] = Field(None, description="Value to use for imputation if strategy is 'constant'. Required if strategy is 'constant'.")

    # Pydantic V2 model validator
    @model_validator(mode='after')
    def check_value_if_constant(self):
        if self.strategy == 'constant' and self.value is None:
            # Note: Pydantic v2 validation doesn't allow raising ValueError directly in model_validator.
            # Instead, we can add a field error or return a different value.
            # For this case, returning self and letting the service layer handle the None value check might be simpler,
            # or adding a custom error field. Let's rely on the service layer check for now,
            # but the validator *can* be used to ensure the 'value' field is present if strategy is 'constant'.
             if self.value is None: # Re-check just in case validator doesn't fully enforce
                  # This specific check might be better in the service function
                  pass # Allow validation to pass, but service function will error
        return self


class ConvertTypeOperation(BaseModel):
    type: Literal["convert_type"]
    columns: List[str] = Field(..., description="List of columns to convert.")
    dtype: str = Field(..., description="Target data type string (e.g., 'int', 'float', 'str', 'datetime64', 'category', 'bool', 'Int64' for nullable int).")
    errors: Optional[Literal["ignore", "raise", "coerce"]] = Field("coerce", description="How to handle errors during conversion (for numeric/datetime conversions).")

class DropDuplicatesOperation(BaseModel):
    type: Literal["drop_duplicates"]
    subset: Optional[List[str]] = Field(None, description="Columns to consider for identifying duplicate rows. Defaults to all columns if None or empty.")
    keep: Optional[Literal["first", "last", False]] = Field("first", description="Which duplicates to keep.")

# Discriminated Union for Cleaning Operations
# Pydantic uses the 'type' field to discriminate between the different operation models
CleaningOperation = Union[DropNaOperation, ImputeNaOperation, ConvertTypeOperation, DropDuplicatesOperation]

# NEW SCHEMA: Request body for cleaning endpoint
class CleaningRequest(BaseModel):
    operations: List[CleaningOperation] = Field(..., description="List of cleaning operations to apply sequentially.")

# NEW SCHEMA: Response body for cleaning endpoint
class CleaningResponse(BaseModel):
    filename: Optional[str] = None
    message: str
    total_rows_before: Optional[int] = None
    total_rows_after: Optional[int] = None
    columns_before: Optional[List[str]] = None
    columns_after: Optional[List[str]] = None
    preview_after: Optional[List[Dict[str, Any]]] = None
    changes_summary: List[str] = Field(default_factory=list, description="Summary of changes made by each operation.")
    errors: List[str] = Field(default_factory=list, description="List of errors or warnings encountered.")