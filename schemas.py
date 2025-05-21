# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List

class QueryResult(BaseModel):
    data: Optional[Any] = None
    is_dataframe: bool = False
    is_series: bool = False
    raw_result_str: Optional[str] = None
    error: Optional[str] = None

class VisualizationRecommendation(BaseModel):
    chart_type_description: Optional[str] = None
    chart_type_extracted: Optional[str] = None
    error: Optional[str] = None

class NaturalLanguageExplanation(BaseModel):
    explanation: Optional[str] = None
    error: Optional[str] = None

class VisualizationData(BaseModel):
    plotly_json: Optional[dict] = None
    # image_path: Optional[str] = None # REMOVED as per requirement
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
    
    
    
class RawQueryRequest(BaseModel):
    query: str = Field(..., description="Raw Pandas query string to execute against the loaded DataFrame (variable name 'df').")

class RawQueryResponse(BaseModel):
    executed_query: str
    result: QueryResult
    
    
# NEW SCHEMAS: For Column Details
class ColumnDetails(BaseModel):
    name: str
    dtype: str # Pandas dtype string
    non_null_count: int
    unique_count: Optional[int] = None
    basic_stats: Optional[Dict[str, Any]] = None # Results from .describe()

class ColumnDetailsResponse(BaseModel):
    filename: Optional[str] = None # Add filename for context
    total_rows: Optional[int] = None # Add total rows
    columns: List[ColumnDetails] = Field(default_factory=list)
    error: Optional[str] = None


# NEW SCHEMAS: For Manual Visualization
class VizSpecsRequest(BaseModel):
    chart_type: str = Field(..., description="Desired chart type (e.g., 'bar', 'line', 'scatter', 'pie', 'grouped_bar').")
    x_column_name: Optional[str] = Field(None, description="Name of the column for the X-axis or categories.")
    y_column_name: Optional[str] = Field(None, description="Name of the column for the Y-axis or values.")
    color_column_name: Optional[str] = Field(None, description="Name of the column for color encoding (grouped bars, line colors, scatter colors).")
    size_column_name: Optional[str] = Field(None, description="Name of the column for size encoding (scatter plots).")