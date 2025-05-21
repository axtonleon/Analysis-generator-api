# routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse # Removed image serving
# import os # No longer needed for image directory
# import time # No longer needed for generating image names

# Import functions from the new service files
from services import data_service, analysis_service, visualization_service

import schemas # Import schemas

router = APIRouter()

# REMOVED: Image saving directory logic
# IMAGE_SAVE_DIRECTORY = "generated_images"
# if not os.path.exists(IMAGE_SAVE_DIRECTORY):
#     os.makedirs(IMAGE_SAVE_DIRECTORY, exist_ok=True)


@router.post("/uploadfile/", response_model=schemas.FileUploadResponse)
async def create_upload_file_and_analyze(file: UploadFile = File(...)):
    """
    Uploads a CSV or Excel file, performs initial automated analyses (up to 5),
    generates insights and Plotly JSON.
    """
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel.")

    # Call the data_service function to load and analyze
    df, error_message, initial_analyses = await data_service.load_and_initially_analyze_data(file)

    if error_message and df is None: # Critical load error, df couldn't be loaded
        raise HTTPException(status_code=500, detail=error_message)

    # Even if df is loaded but there's an error_message (e.g. during analysis), we might proceed
    # or handle it differently. If df is None and no initial_analyses, it's a complete failure.
    if df is None and not initial_analyses:
         raise HTTPException(status_code=500, detail="Failed to load data or perform any initial analysis.")

    response_message = "File processed."
    if initial_analyses:
        response_message += " Initial analyses (if any) are below."
    elif error_message: # df might be loaded, but initial analysis failed
        response_message += f" Could not perform all initial analyses: {error_message}"
    else: # df loaded, no initial analysis for some reason (e.g., no ideas from LLM)
        response_message += " No initial analyses were generated."


    return schemas.FileUploadResponse(
        filename=file.filename,
        message=response_message,
        columns=data_service.get_current_df_columns(), # Get columns from data_service
        preview=data_service.get_current_df_preview(), # Get preview from data_service
        initial_analyses=initial_analyses
    )

@router.post("/ask/", response_model=schemas.ProcessedQuestionResponse)
async def ask_question_route(request: schemas.QuestionRequest):
    """
    Takes a user's natural language question about the previously uploaded data,
    generates a Pandas query, executes it, recommends/generates a visualization
    (Plotly JSON only), and provides a natural language explanation.
    """
    df = data_service.get_current_df() # Get the current DataFrame from data_service
    if df is None:
        raise HTTPException(status_code=400, detail="No data file has been uploaded and processed yet. Please upload a file first via /uploadfile/ endpoint.")

    # Call the analysis_service function to process the question
    response_payload = await analysis_service.process_user_question(question=request.question)

    # The response_payload already contains the Plotly JSON if generated, no image path needed.

    return response_payload


@router.post("/execute-raw-query/", response_model=schemas.RawQueryResponse)
async def execute_raw_query_route(request: schemas.RawQueryRequest):
    """
    Executes a raw Pandas query string provided by the user against the currently
    loaded DataFrame. Returns the query result.
    """
    df = data_service.get_current_df()
    if df is None:
         # Return a RawQueryResponse with an error in the result object
        return schemas.RawQueryResponse(
            executed_query=request.query,
            result=schemas.QueryResult(error="No data file has been uploaded and processed yet.")
        )

    # Call the analysis_service function to execute the raw query
    query_result = analysis_service.execute_raw_pandas_query(query=request.query)

    # The service function already returns the QueryResult object directly
    return schemas.RawQueryResponse(
        executed_query=request.query,
        result=query_result
    )
    
@router.get("/columns/details/", response_model=schemas.ColumnDetailsResponse)
async def get_column_details_route():
    """
    Retrieves detailed information for each column in the currently loaded DataFrame,
    including dtype, non-null count, unique count, and basic statistics.
    """
    # The service function handles checking if df is loaded
    return data_service.get_detailed_column_info()


# NEW ROUTE: Generate Visualization from Specs
@router.post("/generate-viz-from-specs/", response_model=schemas.VisualizationData)
async def generate_viz_from_specs_route(request: schemas.VizSpecsRequest):
    """
    Generates a Plotly JSON visualization based on specified chart type and column names
    from the currently loaded DataFrame. Bypasses LLM visualization recommendation.
    """
    df = data_service.get_current_df()
    if df is None:
        # Return a VisualizationData with an error
        return schemas.VisualizationData(error="No data file has been uploaded and processed yet.")

    # Call the visualization_service function
    viz_data_result = visualization_service.generate_plotly_json_from_specs(
        chart_type=request.chart_type,
        x_col_name=request.x_column_name,
        y_col_name=request.y_column_name,
        color_col_name=request.color_column_name,
        size_col_name=request.size_column_name
    )

    # The service function returns a populated VisualizationData object directly
    return viz_data_result