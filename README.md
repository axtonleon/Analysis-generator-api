# Data Analysis API

A powerful FastAPI-based application that enables users to upload Excel/CSV files and perform natural language analysis on their data. The application provides automated insights, visualizations, and explanations through an intuitive API interface.

## Features

- **File Upload**: Support for CSV and Excel files
- **Natural Language Analysis**: Ask questions about your data in plain English
- **Automated Insights**: Get initial analyses and insights automatically
- **Interactive Visualizations**: Generate Plotly-based visualizations
- **Raw Query Support**: Execute custom Pandas queries
- **Column Analysis**: Detailed information about each column in your dataset

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the root directory:

```bash
touch .env
```

2. Add your environment variables (if needed):

```env
# Add your environment variables here
```

## Running the Application

1. Start the development server:

```bash
uvicorn main:app --reload
```

2. Access the API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### 1. Upload File

- **Endpoint**: `/api/uploadfile/`
- **Method**: POST
- **Description**: Upload a CSV or Excel file for analysis
- **Response**: File details, columns, preview, and initial analyses

### 2. Ask Question

- **Endpoint**: `/api/ask/`
- **Method**: POST
- **Description**: Ask natural language questions about your data
- **Response**: Analysis results, visualizations, and explanations

### 3. Execute Raw Query

- **Endpoint**: `/api/execute-raw-query/`
- **Method**: POST
- **Description**: Execute custom Pandas queries
- **Response**: Query results

### 4. Get Column Details

- **Endpoint**: `/api/columns/details/`
- **Method**: GET
- **Description**: Get detailed information about each column
- **Response**: Column statistics and metadata

### 5. Generate Visualization

- **Endpoint**: `/api/generate-viz-from-specs/`
- **Method**: POST
- **Description**: Generate visualizations based on specifications
- **Response**: Plotly JSON visualization data

## Development

### Code Style

This project follows PEP 8 guidelines and uses several tools to maintain code quality:

- **Black**: For code formatting
- **Flake8**: For linting
- **isort**: For import sorting

To run the code quality tools:

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8
```

### Testing

Run tests using pytest:

```bash
pytest
```

## Project Structure

```
├── main.py              # FastAPI application entry point
├── routes.py            # API route definitions
├── schemas.py           # Pydantic models and schemas
├── helpers.py           # Helper functions
├── services/           # Service layer
│   ├── data_service.py
│   ├── analysis_service.py
│   └── visualization_service.py
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

