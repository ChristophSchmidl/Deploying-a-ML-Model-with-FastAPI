# Deploying a ML model with FastAPI

## Installation

This application uses Python version 3.8. You can create a virtual environment to run this application like this:

```
python -m venv venv
pip install --upgrade pip
pip install -r src/requirements.txt
```

## Usage

In order to start the server, you can execute the following command that then launches the FastAPI app:

```
uvicorn --host 0.0.0.0 --port 5000 --workers 4 src.main:app
```