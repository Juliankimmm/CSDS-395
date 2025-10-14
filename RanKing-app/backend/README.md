# Contest API

A FastAPI-based REST API for managing contests, submissions, and voting.

## Setup
0. Ensure you are in a python environment in your machine
```bash
python -m venv venv
source venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your database credentials
```

3. Create PostgreSQL database:

```bash
createdb contest_db
```
Test for database connection:
```bash
python3 test_db_connection.py
```

4. Run the application:

```bash
uvicorn app.main:app --reload --port 8585
```

The API will be available at `http://localhost:8585`

## API Documentation

Interactive API documentation: `http://localhost:8585/docs`

## Running Tests

```bash
python tests/test_api.py
```

## Project Structure

- `app/` - Main application code
- `app/routers/` - API endpoints
- `app/models/` - Database models
- `app/schemas/` - Pydantic schemas
- `app/utils/` - Utility functions
- `tests/` - Test scripts
- `uploads/` - Uploaded files directory
