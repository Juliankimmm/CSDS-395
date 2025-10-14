# Contest API

A FastAPI-based REST API for managing contests, submissions, and voting.

## Setup

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

4. Run the application:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation: `http://localhost:8000/docs`

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
