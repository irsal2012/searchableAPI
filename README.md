# SearchableAPI

A Python application for searching, storing, and analyzing research articles using vector embeddings and AI.

## Features

- Search for research articles
- Store article data in vector databases (Pinecone)
- Generate research summaries
- Interactive Q&A chat with your research data

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/irsal2012/searchableAPI.git
   cd searchableAPI
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key and Pinecone credentials

## Usage

Run the main application:
```
python app.py
```

## Project Structure

- `app.py` - Main application entry point
- `search_articles.py` - Article search functionality
- `vector_store.py` - Vector database operations
- `pages/` - UI components
  - `research_summary.py` - Research summary generation
  - `qa_chat.py` - Q&A chat interface

## License

MIT
