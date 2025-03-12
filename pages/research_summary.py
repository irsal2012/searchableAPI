import streamlit as st
import pandas as pd
import sys
import os
import time
from datetime import datetime
import io
from openai import OpenAI

# Add parent directory to path to import vector_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store import initialize_pinecone, query_similar_chunks

# Set page config
st.set_page_config(page_title="Research Summary Generator", layout="wide")

# Initialize OpenAI client
client = OpenAI()

# Initialize Pinecone
try:
    pinecone_index = initialize_pinecone()
    pinecone_initialized = True
except Exception as e:
    pinecone_initialized = False
    pinecone_error = str(e)

def load_selected_articles():
    """Load articles that were selected in the main app."""
    try:
        # Try to load from session state first (if coming from main app)
        if 'selected_articles_for_summary' in st.session_state:
            return st.session_state.selected_articles_for_summary
        
        # Check if a search has been performed in the current session
        if 'search_complete' in st.session_state and st.session_state.search_complete:
            # Only try to load from saved files if a search has been completed
            try:
                df = pd.read_json('search_results.json')
                return df
            except:
                try:
                    df = pd.read_csv('search_results.csv')
                    return df
                except:
                    return None
        else:
            # No search has been performed in the current session
            return None
    except Exception as e:
        st.error(f"Error loading articles: {str(e)}")
        return None

def create_summary_prompt(article_data, focus_areas, summary_type):
    """Create prompt for summary generation based on focus areas and summary type."""
    # Base prompt
    prompt = "Please create a "
    
    # Add summary type
    if summary_type == "comprehensive":
        prompt += "comprehensive research summary"
    elif summary_type == "brief":
        prompt += "brief research overview"
    elif summary_type == "technical":
        prompt += "technical research analysis"
    elif summary_type == "clinical":
        prompt += "clinical implications summary"
    
    prompt += " based on the following research articles:\n\n"
    
    # Add article data
    for i, article in enumerate(article_data):
        prompt += f"ARTICLE {i+1}:\n"
        prompt += f"Title: {article['title']}\n"
        prompt += f"Authors: {article['authors']}\n"
        prompt += f"Publication Date: {article['publication_date']}\n"
        prompt += f"Source Type: {article['source_type']}\n"
        
        # Add abstract if available
        if article['abstract']:
            prompt += f"Abstract: {article['abstract']}\n"
            
        # Add summary
        prompt += f"Summary: {article['summary']}\n\n"
    
    # Add focus areas
    prompt += "Please focus on the following areas in your summary:\n"
    for area in focus_areas:
        prompt += f"- {area}\n"
        
    # Add structure requirements
    prompt += "\nPlease structure your summary with the following sections:\n"
    prompt += "1. Research Overview: A high-level summary of the research landscape\n"
    prompt += "2. Key Findings: The most important discoveries and results across all articles\n"
    prompt += "3. Methodologies: Comparison of research methods used\n"
    
    if "research gaps" in [area.lower() for area in focus_areas]:
        prompt += "4. Research Gaps: Identification of areas that need further investigation\n"
        
    if "future directions" in [area.lower() for area in focus_areas]:
        prompt += "5. Future Directions: Potential next steps for research in this area\n"
        
    if "clinical implications" in [area.lower() for area in focus_areas]:
        prompt += "6. Clinical Implications: How these findings might impact clinical practice\n"
    
    return prompt

def generate_research_summary(articles, focus_areas, summary_type="comprehensive"):
    """
    Generate comprehensive research summary using OpenAI.
    
    Args:
        articles (pd.DataFrame): DataFrame containing article information
        focus_areas (list): List of areas to focus on in the summary
        summary_type (str): Type of summary to generate
        
    Returns:
        str: Generated summary
    """
    try:
        # Prepare article data for the prompt
        article_data = []
        for _, article in articles.iterrows():
            article_info = {
                "title": article.get('Title', 'Unknown'),
                "authors": article.get('Authors', 'Unknown'),
                "publication_date": article.get('Publication_Date', 'Unknown'),
                "source_type": article.get('Source_Type', 'Unknown'),
                "summary": article.get('Summary', ''),
                "abstract": article.get('Abstract', '')
            }
            article_data.append(article_info)
            
        # Create prompt based on focus areas and summary type
        prompt = create_summary_prompt(article_data, focus_areas, summary_type)
        
        # Generate summary using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical research assistant specializing in creating comprehensive research summaries. Your summaries are well-structured, insightful, and highlight key findings, methodologies, and gaps in the research."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    # Header with navigation link back to main app
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Research Summary Generator")
    with col2:
        if st.button("← Back to Explorer", use_container_width=True):
            st.switch_page("app.py")
    
    # Divider
    st.markdown("---")
    
    # Brief description
    st.markdown("""
    Generate comprehensive summaries of multiple research articles to get a consolidated view of the research landscape.
    This tool analyzes the selected articles and creates a structured summary highlighting key findings, methodologies, and research gaps.
    """)
    
    # Display vector database status
    if pinecone_initialized:
        st.success("✅ Connected to knowledge base")
    else:
        st.error(f"❌ Not connected to knowledge base: {pinecone_error}")
        st.info("Please add PINECONE_API_KEY to your .env file to enable knowledge base search.")
    
    # Load articles
    articles = load_selected_articles()
    
    if articles is None or articles.empty:
        st.warning("No articles found. Please search and select articles in the main Explorer first.")
        return
    
    # Display article count
    st.info(f"Found {len(articles)} articles for summary generation.")
    
    # Summary options
    st.subheader("Summary Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        summary_type = st.selectbox(
            "Summary Type",
            options=["Comprehensive", "Brief", "Technical", "Clinical"],
            index=0,
            help="Select the type of summary you want to generate"
        )
    
    with col2:
        focus_areas = st.multiselect(
            "Focus Areas",
            options=["Key findings", "Methodologies", "Research gaps", "Future directions", "Clinical implications", "Regulatory aspects"],
            default=["Key findings", "Methodologies", "Research gaps"],
            help="Select areas to focus on in the summary"
        )
    
    # Generate summary button
    if st.button("Generate Research Summary", type="primary"):
        if not focus_areas:
            st.error("Please select at least one focus area.")
            return
            
        with st.spinner("Generating research summary..."):
            # Generate summary
            summary = generate_research_summary(
                articles, 
                focus_areas, 
                summary_type.lower()
            )
            
            # Store in session state
            st.session_state.generated_summary = summary
            
            # Display summary
            st.subheader("Generated Research Summary")
            st.markdown(summary)
            
            # Download options
            st.subheader("Download Options")
            
            # Create download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Text download
                st.download_button(
                    label="Download as Text",
                    data=summary,
                    file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Markdown download
                st.download_button(
                    label="Download as Markdown",
                    data=summary,
                    file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    # Display previously generated summary if available
    elif 'generated_summary' in st.session_state:
        st.subheader("Generated Research Summary")
        st.markdown(st.session_state.generated_summary)
        
        # Download options
        st.subheader("Download Options")
        
        # Create download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Text download
            st.download_button(
                label="Download as Text",
                data=st.session_state.generated_summary,
                file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Markdown download
            st.download_button(
                label="Download as Markdown",
                data=st.session_state.generated_summary,
                file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
