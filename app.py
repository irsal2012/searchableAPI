import streamlit as st
import pandas as pd
import plotly.express as px
from search_articles import search_articles, get_search_domains
import json
import time
import os
import traceback
from datetime import datetime
from vector_store import initialize_pinecone, store_article_chunks

st.set_page_config(page_title="Medical Research Explorer", layout="wide")

# Initialize session state variables
if 'search_complete' not in st.session_state:
    st.session_state.search_complete = False
if 'selected_articles' not in st.session_state:
    st.session_state.selected_articles = {}
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'original_results' not in st.session_state:
    st.session_state.original_results = None
if 'total_chunks_stored' not in st.session_state:
    st.session_state.total_chunks_stored = 0

def load_results():
    # The app doesn't need to read search_results.csv
    return None

def display_summary_stats(df):
    if df is None or df.empty:
        st.warning("No data available to display statistics")
        return
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(df))
    
    with col2:
        source_counts = df['Source_Type'].value_counts()
        st.metric("Research Papers", source_counts.get('Research Paper', 0))
    
    with col3:
        st.metric("Clinical Trials", source_counts.get('Clinical Trial', 0))

def create_source_distribution(df):
    if df is None or df.empty:
        st.warning("No data available to create source distribution chart")
        return
        
    source_counts = df['Source_Type'].value_counts().reset_index()
    source_counts.columns = ['Source Type', 'Count']
    
    fig = px.pie(source_counts, values='Count', names='Source Type', 
                 title='Distribution of Source Types')
    st.plotly_chart(fig)

def create_phase_distribution(df):
    if df is None or df.empty:
        st.warning("No data available to create phase distribution chart")
        return
        
    # Filter out NaN values
    phase_df = df[df['Development_Phase'].notna()]
    
    if phase_df.empty:
        st.info("No development phase data available")
        return
        
    phase_counts = phase_df['Development_Phase'].value_counts().reset_index()
    phase_counts.columns = ['Development Phase', 'Count']
    
    fig = px.bar(phase_counts, x='Development Phase', y='Count',
                 title='Distribution of Development Phases')
    st.plotly_chart(fig)

# Callback for selection toggle
def toggle_selection(index):
    if index in st.session_state.selected_articles:
        st.session_state.selected_articles[index] = not st.session_state.selected_articles[index]
    else:
        st.session_state.selected_articles[index] = True

# Callback for select all
def select_all():
    if st.session_state.original_results is not None:
        for i in range(len(st.session_state.original_results)):
            st.session_state.selected_articles[i] = True

# Callback for deselect all
def deselect_all():
    if st.session_state.original_results is not None:
        for i in range(len(st.session_state.original_results)):
            st.session_state.selected_articles[i] = False

# Callback for execute
def execute_selection():
    if st.session_state.original_results is not None:
        df = st.session_state.original_results
        selected_indices = [i for i, selected in st.session_state.selected_articles.items() if selected]
        
        if selected_indices:
            selected_df = df.iloc[selected_indices].copy()
            
            # Save to files
            selected_df.to_csv('search_results.csv', index=False)
            selected_df.to_json('search_results.json', orient='records', indent=2)
            
            # Store in session state for display and for research summary
            st.session_state.processed_results = selected_df
            st.session_state.selected_articles_for_summary = selected_df
            
            # Create a progress container for vector storage
            progress_container = st.empty()
            with progress_container.container():
                st.subheader("Storing Articles in Vector Database")
                progress_bar = st.progress(0)
                status_text = st.empty()
                debug_info = st.expander("Debug Information")
                
                try:
                    # Initialize Pinecone
                    status_text.text("Initializing Pinecone...")
                    debug_info.write("Checking Pinecone API key...")
                    api_key = os.getenv("PINECONE_API_KEY")
                    if not api_key:
                        raise ValueError("PINECONE_API_KEY environment variable is not set")
                    debug_info.write("API key found, initializing Pinecone...")
                    
                    index = initialize_pinecone()
                    debug_info.write("Pinecone initialized successfully")
                    
                    # Store articles in Pinecone
                    total_articles = len(selected_df)
                    total_chunks = 0
                    
                    debug_info.write(f"Processing {total_articles} articles")
                    
                    for i, (_, article) in enumerate(selected_df.iterrows()):
                        # Convert article to dictionary
                        article_dict = article.to_dict()
                        
                        # Update progress
                        progress_value = (i / total_articles)
                        progress_bar.progress(progress_value)
                        status_text.text(f"Processing article {i+1}/{total_articles}: {article_dict.get('Title', 'Unknown')}")
                        
                        # Log article details
                        debug_info.write(f"Article {i+1}: {article_dict.get('Title', 'Unknown')}")
                        debug_info.write(f"URL: {article_dict.get('URL', 'No URL')}")
                        debug_info.write(f"Summary length: {len(article_dict.get('Summary', '')) if article_dict.get('Summary') else 0} characters")
                        
                        # Define status callback for article processing
                        def article_status_callback(message, value):
                            status_text.text(f"Article {i+1}/{total_articles}: {message}")
                            debug_info.write(f"Status: {message}")
                        
                        # Store article chunks
                        try:
                            chunks_stored = store_article_chunks(article_dict, index, article_status_callback)
                            debug_info.write(f"Stored {chunks_stored} chunks for this article")
                            total_chunks += chunks_stored
                        except Exception as article_error:
                            debug_info.write(f"Error processing article: {str(article_error)}")
                            # Continue with next article instead of failing completely
                    
                    # Update final progress
                    progress_bar.progress(1.0)
                    if total_chunks > 0:
                        status_text.text(f"✅ Successfully stored {total_chunks} chunks from {total_articles} articles in Pinecone")
                        # Store total chunks in session state for display
                        st.session_state.total_chunks_stored = total_chunks
                    else:
                        status_text.warning(f"⚠️ No chunks were stored. Check the debug information for details.")
                        st.session_state.total_chunks_stored = 0
                    time.sleep(2)  # Show success message for a moment
                    
                except Exception as e:
                    status_text.error(f"Error storing articles in Pinecone: {str(e)}")
                    debug_info.write(f"Error details: {str(e)}")
                    import traceback
                    debug_info.write(f"Traceback: {traceback.format_exc()}")
                    time.sleep(3)  # Show error message for a moment
            
            # Clear the progress container
            progress_container.empty()
            
            st.session_state.execute_success = True
        else:
            st.session_state.execute_warning = True

def main():
    # Create a layout with columns for the title and buttons
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col1:
        st.title("Medical Research Explorer")
    
    with col2:
        # Research Summary button
        if st.button("📊 Summary", use_container_width=True, help="Generate Research Summary"):
            st.switch_page("pages/research_summary.py")
    
    with col3:
        # Q&A button in the top right
        if st.button("💬 Q&A Chat", use_container_width=True, help="Open the Interactive Q&A Chat"):
            st.switch_page("pages/qa_chat.py")
    
    # Sidebar
    st.sidebar.title("Search Controls")
    
    # Search parameters
    query = st.sidebar.text_input("Search Query", placeholder="Enter your search query")
    years = st.sidebar.slider("Years to Search", min_value=1, max_value=10, value=5, help="Number of years to look back")
    num_results = st.sidebar.slider("Number of Results", min_value=5, max_value=50, value=20)
    
    # Get available source types from search_articles
    domains = get_search_domains()
    available_source_types = list(domains.keys())
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        source_types = st.multiselect(
            "Source Types",
            ["Research Paper", "Clinical Trial", "Regulatory Document", "Company Document", "News"],
            default=["Research Paper", "Clinical Trial"],
            help="Select which types of sources to search. PubMed and ClinicalTrials.gov are always included."
        )
        
        st.info("""
        Search Strategy:
        1. Direct API access to PubMed and ClinicalTrials.gov
        2. Limited searches on selected domains
        3. Automatic metadata extraction and summarization
        """)
    
    search_button = st.sidebar.button(
        "Run Search",
        #help="This will search PubMed and ClinicalTrials.gov APIs directly, plus selected domains. May take several minutes to complete."
        help="May take several minutes to complete."
    )
    
    # Initialize df variable
    df = None
    
    # Main content area for search status and results
    main_container = st.container()
    
    with main_container:
        if search_button:
            if query:
                # Create containers for progress tracking
                progress_container = st.container()
                with progress_container:
                    progress = st.progress(0)
                    status = st.empty()
                    results_count = st.empty()
                
                try:
                    def update_progress(message, value):
                        status.text(message)
                        progress.progress(value)
                    
                    # Start search
                    update_progress("Searching PubMed...", 20)
                    
                    try:
                        df = search_articles(
                            query=query,
                            num_results=num_results,
                            years_back=years,
                            source_types=source_types if source_types else None,
                            status_callback=update_progress
                        )
                        
                        # Store the original results in session state
                        if not df.empty:
                            st.session_state.original_results = df.copy()
                            
                            # Initialize selection state for each article if not already done
                            # Set all checkboxes to unchecked by default
                            st.session_state.selected_articles = {i: False for i in range(len(df))}
                            
                            # Save all results initially
                            df.to_csv('search_results.csv', index=False)
                            df.to_json('search_results.json', orient='records', indent=2)
                            
                            # Store in session state for display - make results always available
                            st.session_state.processed_results = df.copy()
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
                        df = pd.DataFrame()  # Empty dataframe
                    
                    # Update final status
                    progress.progress(100)
                    time.sleep(1)  # Brief pause to show completion
                    progress.empty()
                    status.empty()
                    
                    if not df.empty:
                        st.session_state.search_complete = True
                        results_count.success(f"✓ Found {len(df)} relevant articles")
                    else:
                        results_count.warning("No results found. Try modifying your search terms.")
                        
                except Exception as e:
                    st.error(f"An error occurred during the search: {str(e)}")
                    progress.empty()
                    status.empty()
            else:
                st.error("Please enter a search query")
        else:
            df = load_results()
        
        # Use the stored results from session state
        if st.session_state.original_results is not None:
            df = st.session_state.original_results
            
            # Main content
            st.header("Research Overview")
            
            # Add refresh button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("↻ Refresh"):
                    st.cache_data.clear()
                    st.rerun()
            
            # Summary statistics
            display_summary_stats(df)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                create_source_distribution(df)
            
            with col2:
                create_phase_distribution(df)
            
            # Filters
            st.header("Filter Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source_filter = st.multiselect(
                    "Source Type",
                    options=df['Source_Type'].unique(),
                    default=df['Source_Type'].unique()
                )
            
            with col2:
                phase_filter = st.multiselect(
                    "Development Phase",
                    options=[x for x in df['Development_Phase'].unique() if pd.notna(x)],
                    default=[x for x in df['Development_Phase'].unique() if pd.notna(x)]
                )
            
            with col3:
                study_filter = st.multiselect(
                    "Study Type",
                    options=[x for x in df['Study_Type'].unique() if pd.notna(x)],
                    default=[x for x in df['Study_Type'].unique() if pd.notna(x)]
                )
            
            # Filter the dataframe
            try:
                filtered_df = df[
                    (df['Source_Type'].isin(source_filter)) &
                    (df['Development_Phase'].isin(phase_filter) | df['Development_Phase'].isna()) &
                    (df['Study_Type'].isin(study_filter) | df['Study_Type'].isna())
                ]
            except Exception as e:
                st.error(f"Error filtering results: {str(e)}")
                filtered_df = df
            
            # Select/Deselect All/Execute buttons in a row (aligned to the right)
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col2:
                st.button("Select All", on_click=select_all)
            with col3:
                st.button("Deselect All", on_click=deselect_all)
            with col4:
                st.button("Execute", on_click=execute_selection)
            
            # Show success/warning messages
            if 'execute_success' in st.session_state and st.session_state.execute_success:
                st.success(f"Processed {len(st.session_state.processed_results)} selected articles")
                # Display total chunks stored
                if st.session_state.total_chunks_stored > 0:
                    st.info(f"Total chunks stored in vector database: {st.session_state.total_chunks_stored}")
                st.session_state.execute_success = False
            
            if 'execute_warning' in st.session_state and st.session_state.execute_warning:
                st.warning("No articles selected. Please select at least one article to process.")
                st.session_state.execute_warning = False
            
            # Display all results with selection buttons on the right
            st.header("Research Results")
            
            if filtered_df.empty:
                st.warning("No results match the selected filters. Try adjusting your filter criteria.")
            else:
                for i, (_, row) in enumerate(filtered_df.iterrows()):
                    # Use article title if available, otherwise create a descriptive title
                    if pd.notna(row['Title']) and row['Title']:
                        title = row['Title']
                    else:
                        # Fallback to previous method if title is not available
                        title = f"{row['Source_Type']}"
                        if pd.notna(row['Journal']) and row['Journal']:
                            title += f": {row['Journal']}"
                        elif 'URL' in row and row['URL']:
                            # Extract domain from URL for display
                            import re
                            domain = re.search(r'https?://(?:www\.)?([^/]+)', row['URL'])
                            if domain:
                                title += f": {domain.group(1)}"
                    
                    # Add selection button on the right
                    col1, col2 = st.columns([15, 1])
                    
                    with col1:
                        with st.expander(title):
                            st.markdown(f"[Open Source]({row['URL']})")
                            st.write("**Publication Date:**", row['Publication_Date'] if pd.notna(row['Publication_Date']) else "Not available")
                            if pd.notna(row['Authors']) and row['Authors']:
                                st.write("**Authors:**", row['Authors'])
                            if pd.notna(row['Journal']) and row['Journal']:
                                st.write("**Journal:**", row['Journal'])
                            if pd.notna(row['Development_Phase']) and row['Development_Phase']:
                                st.write("**Development Phase:**", row['Development_Phase'])
                            if pd.notna(row['Study_Type']) and row['Study_Type']:
                                st.write("**Study Type:**", row['Study_Type'])
                            # Add abstract section before summary
                            if pd.notna(row['Abstract']) and row['Abstract']:
                                st.write("**Abstract:**")
                                st.write(row['Abstract'])
                            
                            st.write("**Summary:**")
                            st.write(row['Summary'] if pd.notna(row['Summary']) else "No summary available")
                    
                    with col2:
                        # Use the index in the original dataframe to track selection
                        original_index = df.index[df['URL'] == row['URL']].tolist()[0]
                        
                        # Get current selection state
                        is_selected = st.session_state.selected_articles.get(original_index, False)
                        
                        # Create selection button
                        button_label = "☑" if is_selected else "☐"
                        st.button(
                            button_label, 
                            key=f"select_{original_index}",
                            on_click=toggle_selection,
                            args=(original_index,)
                        )
            
            # Get the dataframe to use for downloads
            display_df = st.session_state.processed_results if st.session_state.processed_results is not None else df
            
            # Download buttons for results - now below Research Results
            st.header("Download Articles")
            col1, col2 = st.columns(2)
            
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="research_results_{dt}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_str = display_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"research_results_{dt}.json",
                    mime="application/json"
                )
        else:
            st.info("Enter a search query and click 'Run Search' to start exploring research articles.")

if __name__ == "__main__":
    main()
