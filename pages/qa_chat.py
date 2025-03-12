import streamlit as st
from openai import OpenAI
import time
import sys
import os

# Add parent directory to path to import vector_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store import initialize_pinecone, get_relevant_context

# Set page config
st.set_page_config(page_title="Medical Research Q&A Chat", layout="wide")

# Initialize OpenAI client
client = OpenAI()

# Initialize Pinecone
try:
    pinecone_index = initialize_pinecone()
    pinecone_initialized = True
except Exception as e:
    pinecone_initialized = False
    pinecone_error = str(e)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your medical research assistant. How can I help you with your research questions today?"}
    ]

def generate_response(prompt):
    """Generate AI response using OpenAI API with vector search augmentation"""
    try:
        # Get relevant context from vector database if initialized
        context = ""
        if pinecone_initialized:
            with st.spinner("Searching knowledge base..."):
                context = get_relevant_context(prompt, pinecone_index)
        
        # If no context is found, return a message indicating no information is available
        if not context:
            return "I don't have any information about this topic in my knowledge base. Please try a different question or search for relevant articles in the Explorer."
        
        # Create system message with strict instructions to only use the provided context
        system_message = """You are a medical research assistant. 
        
IMPORTANT: ONLY provide information that is explicitly mentioned in the knowledge base context below. 
DO NOT use any information beyond what is provided in this context. 
If the context doesn't fully answer the question, acknowledge the limitations of the available information.
DO NOT make up or infer information that isn't explicitly stated in the context.

For each piece of information you use, you MUST cite the source by including the article title, URL, and authors at the end of your response in this format:

Source: [Article Title]
Link: [URL]
Authors: [Authors]

If multiple sources are used, list each one separately.

Here is the information from our knowledge base:

"""
        
        system_message += context
        
        # Add user message to the context
        messages = [
            {"role": "system", "content": system_message},
        ] + st.session_state.messages  # Include chat history
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,  # Further increased for longer responses with detailed citations
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        if not pinecone_initialized:
            return f"An error occurred: Vector database is not initialized. Please check your Pinecone API key. Error: {pinecone_error}"
        return f"An error occurred: {str(e)}"

def main():
    # Header with navigation link back to main app
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Medical Research Q&A Chat")
    with col2:
        if st.button("← Back to Explorer", use_container_width=True):
            st.switch_page("app.py")
    
    # Divider
    st.markdown("---")
    
    # Brief description
    st.markdown("""
    Ask questions about medical research, clinical trials, treatment approaches, or any other research-related topics.
    This AI assistant can help you understand complex medical concepts and find relevant information.
    """)
    
    # Display vector database status
    if pinecone_initialized:
        st.success("✅ Connected to knowledge base")
    else:
        st.error(f"❌ Not connected to knowledge base: {pinecone_error}")
        st.info("Please add PINECONE_API_KEY to your .env file to enable knowledge base search.")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask a question about medical research..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with a typing indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Generate response
            response = generate_response(prompt)
            
            # Simulate typing
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            # Display final response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
