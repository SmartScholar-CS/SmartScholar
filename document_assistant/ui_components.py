# ui_components.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd
from document_assistant.core import logger

class DocumentContainer:
    """Handles document visualization"""
    
    @staticmethod
    def render(doc_name: str, doc_info: Dict):
        """Render a complete document container"""
        try:
            with st.container():
                st.markdown(f"## 📄 {doc_name.split('.')[0]}")
                
                # Layout columns
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Metadata and Summary
                    DocumentContainer._render_metadata(doc_info)
                    DocumentContainer._render_summary(doc_info)
                    DocumentContainer._render_classification(doc_info)
                    DocumentContainer._render_stats(doc_info['stats'])
                    
                with col2:
                    # Image and Similarities
                    DocumentContainer._render_image(doc_info)
                    DocumentContainer._render_similarities(doc_info)
                
                st.markdown("---")  # Divider
                
        except Exception as e:
            logger.error(f"Error rendering document container: {str(e)}")
            st.error("Error displaying document information")

    @staticmethod
    def _render_metadata(doc_info: Dict):
        """Render document metadata"""
        try:
            if 'metadata' in doc_info:
                metadata = doc_info['metadata']
                st.markdown("### 📋 Document Information")
                
                # Create two columns for metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    if metadata.get('title'):
                        st.markdown(f"**Title:** {metadata['title']}")
                    if metadata.get('author'):
                        st.markdown(f"**Author(s):** {metadata['author']}")
                    if metadata.get('creation_date'):
                        st.markdown(f"**Created:** {metadata['creation_date']}")
                
                with col2:
                    if metadata.get('subject'):
                        st.markdown(f"**Subject:** {metadata['subject']}")
                    if metadata.get('keywords'):
                        st.markdown(f"**Keywords:** {metadata['keywords']}")
                    
        except Exception as e:
            logger.error(f"Metadata rendering error: {str(e)}")

    @staticmethod
    def _render_summary(doc_info: Dict):
        """Render document summary"""
        try:
            if 'summary' in doc_info:
                st.markdown("### 📝 Summary")
                st.markdown("""
                    <style>
                    .summary-box {
                        height: 200px;
                        overflow-y: auto;
                        border: 1px solid #ddd;
                        padding: 10px;
                        border-radius: 5px;
                        text-align: justify;
                        background-color: rgba(255, 255, 255, 0.05);
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{doc_info["summary"]}</div>', 
                          unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Summary rendering error: {str(e)}")

    @staticmethod
    def _render_classification(doc_info: Dict):
        """Render document classification"""
        try:
            if 'classification' in doc_info:
                st.markdown("### 🏷️ Topic Classification")
                
                # Create DataFrame for topics
                df = pd.DataFrame({
                    'Topic': doc_info['classification']['topics'][:5],
                    'Confidence': [f"{score*100:.1f}%" for score in 
                                 doc_info['classification']['scores'][:5]]
                })
                
                # Display topics table
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                )
                
# ui_components.py (continued)
                # Create visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=[float(s.strip('%')) for s in df['Confidence']],
                        y=df['Topic'],
                        orientation='h',
                        marker_color='rgb(55, 83, 109)',
                        marker=dict(
                            color=px.colors.qualitative.Set3,
                            line=dict(color='rgb(8,48,107)', width=1.5)
                        )
                    )
                ])
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Confidence (%)",
                    yaxis_title="Topic",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            logger.error(f"Classification rendering error: {str(e)}")
            st.warning("Could not display classification results")

    @staticmethod
    def _render_stats(stats: Dict):
        """Render document statistics"""
        try:
            st.markdown("### 📊 Document Statistics")
            
            # Create metrics in columns
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Words", f"{stats['word_count']:,}")
            with cols[1]:
                st.metric("Characters", f"{stats['char_count']:,}")
            with cols[2]:
                st.metric("Size", f"{stats['file_size']/1024:.1f} KB")
            with cols[3]:
                if 'num_pages' in stats:
                    st.metric("Pages", stats['num_pages'])
                elif 'line_count' in stats:
                    st.metric("Lines", stats['line_count'])
            
            # Additional stats based on file type
            if stats['type'] == 'docx' and 'table_count' in stats:
                st.markdown(f"**Tables:** {stats['table_count']}")
            if 'encoding' in stats:
                st.markdown(f"**Encoding:** {stats['encoding']}")
                
        except Exception as e:
            logger.error(f"Statistics rendering error: {str(e)}")

    @staticmethod
    def _render_image(doc_info: Dict):
        """Render document visualization"""
        try:
            if 'image' in doc_info:
                st.markdown("### 🎨 Document Visualization")
                st.markdown("""
                    <style>
                    [data-testid="stImage"] {
                        border-radius: 10px;
                        overflow: hidden;
                        border: 1px solid rgba(128, 128, 128, 0.2);
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.image(
                    doc_info['image'],
                    use_container_width=True,
                    output_format="PNG"
                )
        except Exception as e:
            logger.error(f"Image rendering error: {str(e)}")

    @staticmethod
    def _render_similarities(doc_info: Dict):
        """Render document similarities"""
        try:
            if 'similarities' in doc_info and doc_info['similarities']:
                st.markdown("### 🔄 Similar Documents")
                
                similarities = doc_info['similarities']
                if similarities:  # Check if not empty
                    for other_doc, score in sorted(similarities.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True):
                        # Determine color based on score
                        score_color = ("green" if score > 70 else 
                                     "orange" if score > 40 else "red")
                        
                        # Create similarity indicator
                        st.markdown(f"""
                            <div style='
                                padding: 10px;
                                margin: 5px 0;
                                border-radius: 5px;
                                background-color: rgba(255,255,255,0.05);
                                border-left: 4px solid {score_color};
                            '>
                                <div style='display: flex; justify-content: space-between;'>
                                    <strong>{other_doc.split('.')[0]}</strong>
                                    <span>{score:.1f}% similar</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            logger.error(f"Similarities rendering error: {str(e)}")

class ChatInterface:
    """Handles the chat interface"""
    
    @staticmethod
    def render():
        """Render the chat interface"""
        try:
            st.markdown("### 💬 Chat with Documents")
            
            if not st.session_state.active_docs:
                st.info("Please upload and select documents to start chatting!")
                return
                
            st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
            
            # Create containers
            chat_container = st.container()
            input_container = st.container()
            
            # Handle input
            with input_container:
                prompt = st.chat_input("Ask me anything about your documents...")
            
            # Display messages
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Handle new message
                if prompt:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = ChatInterface.get_response(
                                prompt,
                                list(st.session_state.active_docs)
                            )
                            st.markdown(response)
                    
                    # Update chat history
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ])
                    
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Chat interface error: {str(e)}")
            st.error("Chat interface error. Please try again.")

    @staticmethod
    def get_response(question: str, active_docs: List[str]) -> str:
        """Generate response using Gemini model"""
        try:
            # Get context from documents
            context = ChatInterface.get_context(active_docs)
            
            # Create prompt
            prompt = f"""You are analyzing these documents:
            {context}

            Question: {question}

            Please provide a comprehensive answer based on the documents.
            - Reference specific documents when citing information
            - Be clear about uncertain or missing information
            - Keep the response focused and professional
            - Add relevant examples from the documents
            """

            # Generate response
            response = st.session_state.llm_model.generate_content(
                prompt,
                generation_config=st.session_state.llm_config
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    @staticmethod
    def get_context(active_docs: List[str]) -> str:
        """Combine context from multiple documents"""
        try:
            context = []
            for doc_name in active_docs:
                doc = st.session_state.documents.get(doc_name)
                if doc:
                    # Add document information
                    context.append(f"\nDocument: {doc_name}")
                    if 'title' in doc:
                        context.append(f"Title: {doc['title']}")
                    if 'summary' in doc:
                        context.append(f"Summary: {doc['summary']}")
                    # Add content preview
                    context.append(f"Content: {doc['content'][:1000]}...")
                    context.append("---")
            
            return "\n".join(context)
            
        except Exception as e:
            logger.error(f"Context generation error: {str(e)}")
            return "Error retrieving document context"