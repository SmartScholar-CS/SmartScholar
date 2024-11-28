# ui_components.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import google.generativeai as genai
import pandas as pd
from document_assistant.core import logger

class DocumentContainer:
    """Handles document visualization"""
    
    @staticmethod
    def render(doc_name: str, doc_info: Dict):
        """Render a complete document container"""
        try:
            with st.container():
                # Layout columns
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Document Information Section
                    st.markdown("### 📋 Document Information")
                    
                    # Title
                    if 'metadata' in doc_info and doc_info['metadata'].get('title'):
                        st.markdown(f"**Title:** {doc_info['metadata']['title']}")
                    
                    # Authors
                    if 'metadata' in doc_info and doc_info['metadata'].get('authors'):
                        st.markdown("**Authors:**")
                        authors = doc_info['metadata']['authors']
                        if authors:
                            for author in authors:
                                st.markdown(f"- {author}")
                        else:
                            st.markdown("*No authors found*")
                    
                    # Year
                    if 'metadata' in doc_info and doc_info['metadata'].get('year'):
                        st.markdown(f"**Year:** {doc_info['metadata']['year']}")

                    st.markdown("---")  # Divider
                    
                    # Summary and other sections
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
    # def render(doc_name: str, doc_info: Dict):
    #     """Render a complete document container"""
    #     try:
    #         with st.container():
    #             st.markdown(f"## 📄 {doc_name.split('.')[0]}")
                
    #             # Layout columns
    #             col1, col2 = st.columns([3, 2])
                
    #             with col1:
    #                 # Metadata and Summary
    #                 DocumentContainer._render_metadata(doc_info)
    #                 DocumentContainer._render_summary(doc_info)
    #                 DocumentContainer._render_classification(doc_info)
    #                 DocumentContainer._render_stats(doc_info['stats'])
                    
    #             with col2:
    #                 # Image and Similarities
    #                 DocumentContainer._render_image(doc_info)
    #                 DocumentContainer._render_similarities(doc_info)
                
    #             st.markdown("---")  # Divider
                
    #     except Exception as e:
    #         logger.error(f"Error rendering document container: {str(e)}")
    #         st.error("Error displaying document information")

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
        """Render document statistics with better formatting"""
        st.markdown("### 📊 Document Statistics")
        
        # Format numbers with appropriate scaling
        def format_number(n: int) -> str:
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)
        
        # Create metrics in columns with better spacing
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        
        with col1:
            word_count = format_number(stats['word_count'])
            st.metric("Words", word_count)
        
        with col2:
            char_count = format_number(stats['char_count'])
            st.metric("Characters", char_count)
        
        with col3:
            size_kb = stats['file_size']/1024
            size_display = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
            st.metric("Size", size_display)
        
        with col4:
            if 'num_pages' in stats:
                st.metric("Pages", stats['num_pages'])
            elif 'line_count' in stats:
                line_count = format_number(stats['line_count'])
                st.metric("Lines", line_count)
            
            # Additional stats based on file type
            if stats['type'] == 'docx' and 'table_count' in stats:
                st.markdown(f"**Tables:** {stats['table_count']}")
            if 'encoding' in stats:
                st.markdown(f"**Encoding:** {stats['encoding']}")

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
        """Render document similarities with improved formatting"""
        try:
            if 'similarities' in doc_info and doc_info['similarities']:
                st.markdown("### 🔄 Similarity Score")
                
                for other_doc, score in sorted(doc_info['similarities'].items(), 
                                            key=lambda x: x[1], 
                                            reverse=True):
                    # Truncate long document names
                    doc_display = other_doc.split('.')[0]
                    if len(doc_display) > 30:
                        doc_display = doc_display[:27] + "..."
                    
                    # Determine color based on score
                    score_color = ("green" if score > 70 else 
                                "orange" if score > 40 else "red")
                    
                    # Create similarity indicator with tooltip
                    st.markdown(f"""
                        <div style='
                            padding: 8px;
                            margin: 4px 0;
                            border-radius: 4px;
                            background-color: rgba(255,255,255,0.05);
                            border-left: 4px solid {score_color};
                            overflow: hidden;
                            text-overflow: ellipsis;
                            white-space: nowrap;
                        ' title='{other_doc.split('.')[0]}'>
                            <div style='
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            '>
                                <span style='
                                    max-width: 80%;
                                    overflow: hidden;
                                    text-overflow: ellipsis;
                                    white-space: nowrap;
                                '>
                                    <strong>{doc_display}</strong>
                                </span>
                                <span>{score:.1f}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            logger.error(f"Similarities rendering error: {str(e)}")

# In ui_components.py

class ChatInterface:
    """Handles the chat interface"""
    
    @staticmethod
    def render():
        """Render the chat interface"""
        st.markdown("### 💬 Chat with Documents")
        
        if not st.session_state.active_docs:
            st.info("Please upload and select documents to start chatting!")
            return
            
        # Show active documents
        st.info(f"📚 Currently analyzing: {', '.join(st.session_state.active_docs)}")
        
        # Create message containers
        chat_container = st.container()
        input_container = st.container()
        
        with input_container:
            prompt = st.chat_input("Ask me anything about your documents...")
        
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
                        try:
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
                        except Exception as e:
                            logger.error(f"Chat error: {str(e)}")
                            st.error(f"An error occurred: {str(e)}")
                
                st.rerun()

    @staticmethod
    def get_response(question: str, active_docs: List[str]) -> str:
        """Generate response using Gemini model"""
        try:
            if not hasattr(st.session_state, 'llm_model'):
                # Reinitialize if needed
                genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
                st.session_state.llm_model = genai.GenerativeModel('gemini-1.5-pro')
                st.session_state.llm_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )

            # Get context from documents
            context = ChatInterface.get_context(active_docs)
            
            # Create prompt
            prompt = f"""You are analyzing these documents:
            {context}

            Question: {question}

            Please provide a comprehensive answer based on ALL selected documents. 
            If referring to specific information, mention which document it came from.
            If you can't find the information in any document, say so clearly.
            Maintain a friendly, professional, conversational tone and explain in simple easy terms.
            
            Guidelines:
            - Cite specific documents when referencing information
            - Be clear about uncertain or missing information
            - Use examples from the documents when relevant
            - Keep the response focused and concise
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
            context_parts = []
            for doc_name in active_docs:
                doc = st.session_state.documents.get(doc_name)
                if doc:
                    # Add document information
                    context_parts.append(f"\nDocument: {doc_name}")
                    
                    # Add summary if available
                    if 'summary' in doc:
                        context_parts.append(f"Summary: {doc['summary']}")
                    
                    # Add classification if available
                    if 'classification' in doc:
                        topics = doc['classification']['topics'][:3]  # Top 3 topics
                        context_parts.append(f"Topics: {', '.join(topics)}")
                    
                    # Add content preview
                    content_preview = doc['content'][:1000].replace('\n', ' ')
                    context_parts.append(f"Content Preview: {content_preview}...")
                    context_parts.append("---")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context generation error: {str(e)}")
            return "Error retrieving document context"