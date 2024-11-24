# document_processor.py
import asyncio
from document_assistant.core import logger, MAX_FILE_SIZE
from document_assistant.processors import PDFProcessor, DocxProcessor, DocProcessor, TxtProcessor
from document_assistant.models import ModelManager
from typing import Dict, List, Optional
from pathlib import Path
import streamlit as st

class DocumentProcessor:
    """Main document processing class"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.processors = {
            'pdf': PDFProcessor.process,
            'docx': DocxProcessor.process,
            'doc': DocProcessor.process,
            'txt': TxtProcessor.process
        }
        self.model_manager = model_manager
        print(f"DocumentProcessor initialized with model manager: {self.model_manager is not None}")

    def check_models(self) -> bool:
        if not self.model_manager:
            st.error("⚠️ Models not properly initialized")
            return False
        if not self.model_manager.check_initialized():
            st.warning("⌛ Waiting for models to initialize...")
            return False
        return True

    async def process_documents(self, files: List) -> None:
        """Process multiple documents asynchronously"""
        try:
            if not self.check_models():
                st.error("Models not fully initialized. Please wait...")
                return

            # Process files one at a time to maintain order
            for file in files:
                if file.name not in st.session_state.documents:
                    await self.process_single_document(file)
                    
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            st.error("An error occurred during document processing")

    def process_single_document(self, file) -> None:
        """Process a single document synchronously"""
        try:
            progress = st.empty()
            progress.text(f"📄 Processing {file.name}...")
            
            # 1. Extract text and metadata
            doc_info = self._process_text(file)
            if not doc_info:
                return

            # 2. Generate summary
            if hasattr(self.model_manager, 'summary_model'):
                progress.text("📝 Generating summary...")
                summary = self.model_manager.summary_model.generate_summary(doc_info['content'][:2000])
                if summary and summary != "Could not generate summary.":
                    doc_info['summary'] = summary

            # 3. Classify document
            if hasattr(self.model_manager, 'classifier'):
                progress.text("🏷️ Classifying content...")
                classification = self.model_manager.classifier(
                    doc_info['content'][:1500],
                    candidate_labels=[
                    # AI and Computer Science
                    "Artificial Intelligence", "Machine Learning", "Deep Learning",
                    "Natural Language Processing", "Computer Vision", "Robotics",
                    "Data Science", "Big Data", "Cloud Computing", "Cybersecurity",
                    "Internet of Things", "Blockchain", "Software Engineering",
                    
                    # Mathematics and Statistics
                    "Mathematics", "Statistics", "Linear Algebra",
                    "Probability Theory", "Mathematical Optimization",
                    "Graph Theory", "Number Theory", "Applied Mathematics",
                    
                    # Physical Sciences
                    "Physics", "Quantum Computing", "Theoretical Physics",
                    "Astrophysics", "Materials Science", "Nanotechnology",
                    "Chemistry", "Chemical Engineering", "Environmental Science",
                    
                    # Life Sciences
                    "Biology", "Biotechnology", "Genetics", "Neuroscience",
                    "Bioinformatics", "Molecular Biology", "Medical Science",
                    "Pharmaceutical Science", "Healthcare Technology",
                    
                    # Engineering
                    "Electrical Engineering", "Mechanical Engineering",
                    "Civil Engineering", "Aerospace Engineering",
                    "Control Systems", "Signal Processing", "Microelectronics",
                    
                    # Social Sciences and Business
                    "Economics", "Finance", "Business Analytics",
                    "Management Science", "Operations Research",
                    "Information Systems", "Digital Transformation",
                    
                    # Interdisciplinary Fields
                    "Cognitive Science", "Computational Biology",
                    "Human-Computer Interaction", "Information Theory",
                    "Systems Engineering", "Network Science",
                    "Quantum Information", "Sustainable Technology"
                ],
                    multi_label=True
                )
                if classification:
                    doc_info['classification'] = {
                        'topics': classification['labels'],
                        'scores': classification['scores']
                    }

            # 4. Generate image
            if hasattr(self.model_manager, 'image_generator'):
                progress.text("🎨 Creating visualization...")
                image, error = self.model_manager.image_generator.generate_image(
                    doc_info.get('summary', doc_info['content'][:500]),
                    Path(file.name).stem
                )
                if image:
                    doc_info['image'] = image

            # Store document
            st.session_state.documents[file.name] = doc_info
            st.session_state.active_docs.add(file.name)

            # 5. Calculate similarities
            if len(st.session_state.documents) > 1:
                progress.text("🔄 Analyzing similarities...")
                self._update_similarities(file.name, doc_info)

            progress.empty()
            st.success(f"✅ Processed {file.name} successfully!")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {str(e)}")

    # def _process_text(self, file) -> Optional[Dict]:
    #     """Process document text"""
    #     try:
    #         file_ext = Path(file.name).suffix[1:].lower()
    #         if file_ext not in self.processors:
    #             raise ValueError(f"Unsupported file format: {file_ext}")
            
    #         result = self.processors[file_ext](file)
    #         if not result or not result.get('content'):
    #             raise ValueError("No content could be extracted")
            
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"Text processing error: {str(e)}")
    #         st.error(f"Could not process {file.name}: {str(e)}")
    #         return None


    def _update_similarities(self, new_file: str, new_doc: Dict) -> None:
        """Update similarities for all documents"""
        try:
            docs = st.session_state.documents
            if len(docs) < 2:
                return

            # Initialize similarities dict
            if 'similarities' not in new_doc:
                new_doc['similarities'] = {}

            for doc_name, doc in docs.items():
                if doc_name != new_file:
                    if 'similarities' not in doc:
                        doc['similarities'] = {}

                    # Calculate similarity using similarity calculator
                    if hasattr(self.model_manager, 'similarity_calculator'):
                        scores = self.model_manager.similarity_calculator.calculate_similarity(
                            new_doc['content'][:1000],
                            [doc['content'][:1000]]
                        )

                        if scores and len(scores) > 0:
                            similarity = scores[0]
                            new_doc['similarities'][doc_name] = similarity
                            doc['similarities'][new_file] = similarity

        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")


    def _process_text(self, file) -> Optional[Dict]:
        """Process document text"""
        try:
            file_ext = Path(file.name).suffix[1:].lower()
            if file_ext not in self.processors:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            result = self.processors[file_ext](file)
            if not result or not result.get('content'):
                raise ValueError("No content could be extracted")
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            st.error(f"Could not process {file.name}: {str(e)}")
            return None