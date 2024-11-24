# models.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import requests
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import io
import time
import concurrent.futures
from document_assistant.core import logger, API_CONFIG, HEADERS, DEVICE

class ModelManager:
    """Manages all AI model initialization and usage"""
    
    def __init__(self):
        self.classifier = None
        self.summary_model = None
        self.similarity_model = None
        self.image_generator = None
        self.initialized = False
        self.status = st.empty()

    def classify_document(self, text: str) -> Optional[Dict]:
        """Classify document content"""
        try:
            if not self.classifier:
                return None

            # Default categories for academic papers
            categories = [
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
            ]

            # Truncate text if too long
            if len(text) > 1024:
                text = text[:1024]

            # Run classification
            result = self.classifier(
                text,
                candidate_labels=categories,
                multi_label=True
            )

            # Format results
            topic_scores = list(zip(result['labels'], result['scores']))
            topic_scores.sort(key=lambda x: x[1], reverse=True)

            return {
                'topics': [t[0] for t in topic_scores],
                'scores': [t[1] for t in topic_scores]
            }

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return None
        
    def initialize_models(self) -> bool:
        """Initialize all models synchronously"""
        try:
            start_time = time.time()
            self.status.info("🚀 Initializing AI models...")

            # Initialize classifier
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-deberta-v3-small",
                    device=DEVICE
                )
                logger.info("Classifier initialized")
            except Exception as e:
                logger.error(f"Classifier initialization failed: {str(e)}")
                return False

            # Initialize similarity model
            try:
                self.similarity_model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2',
                    device=DEVICE
                )
                logger.info("Similarity model initialized")
            except Exception as e:
                logger.error(f"Similarity model initialization failed: {str(e)}")
                return False

            # Initialize summary model
            try:
                self.summary_model = SummaryGenerator()
            except Exception as e:
                logger.error(f"Summary model initialization failed: {str(e)}")
                return False

            # Initialize image generator
            try:
                self.image_generator = ImageGenerator()
            except Exception as e:
                logger.error(f"Image generator initialization failed: {str(e)}")
                return False

            # Mark as initialized
            self.initialized = True
            
            # Show success message
            elapsed_time = time.time() - start_time
            self.status.success(f"✅ Models initialized successfully! ({elapsed_time:.1f}s)")
            time.sleep(1)
            self.status.empty()
            
            return True
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            self.status.error(f"Failed to initialize models: {str(e)}")
            return False

    def check_initialized(self) -> bool:
        """Verify all models are properly initialized"""
        return (self.classifier is not None and 
                self.summary_model is not None and 
                self.similarity_model is not None and
                self.image_generator is not None and
                self.initialized)

class SummaryGenerator:
    """Handles document summarization"""
    
    def __init__(self):
        self.url = API_CONFIG['summary_url']
        self.headers = HEADERS
        self.timeout = 45  # Increased timeout
        self.max_retries = 3

    def generate_summary(self, text: str) -> str:
        try:
            # Split text into smaller chunks
            chunks = self._chunk_text(text)
            summaries = []

            for chunk in chunks:
                for attempt in range(self.max_retries):
                    try:
                        payload = {
                            "inputs": chunk,
                            "parameters": {
                                "max_length": 300,
                                "min_length": 50,
                                "do_sample": False
                            }
                        }

                        response = requests.post(
                            self.url,
                            headers=self.headers,
                            json=payload,
                            timeout=self.timeout
                        )

                        if response.status_code == 200:
                            summaries.append(response.json()[0]['summary_text'])
                            break
                        elif response.status_code == 503:
                            if attempt < self.max_retries - 1:
                                time.sleep(2 * (attempt + 1))  # Exponential backoff
                                continue
                    except requests.Timeout:
                        if attempt < self.max_retries - 1:
                            time.sleep(2 * (attempt + 1))
                            continue
                    except Exception as e:
                        logger.error(f"Chunk summary error: {str(e)}")
                        break

            if summaries:
                return " ".join(summaries)
            return "Could not generate summary."

        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

# models.py (continued)

class ImageGenerator:
    """Handles document visualization"""
    
    def __init__(self):
        self.url = API_CONFIG['image_url']
        self.headers = HEADERS

    def generate_image(self, text: str, title: str = "") -> Tuple[Optional[Image.Image], Optional[str]]:
        try:
            prompt = self._create_prompt(text, title)
            payload = {"inputs": prompt}

            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image, None
            elif response.status_code == 503:
                return None, "⏳ Rate limit reached. Please try again in a moment..."
            else:
                error_msg = response.json().get('error', 'Unknown error')
                return None, f"Image generation failed: {error_msg}"

        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None, str(e)

    def _create_prompt(self, text: str, title: str) -> str:
        """Create an optimized prompt for academic/technical content"""
        return f"""Create a professional concept visualization for an academic/technical document:
        Title: {title}
        Content Summary: {text[:300]}
        Requirements:
        - Professional academic/technical style
        - Abstract representation of key concepts
        - Clean, minimalist design
        - Scientific/technical aesthetic
        - Relevant visual metaphors
        - No text or labels
        - Single cohesive image
        - Subtle color scheme
        - High-quality rendering
        - Clear visual hierarchy
        """

class CitationExtractor:
    """Handles citation and reference extraction"""
    
    def __init__(self):
        self.patterns = {
            'doi': r'\b(10\.\d{4,}/[-._;()/:\w]+)\b',
            'authors': r'([A-Z][a-z]+(?:,? (?:and |& )?[A-Z][a-z]+)*)',
            'year': r'\b(19|20)\d{2}\b',
            'references': r'References?|Bibliography|Works Cited'
        }

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        try:
            # Extract basic metadata
            metadata = {
                'title': self._extract_title(text),
                'authors': self._extract_authors(text),
                'year': self._extract_year(text),
                'doi': self._extract_doi(text),
                'references': self._extract_references(text)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
            return {}

    def _extract_title(self, text: str) -> str:
        """Extract document title"""
        try:
            # Look for title in first few lines
            lines = text.split('\n')
            for line in lines[:5]:
                line = line.strip()
                # Title is usually the longest line in the first few lines
                if len(line) > 10 and len(line.split()) <= 20:
                    return line
            return "Title not found"
        except Exception as e:
            logger.error(f"Title extraction error: {str(e)}")
            return "Error extracting title"

    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names"""
        try:
            import re
            # Look for author pattern in first part of text
            first_page = text[:1000]
            authors = []
            
            # Common author section indicators
            author_section = re.search(r'Authors?:|by\s+', first_page, re.IGNORECASE)
            if author_section:
                # Extract names after the indicator
                names = re.findall(self.patterns['authors'], 
                                 first_page[author_section.end():])
                authors.extend(names)
            
            return list(set(authors))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Author extraction error: {str(e)}")
            return []

    def _extract_year(self, text: str) -> Optional[str]:
        """Extract publication year"""
        try:
            import re
            years = re.findall(self.patterns['year'], text)
            if years:
                return years[0]
            return None
        except Exception as e:
            logger.error(f"Year extraction error: {str(e)}")
            return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI if present"""
        try:
            import re
            doi_match = re.search(self.patterns['doi'], text)
            if doi_match:
                return doi_match.group(0)
            return None
        except Exception as e:
            logger.error(f"DOI extraction error: {str(e)}")
            return None

    def _extract_references(self, text: str) -> List[str]:
        """Extract references section"""
        try:
            import re
            # Find references section
            ref_match = re.split(self.patterns['references'], text, flags=re.IGNORECASE)
            if len(ref_match) > 1:
                # Get text after references header
                references = ref_match[-1].strip()
                # Split into individual references
                refs = [ref.strip() for ref in references.split('\n') 
                       if ref.strip() and len(ref.strip()) > 20]
                return refs
            return []
        except Exception as e:
            logger.error(f"References extraction error: {str(e)}")
            return []

class SimilarityCalculator:
    """Handles document similarity calculations"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def calculate_similarity(self, source_text: str, comparison_texts: List[str]) -> Optional[List[float]]:
        """Calculate semantic similarity between documents"""
        try:
            with torch.no_grad():
                # Generate embeddings
                source_embedding = self.model.encode(source_text, convert_to_tensor=True)
                comparison_embeddings = self.model.encode(comparison_texts, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarities = torch.nn.functional.cosine_similarity(
                    source_embedding.unsqueeze(0),
                    comparison_embeddings
                )
                
                # Convert to percentages
                return [float(score) * 100 for score in similarities]
                
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return None

    def find_similar_sections(self, source_text: str, target_text: str, 
                            chunk_size: int = 200) -> List[Tuple[str, float]]:
        """Find similar sections between documents"""
        try:
            # Split texts into chunks
            source_chunks = self._chunk_text(source_text, chunk_size)
            target_chunks = self._chunk_text(target_text, chunk_size)
            
            similar_sections = []
            
            # Compare chunks
            for source_chunk in source_chunks:
                scores = self.calculate_similarity(source_chunk, target_chunks)
                if scores:
                    max_score = max(scores)
                    if max_score > 70:  # High similarity threshold
                        similar_sections.append((source_chunk, max_score))
            
            return similar_sections
            
        except Exception as e:
            logger.error(f"Section similarity error: {str(e)}")
            return []

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks for comparison"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks