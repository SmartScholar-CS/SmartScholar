# analysis_components.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import feedparser
import requests
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_theme_colors():
    """Get theme-specific colors for visualizations"""
    if st.get_option("theme.base") == "light":
        return {
            "bg_color": "white",
            "text_color": "black",
            "grid_color": "rgba(0,0,0,0.1)",
            "plot_bg": "rgba(255,255,255,0.95)",
            "marker_colors": px.colors.qualitative.Set3,
            "base_colors": px.colors.sequential.Viridis
        }
    else:
        return {
            "bg_color": "rgba(0,0,0,0)",
            "text_color": "white",
            "grid_color": "rgba(255,255,255,0.1)", 
            "plot_bg": "rgba(0,0,0,0)",
            "marker_colors": px.colors.qualitative.Set3,
            "base_colors": px.colors.sequential.Viridis
        }

class SessionStateManager:
    """Manages session state initialization and access"""
    
    @staticmethod
    def init_session_state():
        """Initialize all required session state variables"""
        if 'analysis_loading_state' not in st.session_state:
            st.session_state.analysis_loading_state = {
                'trends': {'status': 'pending', 'data': None},
                'geographic': {'status': 'pending', 'data': None},
                'impact': {'status': 'pending', 'data': None}
            }
        
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {
                'trends_data': None,
                'geographic_data': None,
                'impact_data': None,
                'last_update': None
            }

        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
            
        if 'previous_filters' not in st.session_state:
            st.session_state.previous_filters = None

    @staticmethod
    def update_loading_state(state_key: str, status: str, data=None):
        """Safely update loading state"""
        if 'analysis_loading_state' not in st.session_state:
            SessionStateManager.init_session_state()
        
        st.session_state.analysis_loading_state[state_key] = {
            'status': status,
            'data': data
        }

    @staticmethod
    def get_cached_data(cache_key: str, data_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        try:
            if st.session_state.data_cache.get(cache_key, {}).get(data_key):
                return st.session_state.data_cache[cache_key][data_key]
            return None
        except Exception as e:
            logger.error(f"Cache access error: {str(e)}")
            return None

    @staticmethod
    def set_cached_data(cache_key: str, data_key: str, data: pd.DataFrame):
        """Set data in cache"""
        try:
            if cache_key not in st.session_state.data_cache:
                st.session_state.data_cache[cache_key] = {}
            st.session_state.data_cache[cache_key][data_key] = data
        except Exception as e:
            logger.error(f"Cache update error: {str(e)}")

class CategoryMapper:
    """Maps research categories and handles category relationships"""
    
    # Main category groupings
    CATEGORY_GROUPS = {
        'AI_CS': ["Artificial Intelligence", "Machine Learning", "Deep Learning",
                 "Natural Language Processing", "Computer Vision", "Robotics",
                 "Data Science", "Big Data", "Cloud Computing", "Cybersecurity",
                 "Internet of Things", "Blockchain", "Software Engineering"],
                 
        'MATH_STATS': ["Mathematics", "Statistics", "Linear Algebra",
                     "Probability Theory", "Mathematical Optimization",
                     "Graph Theory", "Number Theory", "Applied Mathematics"],
                     
        'PHYSICS': ["Physics", "Quantum Computing", "Theoretical Physics",
                   "Astrophysics", "Materials Science", "Nanotechnology"],
                   
        'LIFE_SCI': ["Biology", "Biotechnology", "Genetics", "Neuroscience",
                    "Bioinformatics", "Molecular Biology", "Medical Science",
                    "Pharmaceutical Science", "Healthcare Technology"],
                    
        'ENGINEERING': ["Electrical Engineering", "Mechanical Engineering",
                      "Civil Engineering", "Aerospace Engineering",
                      "Control Systems", "Signal Processing", "Microelectronics"],
                      
        'SOCIAL_BIZ': ["Economics", "Finance", "Business Analytics",
                      "Management Science", "Operations Research",
                      "Information Systems", "Digital Transformation"],
                      
        'INTERDISCIPLINARY': ["Cognitive Science", "Computational Biology",
                            "Human-Computer Interaction", "Information Theory",
                            "Systems Engineering", "Network Science",
                            "Quantum Information", "Sustainable Technology"]
    }

    # ArXiv category mappings
    ARXIV_MAPPINGS = {
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cs.CL': 'Natural Language Processing',
        'cs.CV': 'Computer Vision',
        'cs.RO': 'Robotics',
        'cs.DB': 'Data Science',
        'cs.DC': 'Cloud Computing',
        'cs.CR': 'Cybersecurity',
        'cs.NI': 'Internet of Things',
        'cs.SE': 'Software Engineering',
        
        'math.PR': 'Probability Theory',
        'math.OC': 'Mathematical Optimization',
        'math.CO': 'Graph Theory',
        'math.NT': 'Number Theory',
        'stat.ML': 'Machine Learning',
        'stat.TH': 'Statistics',
        'stat.AP': 'Applied Mathematics',
        
        'physics.optics': 'Physics',
        'quant-ph': 'Quantum Computing',
        'physics.theo-ph': 'Theoretical Physics',
        'astro-ph': 'Astrophysics',
        'cond-mat.mtrl-sci': 'Materials Science',
        'physics.app-ph': 'Applied Physics',
        
        'q-bio': 'Biology',
        'q-bio.BM': 'Biomolecular',
        'q-bio.GN': 'Genetics',
        'q-bio.NC': 'Neuroscience',
        'q-bio.QM': 'Bioinformatics',
        'q-bio.MN': 'Molecular Biology'
    }

    @classmethod
    def get_main_category(cls, arxiv_category: str) -> str:
        """Get main category with improved handling"""
        try:
            # Try direct mapping first
            if arxiv_category in cls.ARXIV_MAPPINGS:
                return cls.ARXIV_MAPPINGS[arxiv_category]
                
            # Try to match from CATEGORY_GROUPS
            for group_name, categories in cls.CATEGORY_GROUPS.items():
                if arxiv_category in categories:
                    return arxiv_category
                    
            # Use prefix matching as fallback
            category_prefix = arxiv_category.split('.')[0]
            prefix_map = {
                'cs': 'Computer Science',
                'math': 'Mathematics',
                'stat': 'Statistics',
                'physics': 'Physics',
                'quant-ph': 'Quantum Computing',
                'cond-mat': 'Materials Science',
                'q-bio': 'Biology',
                'econ': 'Economics'
            }
            if category_prefix in prefix_map:
                return prefix_map[category_prefix]
                
            # Find closest match in CATEGORY_GROUPS
            for group, categories in cls.CATEGORY_GROUPS.items():
                for cat in categories:
                    if arxiv_category.lower() in cat.lower():
                        return cat
                        
            return arxiv_category  # Return original if no mapping found
            
        except Exception as e:
            logger.error(f"Category mapping error: {str(e)}")
            return "Other"

    @classmethod
    def get_category_group(cls, category: str) -> str:
        """Get category group with validation"""
        try:
            # Direct lookup in groups
            for group, categories in cls.CATEGORY_GROUPS.items():
                if category in categories:
                    return group
                    
            # Try mapping via ARXIV_MAPPINGS
            if category in cls.ARXIV_MAPPINGS:
                mapped = cls.ARXIV_MAPPINGS[category]
                for group, categories in cls.CATEGORY_GROUPS.items():
                    if mapped in categories:
                        return group
                        
            return 'OTHER'
            
        except Exception as e:
            logger.error(f"Category group lookup error: {str(e)}")
            return 'OTHER'

    @classmethod
    def get_subcategories(cls, main_category: str) -> List[str]:
        """Get subcategories with validation"""
        try:
            # Look for category in all groups
            for group, categories in cls.CATEGORY_GROUPS.items():
                if main_category in categories:
                    return [cat for cat in categories if cat != main_category]
                    
            # If not found, check if it's a group name
            if main_category in cls.CATEGORY_GROUPS:
                return cls.CATEGORY_GROUPS[main_category]
                
            return []
            
        except Exception as e:
            logger.error(f"Subcategory lookup error: {str(e)}")
            return []
        
    @classmethod
    def get_arxiv_categories(cls, main_categories: List[str]) -> List[str]:
        """Get relevant arXiv categories for main categories"""
        arxiv_cats = []
        for main_cat in main_categories:
            cats = [
                cat for cat, mapped in cls.ARXIV_MAPPINGS.items()
                if cls.get_category_group(mapped) == main_cat
            ]
            arxiv_cats.extend(cats)
        return arxiv_cats or ['cs.AI', 'cs.ML', 'physics', 'math']

class DataLoader:
    """Handles asynchronous loading of research data"""
    
    def __init__(self, arxiv_endpoint: str):
        """Initialize the data loader"""
        self.arxiv_endpoint = arxiv_endpoint
        self.category_mapper = CategoryMapper()
        self.session = requests.Session()
        
    async def load_research_data(self, year: int, period: str, categories: List[str], 
                            sort_by: str) -> Dict[str, pd.DataFrame]:
        """Load research data with caching and validation"""
        try:
            # Create cache key
            cache_key = f"{year}_{period}_{'-'.join(categories)}_{sort_by}"
            print(f"\nDebug - Loading research data with cache key: {cache_key}")
            
            # Clear cache if filters changed
            if hasattr(st.session_state, 'last_cache_key') and st.session_state.last_cache_key != cache_key:
                print("Filters changed, clearing cache")
                SessionStateManager.init_session_state()
            st.session_state.last_cache_key = cache_key
            
            # Generate new data
            print("\nGenerating new data...")
            trends_data = self._fetch_arxiv_data(year, period, categories, sort_by)
            citation_data = self._fetch_citation_data(year, categories)
            
            # Process and combine data
            processed_data = {}
            
            if not trends_data.empty:
                processed_data['trends'] = trends_data
                geo_data = self._calculate_geographic_distribution(trends_data)
                if not geo_data.empty:
                    processed_data['geographic'] = geo_data
                    
            if not citation_data.empty:
                processed_data['impact'] = citation_data
            
            print("\nProcessed Data Summary:")
            print(f"Trends data shape: {trends_data.shape if not trends_data.empty else 'Empty'}")
            print(f"Geographic data: {len(processed_data.get('geographic', []))} countries")
            print(f"Impact data shape: {citation_data.shape if not citation_data.empty else 'Empty'}")
            
            return processed_data
                
        except Exception as e:
            logger.error(f"Research data loading error: {str(e)}")
            return {}
            
    def _fetch_arxiv_data(self, year: int, period: str, categories: List[str], 
                        sort_by: str) -> Optional[pd.DataFrame]:
        """Generate detailed arXiv data with subcategories"""
        try:
            print(f"\nDebug - Generating detailed arXiv data:")
            print(f"Year: {year}, Period: {period}")
            print(f"Categories: {categories}")
            
            sample_data = []
            
            for category in categories:
                print(f"\nProcessing main category: {category}")
                
                # Get subcategories
                if category in self.category_mapper.CATEGORY_GROUPS:
                    subcategories = self.category_mapper.CATEGORY_GROUPS[category]
                else:
                    subcategories = self.category_mapper.get_subcategories(category)
                    
                print(f"Found {len(subcategories)} subcategories")
                
                # Set up date range
                days = {
                    "Last Week": 7,
                    "Last Month": 30,
                    "Last 3 Months": 90,
                    "Last 6 Months": 180,
                    "Last Year": 365
                }
                period_days = days[period]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=period_days)
                
                # Generate data for each subcategory
                for subcategory in subcategories:
                    # Generate varied paper counts for each subcategory
                    n_papers = np.random.randint(200, 800)
                    print(f"  Generating {n_papers} papers for {subcategory}")
                    
                    # Generate papers with dates
                    for _ in range(n_papers):
                        pub_date = start_date + timedelta(
                            days=np.random.randint(0, period_days)
                        )
                        
                        paper = {
                            'topic': subcategory,
                            'main_category': category,
                            'published': pub_date,
                            'count': np.random.randint(1, 5),  # Citations per paper
                            'year': year
                        }
                        sample_data.append(paper)
            
            # Create DataFrame
            df = pd.DataFrame(sample_data)
            
            # Print detailed summary
            print("\nGenerated Data Summary:")
            print(df.groupby(['main_category', 'topic'])['count'].agg(['count', 'sum']))
            
            # Aggregate by topic
            if not df.empty:
                trends = df.groupby(['main_category', 'topic']).agg({
                    'count': 'sum',
                    'published': 'max',
                    'year': 'first'
                }).reset_index()
                
                # Calculate metrics
                trends['days_since_last'] = (
                    pd.Timestamp.now() - trends['published']
                ).dt.total_seconds() / (24 * 3600)
                trends['growth'] = 1 / np.maximum(trends['days_since_last'], 1)
                
                print("\nProcessed Trends Data:")
                print(trends)
                
                return trends
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Detailed arXiv data generation error: {str(e)}")
            return pd.DataFrame()
            
    def _fetch_citation_data(self, year: int, categories: List[str]) -> Optional[pd.DataFrame]:
        """Generate detailed citation data for all subcategories"""
        try:
            print(f"\nDebug - Generating detailed citation data:")
            print(f"Year: {year}")
            print(f"Categories: {categories}")
            
            citation_data = []
            
            for category in categories:
                print(f"\nProcessing category: {category}")
                
                # Get subcategories
                if category in self.category_mapper.CATEGORY_GROUPS:
                    subcategories = self.category_mapper.CATEGORY_GROUPS[category]
                else:
                    subcategories = self.category_mapper.get_subcategories(category)
                    
                print(f"Found {len(subcategories)} subcategories")
                
                for subcategory in subcategories:
                    # Generate metrics with year-based variations
                    base_papers = np.random.randint(500, 2000)
                    year_factor = 1 - ((datetime.now().year - year) * 0.1)
                    paper_count = int(base_papers * max(0.2, year_factor))
                    
                    # Calculate citations with some randomness
                    citation_factor = np.random.uniform(2, 8)  # Different impact factors
                    citation_count = int(paper_count * citation_factor)
                    
                    # Add subcategory data
                    entry = {
                        'field': subcategory,
                        'main_category': category,
                        'field_level': 'sub',
                        'paper_count': paper_count,
                        'citation_count': citation_count,
                        'year': year,
                        'impact_factor': citation_factor
                    }
                    citation_data.append(entry)
                    
                    print(f"  {subcategory}: {paper_count} papers, {citation_count} citations")
            
            if citation_data:
                df = pd.DataFrame(citation_data)
                
                print("\nGenerated Citation Data Summary:")
                summary = df.groupby('main_category').agg({
                    'paper_count': 'sum',
                    'citation_count': 'sum',
                    'impact_factor': 'mean'
                })
                print(summary)
                
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Detailed citation data generation error: {str(e)}")
            return pd.DataFrame()
                        
    def _generate_sample_citations(self) -> int:
        """Generate realistic sample citation counts"""
        return int(np.random.lognormal(mean=5, sigma=1.5))

    def _generate_sample_papers(self) -> int:
        """Generate realistic sample paper counts"""
        return int(np.random.lognormal(mean=4, sigma=1.0))

    def _process_research_data(self, arxiv_data: pd.DataFrame, citation_data: pd.DataFrame) -> Dict[str, Any]:
        """Process research data with proper error handling"""
        try:
            processed_data = {}
            
            # Process trends data
            if arxiv_data is not None:
                processed_data['trends'] = arxiv_data
                processed_data['geographic'] = self._calculate_geographic_distribution(arxiv_data)
            
            # Process impact data
            if citation_data is not None:
                processed_data['impact'] = citation_data
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            return {
                'trends': pd.DataFrame(),
                'geographic': pd.DataFrame(),
                'impact': citation_data if citation_data is not None else pd.DataFrame()
            }

    def _calculate_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate research trends safely"""
        try:
            if data is None or data.empty:
                return pd.DataFrame()
                
            trends = data.groupby('main_category').agg({
                'title': 'count',
                'published': lambda x: (datetime.now() - x.max()).total_seconds() / (24 * 3600)
            }).reset_index()
            
            trends.columns = ['topic', 'count', 'days_since_last']
            trends['days_since_last'] = trends['days_since_last'].replace(0, 1)
            trends['growth'] = 1 / trends['days_since_last']
            
            return trends
            
        except Exception as e:
            logger.error(f"Trends calculation error: {str(e)}")
            return pd.DataFrame()

    def _calculate_geographic_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive geographic distribution for all categories"""
        try:
            if data.empty:
                return pd.DataFrame()
                
            print("\nDebug - Generating comprehensive geographic distribution")
            
            # Define a broader set of countries
            countries = {
                # Major research hubs
                'USA': ('United States', 0.20),
                'CHN': ('China', 0.18),
                'GBR': ('United Kingdom', 0.08),
                'DEU': ('Germany', 0.07),
                'JPN': ('Japan', 0.06),
                'FRA': ('France', 0.05),
                
                # Strong research nations
                'CAN': ('Canada', 0.04),
                'ITA': ('Italy', 0.03),
                'ESP': ('Spain', 0.03),
                'AUS': ('Australia', 0.03),
                'KOR': ('South Korea', 0.03),
                'NLD': ('Netherlands', 0.02),
                
                # Emerging research powerhouses
                'IND': ('India', 0.04),
                'BRA': ('Brazil', 0.02),
                'RUS': ('Russia', 0.02),
                'SGP': ('Singapore', 0.02),
                'ISR': ('Israel', 0.02),
                'CHE': ('Switzerland', 0.02),
                
                # Additional research active nations
                'SWE': ('Sweden', 0.01),
                'DNK': ('Denmark', 0.01),
                'NOR': ('Norway', 0.01),
                'FIN': ('Finland', 0.01),
                'AUT': ('Austria', 0.01),
                'BEL': ('Belgium', 0.01)
            }
            
            # Get all unique topics from trends data
            all_topics = set(data['topic'].unique())
            total_papers = data['count'].sum()
            
            print(f"\nTotal papers for distribution: {total_papers}")
            print(f"Number of topics: {len(all_topics)}")
            
            geo_data = []
            
            # Generate data for each country
            for code, (name, weight) in countries.items():
                # Base papers for country
                base_papers = int(total_papers * weight * np.random.uniform(0.9, 1.1))
                country_total_citations = 0
                topic_data = []
                
                print(f"\nProcessing {name} ({code}):")
                print(f"Base papers: {base_papers}")
                
                # Generate data for each topic
                for topic in all_topics:
                    # Calculate papers for this topic
                    topic_papers = int(base_papers * np.random.uniform(0.02, 0.15))
                    impact_factor = np.random.lognormal(1, 0.5)
                    topic_citations = int(topic_papers * impact_factor)
                    
                    topic_data.append({
                        'topic': topic,
                        'papers': topic_papers,
                        'citations': topic_citations,
                        'impact_factor': impact_factor
                    })
                    
                    country_total_citations += topic_citations
                    print(f"  {topic}: {topic_papers} papers, {topic_citations} citations")
                
                # Create country entry with detailed topic breakdown
                country_data = {
                    'country_code': code,
                    'country_name': name,
                    'papers': sum(t['papers'] for t in topic_data),
                    'citations': country_total_citations,
                    'research_intensity': base_papers / 1000000,
                    'topic_breakdown': topic_data
                }
                geo_data.append(country_data)
                
                print(f"Total for {name}: {country_data['papers']} papers, {country_data['citations']} citations")
            
            # Convert to DataFrame with expanded topic columns
            expanded_data = []
            for country in geo_data:
                row = {
                    'country_code': country['country_code'],
                    'country_name': country['country_name'],
                    'total_papers': country['papers'],
                    'total_citations': country['citations'],
                    'research_intensity': country['research_intensity']
                }
                
                # Add topic-specific columns
                for topic_data in country['topic_breakdown']:
                    prefix = topic_data['topic'].replace(' ', '_').lower()
                    row[f'{prefix}_papers'] = topic_data['papers']
                    row[f'{prefix}_citations'] = topic_data['citations']
                    row[f'{prefix}_impact'] = topic_data['impact_factor']
                
                expanded_data.append(row)
            
            df = pd.DataFrame(expanded_data)
            print("\nFinal DataFrame shape:", df.shape)
            print("Columns:", df.columns.tolist())
            
            return df
            
        except Exception as e:
            logger.error(f"Geographic data generation error: {str(e)}")
            print(f"Error details: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_impact_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate impact metrics safely"""
        try:
            if data is None or data.empty:
                return pd.DataFrame()
                
            data['impact_factor'] = data['citation_count'] / data['paper_count'].replace(0, 1)
            data['h_index'] = data.apply(lambda x: self._calculate_h_index(x['citation_count']), axis=1)
            
            return data
            
        except Exception as e:
            logger.error(f"Impact metrics calculation error: {str(e)}")
            return pd.DataFrame()

    def _calculate_h_index(self, citations: int) -> int:
        """Calculate h-index from citation count"""
        try:
            citation_dist = np.random.lognormal(mean=2, sigma=1, size=citations)
            citation_dist = sorted([int(c) for c in citation_dist], reverse=True)
            
            h = 0
            for i, cites in enumerate(citation_dist, 1):
                if cites >= i:
                    h = i
                else:
                    break
                    
            return h
            
        except Exception as e:
            logger.error(f"H-index calculation error: {str(e)}")
            return 0

    def _get_country_name(self, code: str) -> str:
        """Get country name from code"""
        country_names = {
            'USA': 'United States',
            'GBR': 'United Kingdom',
            'DEU': 'Germany',
            'FRA': 'France',
            'JPN': 'Japan',
            'CHN': 'China',
            'IND': 'India',
            'CAN': 'Canada',
            'AUS': 'Australia',
            'BRA': 'Brazil'
        }
        return country_names.get(code, code)

class AnalysisInterface:
    """Handles research analysis visualization with user controls"""
    
    def __init__(self):
        """Initialize the Analysis Interface"""
        try:
            # Initialize session state
            SessionStateManager.init_session_state()
            
            # Initialize API endpoints
            self.arxiv_endpoint = 'http://export.arxiv.org/api/query'
            
            # Initialize components
            self.data_loader = DataLoader(self.arxiv_endpoint)
            self.category_mapper = CategoryMapper()
            self.theme = get_theme_colors()
            
            logger.info("AnalysisInterface initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            st.error("Error initializing interface")
            raise

    def _render_filters(self) -> Dict[str, Any]:
        """Render filter controls and return selected values"""
        try:
            st.subheader("Analysis Filters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Year filter
                current_year = datetime.now().year
                years = list(range(current_year, current_year - 5, -1))
                selected_year = st.selectbox(
                    "Select Year",
                    years,
                    index=0
                )
                
            with col2:
                # Time period filter
                time_periods = ["Last Week", "Last Month", "Last 3 Months", 
                              "Last 6 Months", "Last Year"]
                selected_period = st.selectbox(
                    "Time Period",
                    time_periods,
                    index=2
                )
                
            with col3:
                # Sort options
                sort_options = {
                    "Most Recent": "submittedDate",
                    "Most Cited": "citationCount",
                    "Most Relevant": "relevance"
                }
                selected_sort = st.selectbox(
                    "Sort By",
                    list(sort_options.keys()),
                    index=0
                )

            # Category filter
            main_categories = list(CategoryMapper.CATEGORY_GROUPS.keys())
            selected_categories = st.multiselect(
                "Research Fields",
                main_categories,
                default=["AI_CS", "PHYSICS"],
                help="Select one or more research fields to analyze"
            )
            
            # Create filters dictionary
            filters = {
                'year': selected_year,
                'period': selected_period,
                'categories': selected_categories,
                'sort_by': sort_options[selected_sort]
            }
            
            # Check if filters changed
            if st.session_state.previous_filters != filters:
                st.session_state.previous_filters = filters
                SessionStateManager.init_session_state()
            
            return filters
            
        except Exception as e:
            logger.error(f"Filter rendering error: {str(e)}")
            return None

    @staticmethod
    def render():
        """Render the analysis interface"""
        try:
            interface = AnalysisInterface()
            
            # Add custom styling
            st.markdown("""
                <style>
                .loading-overlay {
                    position: relative;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    background: rgba(0,0,0,0.05);
                }
                .loading-spinner {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 200px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Main title
            st.title("📊 Research Analysis Dashboard")
            
            # Render filters and get values
            filters = interface._render_filters()
            
            if not filters:
                st.error("Error loading filters")
                return
                
            # Create tabs
            tab_trends, tab_geo, tab_impact = st.tabs([
                "📈 Research Trends",
                "🌍 Geographic Analysis",
                "📊 Impact Metrics"
            ])
            
            # Load data
            with st.spinner("Loading research data..."):
                data = asyncio.run(interface.data_loader.load_research_data(
                    year=filters['year'],
                    period=filters['period'],
                    categories=filters['categories'],
                    sort_by=filters['sort_by']
                ))
                
            if not data:
                st.error("Error loading research data")
                return
                
            # Render tabs with available data
            with tab_trends:
                if 'trends' in data and not data['trends'].empty:
                    interface._render_trends_section(data['trends'])
                else:
                    st.info("No trends data available for selected filters")
                    
            with tab_geo:
                if 'geographic' in data and not data['geographic'].empty:
                    interface._render_geographic_section(data['geographic'])
                else:
                    st.info("No geographic data available for selected filters")
                    
            with tab_impact:
                if 'impact' in data and not data['impact'].empty:
                    interface._render_impact_section(data['impact'])
                else:
                    st.info("No impact data available for selected filters")
                    
        except Exception as e:
            logger.error(f"Interface rendering error: {str(e)}")
            st.error("Error rendering analysis interface")

    def _render_trends_section(self, data: pd.DataFrame):
        """Render trends section with improved visualization"""
        try:
            if data.empty:
                st.info("No trends data available")
                return
                
            st.header("📈 Research Trends Analysis")

            # Get category colors
            category_colors = self._get_category_colors()

            # Prepare data with colors
            plot_data = data.copy()
            colors = []
            
            for _, row in plot_data.iterrows():
                cat_colors = category_colors.get(row['main_category'], 
                                            category_colors['SOCIAL_BIZ'])  # Default to gray
                if cat_colors['palette']:
                    color_idx = len(colors) % len(cat_colors['palette'])
                    colors.append(cat_colors['palette'][color_idx])
                else:
                    colors.append(cat_colors['base'])

            # Metric cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Publications",
                    f"{plot_data['count'].sum():,}",
                    delta=f"{(plot_data['growth'].mean()*100):.1f}% growth"
                )
            
            with col2:
                st.metric(
                    "Active Fields",
                    len(plot_data['main_category'].unique()),
                    delta=f"{len(plot_data['topic'].unique())} topics"
                )
                
            top_topic = plot_data.loc[plot_data['count'].idxmax()]
            with col3:
                st.metric(
                    "Leading Topic",
                    top_topic['topic'],
                    delta=f"{top_topic['count']:,} papers"
                )

            # Research Activity Chart
            st.subheader("Research Activity by Field")
            
            fig = go.Figure()

            # Add bars with category colors
            for category in plot_data['main_category'].unique():
                cat_data = plot_data[plot_data['main_category'] == category]
                
                fig.add_trace(go.Bar(
                    name=category,
                    x=cat_data['topic'],
                    y=cat_data['count'],
                    text=cat_data['count'].apply(lambda x: f"{x:,}"),
                    textposition='auto',
                    marker_color=category_colors[category]['base'],
                    customdata=cat_data['main_category'],
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Papers: %{y:,}<br>" +
                        "Category: %{customdata}<br>" +
                        "<extra></extra>"
                    )
                ))

            fig.update_layout(
                barmode='group',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=50, b=100),
                xaxis=dict(
                    title="Research Field",
                    tickangle=-45,
                    showgrid=False
                ),
                yaxis=dict(
                    title="Number of Publications",
                    gridcolor='rgba(128,128,128,0.2)',
                    showgrid=True
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Growth Analysis
            st.subheader("Research Growth Analysis")
            
            fig = go.Figure()

            for category in plot_data['main_category'].unique():
                cat_data = plot_data[plot_data['main_category'] == category]
                
                fig.add_trace(go.Scatter(
                    name=category,
                    x=cat_data['growth'] * 100,
                    y=cat_data['count'],
                    mode='markers+text',
                    text=cat_data['topic'],
                    textposition="top center",
                    marker=dict(
                        size=cat_data['count'] / cat_data['count'].max() * 50 + 10,
                        color=category_colors[category]['base'],
                        showscale=False,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Growth Rate: %{x:.1f}%<br>" +
                        "Papers: %{y:,}<br>" +
                        "<extra></extra>"
                    )
                ))

            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(
                    title="Growth Rate (%)",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    title="Number of Publications",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Trends visualization error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error rendering trends section")

    def _get_world_map_data(self) -> Dict[str, str]:
        """Get complete world map country data"""
        return {
            # North America
            'USA': 'United States', 'CAN': 'Canada', 'MEX': 'Mexico',

            # Europe
            'GBR': 'United Kingdom', 'DEU': 'Germany', 'FRA': 'France', 
            'ITA': 'Italy', 'ESP': 'Spain', 'NLD': 'Netherlands',
            'BEL': 'Belgium', 'CHE': 'Switzerland', 'AUT': 'Austria',
            'SWE': 'Sweden', 'NOR': 'Norway', 'DNK': 'Denmark',
            'FIN': 'Finland', 'PRT': 'Portugal', 'IRL': 'Ireland',
            'GRC': 'Greece', 'POL': 'Poland', 'CZE': 'Czech Republic',
            'HUN': 'Hungary', 'SVK': 'Slovakia', 'ROU': 'Romania',
            'BGR': 'Bulgaria', 'HRV': 'Croatia', 'SVN': 'Slovenia',
            'LUX': 'Luxembourg', 'EST': 'Estonia', 'LVA': 'Latvia',
            'LTU': 'Lithuania',

            # Asia
            'CHN': 'China', 'JPN': 'Japan', 'KOR': 'South Korea',
            'IND': 'India', 'RUS': 'Russia', 'TWN': 'Taiwan',
            'SGP': 'Singapore', 'ISR': 'Israel', 'TUR': 'Turkey',
            'IRN': 'Iran', 'SAU': 'Saudi Arabia', 'ARE': 'UAE',
            'MYS': 'Malaysia', 'THA': 'Thailand', 'VNM': 'Vietnam',
            'IDN': 'Indonesia', 'PAK': 'Pakistan', 'BGD': 'Bangladesh',
            'PHL': 'Philippines', 'KAZ': 'Kazakhstan',

            # Oceania
            'AUS': 'Australia', 'NZL': 'New Zealand',

            # South America
            'BRA': 'Brazil', 'ARG': 'Argentina', 'CHL': 'Chile',
            'COL': 'Colombia', 'PER': 'Peru', 'VEN': 'Venezuela',
            'URY': 'Uruguay', 'PRY': 'Paraguay', 'BOL': 'Bolivia',
            'ECU': 'Ecuador',

            # Africa
            'ZAF': 'South Africa', 'EGY': 'Egypt', 'MAR': 'Morocco',
            'NGA': 'Nigeria', 'KEN': 'Kenya', 'ETH': 'Ethiopia',
            'GHA': 'Ghana', 'TUN': 'Tunisia', 'SEN': 'Senegal',
            'TZA': 'Tanzania', 'UGA': 'Uganda', 'DZA': 'Algeria',
            'CIV': 'Ivory Coast', 'CMR': 'Cameroon', 'SDN': 'Sudan',
            'ZWE': 'Zimbabwe', 'AGO': 'Angola', 'MOZ': 'Mozambique',
            'NAM': 'Namibia', 'BWA': 'Botswana'
        }

    def _get_category_colors(self) -> Dict[str, Dict[str, Any]]:
        """Get category colors with gradients"""
        return {
            'AI_CS': {
                'base': '#1f77b4',  # Blue
                'palette': [
                    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
                    '#2ca02c', '#98df8a', '#d62728', '#ff9896',
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                    '#e377c2', '#f7b6d2'
                ],
                'gradient': 'Blues'
            },
            'PHYSICS': {
                'base': '#2ca02c',  # Green
                'palette': [
                    '#2ca02c', '#98df8a', '#b5cf6b', '#cedb9c',
                    '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94'
                ],
                'gradient': 'Greens'
            },
            'MATH_STATS': {
                'base': '#ff7f0e',  # Orange
                'palette': [
                    '#ff7f0e', '#ffbb78', '#ff9896', '#ffd8b1',
                    '#ffa07a', '#ffc04c', '#ffd700', '#ffefd5'
                ],
                'gradient': 'Oranges'
            },
            'LIFE_SCI': {
                'base': '#9467bd',  # Purple
                'palette': [
                    '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                    '#e377c2', '#f7b6d2', '#cc79a7', '#fbb4ae'
                ],
                'gradient': 'Purples'
            },
            'ENGINEERING': {
                'base': '#e377c2',  # Pink
                'palette': [
                    '#e377c2', '#f7b6d2', '#fbb4ae', '#fccde5',
                    '#fb8072', '#fc9272', '#fd8d3c', '#fdae6b'
                ],
                'gradient': 'RdPu'
            },
            'SOCIAL_BIZ': {
                'base': '#7f7f7f',  # Gray
                'palette': [
                    '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                    '#17becf', '#9edae5', '#969696', '#bdbdbd'
                ],
                'gradient': 'Greys'
            }
        }

    def _render_geographic_section(self, data: pd.DataFrame):
        """Render geographic analysis visualizations"""
        try:
            if data.empty:
                st.info("No geographic data available")
                return
                
            st.header("🌍 Geographic Research Distribution")

            # Metrics
            total_countries = len(data)
            total_papers = data['total_papers'].sum()
            avg_impact = data['total_citations'].sum() / data['total_papers'].sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Contributing Countries",
                    total_countries,
                    help="Number of countries with research output"
                )
            
            with col2:
                st.metric(
                    "Global Publications",
                    f"{total_papers:,}",
                    help="Total number of publications"
                )
            
            with col3:
                st.metric(
                    "Average Impact",
                    f"{avg_impact:.2f}",
                    help="Citations per paper"
                )

            # Create choropleth map
            st.subheader("Global Research Distribution")
            fig = go.Figure(data=go.Choropleth(
                locations=data['country_code'],
                z=data['total_papers'],
                text=data['country_name'],
                colorscale='Viridis',
                colorbar_title="Publications",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Publications: %{z:,}<br>" +
                    "<extra></extra>"
                )
            ))

            fig.update_layout(
                title='Research Output by Country',
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                ),
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Top Countries Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top Research Producers")
                top_producers = data.nlargest(10, 'total_papers')
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=top_producers['country_name'],
                        x=top_producers['total_papers'],
                        orientation='h',
                        text=top_producers['total_papers'].apply(lambda x: f"{x:,}"),
                        textposition='auto',
                        marker_color=px.colors.qualitative.Set3[:len(top_producers)]
                    )
                ])

                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Number of Publications",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Research Impact Leaders")
                # Calculate impact factors
                impact_data = data.copy()
                impact_data['impact_factor'] = impact_data['total_citations'] / impact_data['total_papers']
                top_impact = impact_data.nlargest(10, 'impact_factor')
                
                fig = go.Figure(data=[
                    go.Scatter(
                        x=top_impact['total_papers'],
                        y=top_impact['total_citations'],
                        mode='markers+text',
                        text=top_impact['country_name'],
                        textposition="top center",
                        marker=dict(
                            size=top_impact['impact_factor'] * 20,
                            color=top_impact['impact_factor'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Impact Factor")
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>" +
                            "Papers: %{x:,}<br>" +
                            "Citations: %{y:,}<br>" +
                            "<extra></extra>"
                        )
                    )
                ])

                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Number of Publications",
                    yaxis_title="Number of Citations",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            # Detailed Research Breakdown
            st.subheader("Research Field Distribution by Country")
            
            # Get topic columns
            topic_cols = [col for col in data.columns if col.endswith('_papers')]
            topics = [col.replace('_papers', '').replace('_', ' ').title() for col in topic_cols]
            
            # Create heatmap data
            top_countries = data.nlargest(15, 'total_papers')
            heatmap_data = []
            
            for idx, country in enumerate(top_countries['country_name']):
                country_data = top_countries.iloc[idx]
                for topic_col, topic_name in zip(topic_cols, topics):
                    heatmap_data.append({
                        'Country': country,
                        'Field': topic_name,
                        'Papers': country_data[topic_col]
                    })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_matrix = heatmap_df.pivot(index='Country', columns='Field', values='Papers')
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_matrix.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                colorscale='Viridis',
                colorbar=dict(title="Papers"),
                hoverongaps=False,
                hovertemplate=(
                    "<b>%{y}</b><br>" +
                    "%{x}: %{z:,} papers<br>" +
                    "<extra></extra>"
                )
            ))
            
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=100),
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Geographic visualization error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error rendering geographic section")

    def _plot_geographic_map(self, data: pd.DataFrame, metric_choice: str):
        """Plot choropleth map of research metrics"""
        try:
            # Get all countries
            all_countries = self._get_world_map_data()
            
            # Add missing countries to the dataframe with zero values
            existing_countries = set(data['country_code'])
            for code, name in all_countries.items():
                if code not in existing_countries:
                    new_row = {
                        'country_code': code,
                        'country_name': name,
                        'total_papers': 0,
                        'total_citations': 0,
                        'research_intensity': 0
                    }
                    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            # Configure metric-specific settings
            metric_config = {
                "Publications": {
                    "values": data['total_papers'],
                    "colorscale": "Blues",
                    "format": ",.0f"
                },
                "Citations": {
                    "values": data['total_citations'],
                    "colorscale": "Reds",
                    "format": ",.0f"
                },
                "Impact Factor": {
                    "values": data.apply(
                        lambda x: x['total_citations'] / x['total_papers'] 
                        if x['total_papers'] > 0 else 0, 
                        axis=1
                    ),
                    "colorscale": "Viridis",
                    "format": ".2f"
                },
                "Research Intensity": {
                    "values": data['research_intensity'],
                    "colorscale": "Greens",
                    "format": ".2f"
                }
            }
            
            config = metric_config[metric_choice]
            
            # Create choropleth map
            fig = go.Figure(data=go.Choropleth(
                locations=data['country_code'],
                z=config['values'],
                text=data['country_name'],
                colorscale=config['colorscale'],
                colorbar_title=metric_choice,
                zmin=0,
                zmax=config['values'].max(),
                marker=dict(
                    line=dict(width=0.5, color='white')
                ),
                colorbar=dict(
                    title=dict(
                        text=metric_choice,
                        font=dict(size=14)
                    ),
                    tickformat=config['format']
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    metric_choice + ": %{z:" + config['format'] + "}<br>" +
                    "<extra></extra>"
                )
            ))

            # Update layout with improved styling
            fig.update_layout(
                width=1000,
                height=600,
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    lataxis_range=[-60, 90],
                    lonaxis_range=[-180, 180],
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                    coastlinecolor='rgb(204, 204, 204)',
                    showland=True,
                    showcountries=True,
                    showocean=True,
                    oceancolor='rgb(230, 230, 250)'
                ),
                title=dict(
                    text=f"Global {metric_choice} Distribution",
                    x=0.5,
                    y=0.95,
                    font=dict(size=20)
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Map plotting error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error creating geographic visualization")

    def _plot_top_producers(self, data: pd.DataFrame):
        """Plot top research producing countries"""
        try:
            top_producers = data.nlargest(10, 'papers')
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_producers['papers'],
                    y=top_producers['country_name'],
                    orientation='h',
                    text=top_producers['papers'].apply(lambda x: f"{x:,}"),
                    textposition='auto',
                    marker=dict(color=self.theme["marker_colors"])
                )
            ])

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Number of Publications",
                yaxis_title=None,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Top producers plotting error: {str(e)}")
            st.error("Error creating top producers visualization")

    def _plot_impact_leaders(self, data: pd.DataFrame):
        """Plot research impact leaders"""
        try:
            # Calculate impact factor
            data['impact_factor'] = data['citations'] / data['papers']
            top_impact = data.nlargest(10, 'impact_factor')
            
            # Calculate bubble sizes
            max_papers = top_impact['papers'].max()
            sizes = top_impact['papers'].apply(lambda x: max(20, (x / max_papers * 60)))

            fig = go.Figure(data=[
                go.Scatter(
                    x=top_impact['citations'],
                    y=top_impact['impact_factor'],
                    mode='markers+text',
                    text=top_impact['country_name'],
                    textposition="top center",
                    marker=dict(
                        size=sizes,
                        sizemode='area',
                        sizeref=2.*max(sizes)/(40.**2),
                        color=range(len(top_impact)),
                        colorscale='Viridis',
                        showscale=False
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Citations: %{x:,}<br>" +
                        "Impact Factor: %{y:.2f}<br>" +
                        "<extra></extra>"
                    )
                )
            ])

            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Total Citations",
                yaxis_title="Impact Factor",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Impact leaders plotting error: {str(e)}")
            st.error("Error creating impact leaders visualization")

    def _plot_research_activity(self, data: pd.DataFrame):
        """Plot research activity bar chart"""
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=data['topic'],
                    y=data['count'],
                    text=data['count'].apply(lambda x: f"{x:,}"),
                    textposition='auto',
                    marker_color=self.theme["marker_colors"][:len(data)]
                )
            ])

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=30, b=100),
                xaxis_title="Research Field",
                yaxis_title="Number of Publications",
                xaxis_tickangle=-45,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Research activity plotting error: {str(e)}")
            st.error("Error creating research activity visualization")

    def _plot_growth_patterns(self, data: pd.DataFrame):
        """Plot research growth patterns"""
        try:
            fig = go.Figure(data=[
                go.Scatter(
                    x=data['growth'] * 100,
                    y=data['count'],
                    mode='markers+text',
                    text=data['topic'],
                    textposition="top center",
                    marker=dict(
                        size=data['count'] / data['count'].max() * 50 + 10,
                        color=range(len(data)),
                        colorscale='Viridis',
                        showscale=False
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Growth Rate: %{x:.1f}%<br>" +
                        "Publications: %{y:,}<br>" +
                        "<extra></extra>"
                    )
                )
            ])

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=30, b=50),
                xaxis_title="Growth Rate (%)",
                yaxis_title="Number of Publications",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Growth patterns plotting error: {str(e)}")
            st.error("Error creating growth patterns visualization")

    def _display_research_metrics(self, data: pd.DataFrame):
        """Display detailed research metrics table"""
        try:
            display_df = pd.DataFrame({
                'Research Field': data['topic'],
                'Publications': data['count'].apply(lambda x: f"{x:,}"),
                'Growth Rate': data['growth'].apply(lambda x: f"{x*100:.1f}%"),
                'Last Update': data['days_since_last'].apply(lambda x: f"{x:.1f} days ago")
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
        except Exception as e:
            logger.error(f"Metrics display error: {str(e)}")
            st.error("Error displaying research metrics")

    def _render_impact_section(self, data: pd.DataFrame):
        """Render impact metrics visualization with error handling"""
        try:
            if data.empty:
                st.info("No impact data available")
                return
                
            st.header("📊 Research Impact Analysis")
            
            # Calculate metrics safely
            total_citations = data['citation_count'].sum()
            total_papers = data['paper_count'].sum()
            avg_impact = total_citations / total_papers if total_papers > 0 else 0
            max_citations = data['citation_count'].max()
            
            # Metric Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Citations",
                    f"{total_citations:,}",
                    delta=f"{total_papers:,} papers"
                )
            
            with col2:
                st.metric(
                    "Average Impact",
                    f"{avg_impact:.2f}",
                    delta="citations per paper"
                )
                
            with col3:
                high_impact = len(data[data['citation_count'] / data['paper_count'] > avg_impact])
                st.metric(
                    "High Impact Fields",
                    f"{high_impact}",
                    delta=f"{(high_impact/len(data)*100):.0f}% of total"
                )
                
            with col4:
                st.metric(
                    "Peak Citations",
                    f"{max_citations:,}",
                    delta="highest field"
                )
                
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Field Analysis", "Impact Distribution"])
            
            with tab1:
                self._render_field_impact(data)
                
            with tab2:
                self._render_impact_trends(data)
                
        except Exception as e:
            logger.error(f"Impact visualization error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error rendering impact metrics")
            
    def _render_field_impact(self, data: pd.DataFrame):
        """Render field impact analysis"""
        try:
            st.subheader("Research Field Impact")
            
            # Calculate impact factors safely
            data['impact_factor'] = data.apply(
                lambda x: x['citation_count'] / x['paper_count'] 
                if x['paper_count'] > 0 else 0, 
                axis=1
            )
            
            # Sort by impact factor
            plot_data = data.sort_values('impact_factor', ascending=True)
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=plot_data['impact_factor'],
                    y=plot_data['field'],
                    orientation='h',
                    marker=dict(
                        color=plot_data['impact_factor'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Impact Factor")
                    ),
                    text=plot_data['impact_factor'].apply(lambda x: f"{x:.2f}"),
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Impact Factor (Citations per Paper)",
                yaxis_title=None,
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Field Impact Metrics")
            
            metrics_df = pd.DataFrame({
                'Field': data['field'],
                'Papers': data['paper_count'].apply(lambda x: f"{x:,}"),
                'Citations': data['citation_count'].apply(lambda x: f"{x:,}"),
                'Impact Factor': data['impact_factor'].apply(lambda x: f"{x:.2f}")
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Field impact visualization error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error rendering field impact analysis")
                        
    def _render_impact_trends(self, data: pd.DataFrame):
        """Render impact distribution visualization"""
        try:
            st.subheader("Research Impact Distribution")
            
            # Calculate metrics safely
            data = data.copy()
            data['impact_factor'] = data.apply(
                lambda x: x['citation_count'] / x['paper_count'] 
                if x['paper_count'] > 0 else 0,
                axis=1
            )
            
            # Create scatter plot
            fig = go.Figure(data=[
                go.Scatter(
                    x=data['paper_count'],
                    y=data['citation_count'],
                    mode='markers+text',
                    text=data['field'],
                    textposition="top center",
                    marker=dict(
                        size=data['impact_factor'] * 5 + 10,
                        color=data['impact_factor'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Impact Factor")
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Papers: %{x:,}<br>" +
                        "Citations: %{y:,}<br>" +
                        "<extra></extra>"
                    )
                )
            ])
            
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Number of Papers",
                yaxis_title="Number of Citations",
                xaxis=dict(
                    type='log',
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    type='log',
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Impact trends visualization error: {str(e)}")
            print(f"Error details: {str(e)}")
            st.error("Error rendering impact distribution")
                                    
    def _render_field_analysis(self, data: pd.DataFrame):
        """Render field analysis section"""
        try:
            st.subheader("Research Field Impact Analysis")
            
            # Create hierarchical data
            hierarchical_data = []
            
            for _, row in data.iterrows():
                # Add main field
                hierarchical_data.append({
                    "id": row['field'],
                    "parent": "",
                    "value": row['citation_count']
                })
            
            # Create sunburst chart
            fig = go.Figure(data=[
                go.Sunburst(
                    ids=[item['id'] for item in hierarchical_data],
                    parents=[item['parent'] for item in hierarchical_data],
                    values=[item['value'] for item in hierarchical_data],
                    branchvalues='total',
                    hovertemplate=(
                        "<b>%{id}</b><br>" +
                        "Citations: %{value:,}<br>" +
                        "<extra></extra>"
                    )
                )
            ])

            fig.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Field metrics table
            st.subheader("Field Impact Metrics")
            metrics_df = pd.DataFrame({
                'Field': data['field'],
                'Publications': data['paper_count'].apply(lambda x: f"{x:,}"),
                'Citations': data['citation_count'].apply(lambda x: f"{x:,}"),
                'Impact Factor': data['citation_count'] / data['paper_count']
            })
            
            metrics_df['Impact Factor'] = metrics_df['Impact Factor'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                metrics_df,
                use_container_width=True,
                height=400
            )
            
        except Exception as e:
            logger.error(f"Field analysis error: {str(e)}")
            st.error("Error rendering field analysis")

    def _render_impact_distribution(self, data: pd.DataFrame):
        """Render impact distribution section"""
        try:
            st.subheader("Citation Impact Distribution")
            
            # Calculate impact metrics
            impact_data = pd.DataFrame({
                'field': data['field'],
                'citations_per_paper': data['citation_count'] / data['paper_count'],
                'total_citations': data['citation_count'],
                'total_papers': data['paper_count']
            }).sort_values('citations_per_paper', ascending=True)

            # Create layout with two columns
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create horizontal bar chart for citations per paper
                fig = go.Figure(data=[
                    go.Bar(
                        x=impact_data['citations_per_paper'],
                        y=impact_data['field'],
                        orientation='h',
                        marker=dict(
                            color=impact_data['citations_per_paper'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=impact_data['citations_per_paper'].apply(lambda x: f"{x:.2f}"),
                        textposition='auto',
                        hovertemplate=(
                            "<b>%{y}</b><br>" +
                            "Citations per Paper: %{x:.2f}<br>" +
                            "<extra></extra>"
                        )
                    )
                ])

                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Citations per Paper",
                    yaxis_title=None,
                    coloraxis_colorbar_title="Impact",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Create summary statistics
                st.write("### Impact Summary")
                summary_data = {
                    'Metric': [
                        'Highest Impact',
                        'Average Impact',
                        'Median Impact',
                        'Fields Above Average'
                    ],
                    'Value': [
                        f"{impact_data['citations_per_paper'].max():.2f}",
                        f"{impact_data['citations_per_paper'].mean():.2f}",
                        f"{impact_data['citations_per_paper'].median():.2f}",
                        f"{(impact_data['citations_per_paper'] > impact_data['citations_per_paper'].mean()).sum()}"
                    ]
                }
                st.table(pd.DataFrame(summary_data))

            # Add scatter plot showing relationship between papers and citations
            st.subheader("Publication Volume vs. Citation Impact")
            fig = go.Figure(data=[
                go.Scatter(
                    x=impact_data['total_papers'],
                    y=impact_data['total_citations'],
                    mode='markers+text',
                    text=impact_data['field'],
                    textposition="top center",
                    marker=dict(
                        size=impact_data['citations_per_paper'] * 5,
                        color=impact_data['citations_per_paper'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Citations<br>per Paper")
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Papers: %{x:,}<br>" +
                        "Citations: %{y:,}<br>" +
                        "<extra></extra>"
                    )
                )
            ])

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis_title="Number of Publications",
                yaxis_title="Total Citations",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add detailed metrics table
            st.subheader("Detailed Impact Metrics")
            detailed_metrics = pd.DataFrame({
                'Research Field': impact_data['field'],
                'Publications': impact_data['total_papers'].apply(lambda x: f"{x:,}"),
                'Citations': impact_data['total_citations'].apply(lambda x: f"{x:,}"),
                'Impact Factor': impact_data['citations_per_paper'].apply(lambda x: f"{x:.2f}"),
                'Relative Impact': impact_data['citations_per_paper'] / impact_data['citations_per_paper'].mean()
            })

            # Format relative impact
            detailed_metrics['Relative Impact'] = detailed_metrics['Relative Impact'].apply(
                lambda x: f"{x:.2f}x average"
            )

            # Display sortable table
            st.dataframe(
                detailed_metrics,
                use_container_width=True,
                height=400
            )

        except Exception as e:
            logger.error(f"Impact distribution visualization error: {str(e)}")
            st.error("Error rendering impact distribution")

    def _format_number(self, num: float) -> str:
        """Format large numbers with K/M/B suffixes"""
        try:
            if num >= 1e9:
                return f"{num/1e9:.1f}B"
            elif num >= 1e6:
                return f"{num/1e6:.1f}M"
            elif num >= 1e3:
                return f"{num/1e3:.1f}K"
            return f"{num:.0f}"
        except Exception as e:
            logger.error(f"Number formatting error: {str(e)}")
            return str(num)

    def _create_tooltip(self, text: str, help_text: str) -> None:
        """Create a tooltip with help text"""
        st.markdown(f"""
            <div style="display: inline-block;" title="{help_text}">
                {text} ℹ️
            </div>
        """, unsafe_allow_html=True)

    def _handle_error(self, error: Exception, message: str) -> None:
        """Handle errors consistently across visualizations"""
        logger.error(f"{message}: {str(error)}")
        st.error(f"Error: {message}")
        if st.checkbox("Show error details"):
            st.exception(error)