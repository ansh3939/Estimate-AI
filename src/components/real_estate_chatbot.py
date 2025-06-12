import os
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Optional
import json
import datetime
import re

class RealEstateChatbot:
    def __init__(self):
        """Initialize the AI-Powered Real Estate Assistant with comprehensive expertise"""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.fallback_knowledge = self._build_knowledge_base()
        self.conversation_memory = []
        self.user_preferences = {}
        
        # Advanced AI system prompt with comprehensive real estate expertise
        self.system_prompt = """You are ARIA (Advanced Real Estate Intelligence Assistant), an AI-powered real estate expert with comprehensive knowledge of the Indian property market. You possess advanced capabilities across all aspects of real estate:

**CORE EXPERTISE:**
• Property Investment Strategy & Portfolio Management
• Market Analysis & Price Trend Forecasting
• Legal Compliance & Documentation (RERA, GST, Property Laws)
• Financial Planning & Tax Optimization
• Location Intelligence & Neighborhood Analysis
• Construction Quality Assessment & Project Evaluation
• Rental Market Dynamics & Yield Calculations
• Commercial Real Estate & REITs
• Property Valuation & Due Diligence
• Home Loans & Financing Options

**ADVANCED CAPABILITIES:**
• Provide data-driven market insights with specific metrics
• Analyze investment opportunities with ROI calculations
• Explain complex legal procedures in simple terms
• Offer personalized advice based on user's financial situation
• Predict market trends using economic indicators
• Compare properties across multiple parameters
• Guide through entire buying/selling process step-by-step
• Suggest optimal timing for property transactions

**COMMUNICATION STYLE:**
• Professional yet approachable tone
• Use specific examples and case studies
• Provide actionable recommendations
• Ask insightful follow-up questions
• Cite relevant regulations and market data
• Structure responses with clear headings and bullet points

**IMPORTANT:** Always consider current Indian real estate regulations, market conditions, and regional variations. When discussing investments or legal matters, recommend consulting certified professionals for final decisions.

Remember to personalize responses based on user's location, budget, and investment goals. Ask clarifying questions to provide the most relevant advice."""

    def extract_user_context(self, message: str) -> Dict:
        """Extract context and preferences from user message"""
        context = {
            'budget_mentioned': False,
            'location_mentioned': False,
            'property_type_mentioned': False,
            'timeline_mentioned': False,
            'investment_goal': None
        }
        
        # Extract budget information
        budget_patterns = [r'(\d+(?:\.\d+)?)\s*(?:lakh|crore|lac)', r'₹\s*(\d+(?:,\d+)*)', r'budget.*?(\d+)']
        for pattern in budget_patterns:
            if re.search(pattern, message.lower()):
                context['budget_mentioned'] = True
                break
        
        # Extract location information
        indian_cities = ['mumbai', 'delhi', 'bangalore', 'gurugram', 'noida', 'pune', 'hyderabad', 'chennai', 'kolkata', 'ahmedabad']
        for city in indian_cities:
            if city in message.lower():
                context['location_mentioned'] = True
                context['mentioned_location'] = city.title()
                break
        
        # Extract property type
        property_types = ['apartment', 'villa', 'house', 'flat', 'plot', 'commercial', 'office']
        for prop_type in property_types:
            if prop_type in message.lower():
                context['property_type_mentioned'] = True
                context['mentioned_property_type'] = prop_type
                break
        
        # Extract investment goals
        if any(word in message.lower() for word in ['invest', 'investment', 'return', 'roi']):
            context['investment_goal'] = 'investment'
        elif any(word in message.lower() for word in ['home', 'house', 'family', 'live']):
            context['investment_goal'] = 'residence'
        
        return context

    def build_contextual_prompt(self, user_message: str, context: Dict) -> str:
        """Build enhanced prompt with context awareness"""
        contextual_info = []
        
        if context.get('budget_mentioned'):
            contextual_info.append("The user has mentioned a budget. Provide budget-appropriate recommendations.")
        
        if context.get('location_mentioned'):
            location = context.get('mentioned_location', 'the mentioned location')
            contextual_info.append(f"Focus on {location} market conditions, prices, and opportunities.")
        
        if context.get('investment_goal') == 'investment':
            contextual_info.append("Emphasize ROI, rental yields, capital appreciation, and investment strategies.")
        elif context.get('investment_goal') == 'residence':
            contextual_info.append("Focus on lifestyle factors, amenities, schools, and long-term comfort.")
        
        if contextual_info:
            enhanced_prompt = f"CONTEXT: {' '.join(contextual_info)}\n\nUSER QUERY: {user_message}"
            return enhanced_prompt
        
        return user_message

    def get_response(self, user_message: str, chat_history: List[Dict] = None) -> str:
        """Get advanced response from OpenAI GPT-4o with context awareness"""
        try:
            # Extract context from user message
            context = self.extract_user_context(user_message)
            
            # Build contextual prompt
            enhanced_message = self.build_contextual_prompt(user_message, context)
            
            # Add current market data and insights to system prompt
            current_date = datetime.datetime.now().strftime("%B %Y")
            enhanced_system_prompt = f"""{self.system_prompt}

**CURRENT MARKET CONTEXT ({current_date}):**
• Interest rates are affecting home loan demand
• RERA compliance is mandatory for all new projects
• Digital property transactions are increasing
• Tier-2 cities showing strong growth potential
• Work-from-home trend impacting location preferences

**RESPONSE GUIDELINES:**
• Provide specific, actionable advice
• Include relevant market data and trends
• Structure response with clear sections
• Ask follow-up questions to better assist the user
• Mention specific regulations or compliance requirements when relevant"""

            # Prepare messages with enhanced system prompt
            messages = [{"role": "system", "content": enhanced_system_prompt}]
            
            # Add conversation memory for continuity
            if hasattr(self, 'conversation_memory') and self.conversation_memory:
                messages.extend(self.conversation_memory[-6:])  # Keep last 3 exchanges
            
            # Add chat history if available
            if chat_history:
                messages.extend(chat_history[-8:])  # Limit to recent history
            
            # Add current enhanced message
            messages.append({"role": "user", "content": enhanced_message})
            
            # Get response from OpenAI with advanced parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1200,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            assistant_response = response.choices[0].message.content
            
            # Store conversation in memory for context continuity
            self.conversation_memory.append({"role": "user", "content": user_message})
            self.conversation_memory.append({"role": "assistant", "content": assistant_response})
            
            # Keep memory manageable
            if len(self.conversation_memory) > 12:
                self.conversation_memory = self.conversation_memory[-12:]
            
            return assistant_response
            
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                # Use fallback knowledge base when quota exceeded
                fallback_response = self.get_fallback_response(user_message)
                return f"**Real Estate Knowledge Base Response:**\n\n{fallback_response}\n\n*Note: Advanced AI assistant temporarily unavailable due to quota limits. The above information is from our curated real estate knowledge base.*"
            else:
                return f"I'm experiencing technical difficulties connecting to the AI service. Please try again later. Error details: {error_msg}"

    def analyze_sentiment_and_urgency(self, message: str) -> Dict:
        """Analyze user sentiment and urgency level"""
        urgency_keywords = {
            'high': ['urgent', 'immediately', 'asap', 'quickly', 'soon', 'emergency'],
            'medium': ['need', 'looking for', 'planning', 'considering'],
            'low': ['someday', 'future', 'maybe', 'thinking about']
        }
        
        sentiment_keywords = {
            'positive': ['excited', 'great', 'perfect', 'excellent', 'amazing'],
            'neutral': ['okay', 'fine', 'good', 'alright'],
            'negative': ['worried', 'concerned', 'problem', 'issue', 'confused']
        }
        
        message_lower = message.lower()
        
        urgency = 'low'
        for level, keywords in urgency_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                urgency = level
                break
        
        sentiment = 'neutral'
        for mood, keywords in sentiment_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                sentiment = mood
                break
        
        return {'urgency': urgency, 'sentiment': sentiment}

    def get_smart_suggestions(self, context: Dict) -> List[str]:
        """Generate smart follow-up suggestions based on context"""
        suggestions = []
        
        if context.get('budget_mentioned'):
            suggestions.append("Would you like me to suggest properties within your budget range?")
            suggestions.append("Should I explain financing options and EMI calculations?")
        
        if context.get('location_mentioned'):
            location = context.get('mentioned_location', 'that area')
            suggestions.append(f"Would you like to know about upcoming projects in {location}?")
            suggestions.append(f"Should I provide market trends for {location}?")
        
        if context.get('investment_goal') == 'investment':
            suggestions.extend([
                "Would you like a detailed ROI analysis?",
                "Should I explain rental yield calculations?",
                "Would you like to know about tax benefits?"
            ])
        elif context.get('investment_goal') == 'residence':
            suggestions.extend([
                "Would you like information about schools and amenities nearby?",
                "Should I help with home loan pre-approval guidance?",
                "Would you like to know about possession timelines?"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions

    def _build_knowledge_base(self) -> Dict[str, str]:
        """Build a comprehensive real estate knowledge base for fallback responses"""
        return {
            "buying": """**Home Buying Process in India:**
1. Determine budget and get pre-approved for loan
2. Research locations and property types
3. Visit properties and shortlist options
4. Verify legal documents (title deed, approvals, NOC)
5. Negotiate price and terms
6. Sign agreement and pay token amount
7. Complete due diligence and property verification
8. Finalize loan and complete registration
9. Take possession and register utilities

**Key Documents:** Sale deed, title certificate, occupancy certificate, RERA registration, property tax receipts, society NOC""",

            "investment": """**Property Investment Analysis:**
- **Location factors:** Connectivity, infrastructure development, employment hubs
- **Financial metrics:** Rental yield (6-8% good), capital appreciation potential
- **Market timing:** Buy during market corrections, avoid peak pricing
- **Property type:** Residential vs commercial based on goals
- **Legal compliance:** RERA registered projects, clear titles
- **Exit strategy:** Plan for liquidity needs and market cycles

**ROI Calculation:** (Annual rental income + appreciation) / Total investment cost""",

            "legal": """**Legal Aspects of Property Transactions:**
- **Title verification:** Chain of ownership, encumbrance certificate
- **Approvals:** Building plan approval, occupancy certificate, RERA registration
- **Compliance:** Property tax clearance, society NOC, utility connections
- **Registration:** Stamp duty, registration fees vary by state
- **Documentation:** Sale deed, agreement to sell, power of attorney verification

**Red flags:** Disputed properties, incomplete approvals, unclear titles, non-RERA projects""",

            "financing": """**Home Loan and Financing:**
- **Eligibility:** 60-80% of property value, income-based EMI capacity
- **Interest rates:** Currently 8.5-11% for home loans
- **Documentation:** Income proof, property papers, bank statements
- **Processing:** 15-30 days for approval, additional time for disbursement
- **Tax benefits:** Section 80C (principal), Section 24 (interest)

**EMI calculation:** Use property price predictor's built-in EMI calculator for accurate estimates""",

            "market": """**Current Indian Real Estate Market Trends:**
- **Metropolitan growth:** Mumbai, Delhi, Bangalore leading appreciation
- **Emerging markets:** Pune, Hyderabad, Chennai showing strong potential
- **Segment performance:** Mid-segment (₹50L-₹1Cr) most active
- **Supply trends:** Focus on ready-to-move properties increasing
- **Policy impact:** RERA, GST implementation stabilizing market
- **Future outlook:** Steady growth expected with infrastructure development"""
        }

    def get_fallback_response(self, user_message: str) -> str:
        """Provide fallback response using knowledge base"""
        message_lower = user_message.lower()
        
        # Match keywords to knowledge base topics
        if any(word in message_lower for word in ["buy", "buying", "purchase", "first home"]):
            return self.fallback_knowledge["buying"]
        elif any(word in message_lower for word in ["invest", "investment", "roi", "return"]):
            return self.fallback_knowledge["investment"]
        elif any(word in message_lower for word in ["legal", "document", "registration", "title"]):
            return self.fallback_knowledge["legal"]
        elif any(word in message_lower for word in ["loan", "finance", "emi", "bank", "mortgage"]):
            return self.fallback_knowledge["financing"]
        elif any(word in message_lower for word in ["market", "trend", "price", "growth"]):
            return self.fallback_knowledge["market"]
        else:
            return """I can help with these real estate topics:

**Buying Process** - Steps to purchase property in India
**Investment Analysis** - ROI calculation and market factors  
**Legal Aspects** - Documentation and compliance requirements
**Financing** - Home loans and EMI calculations
**Market Trends** - Current real estate market insights

Please ask about any of these topics for detailed guidance. For personalized advice, consult local real estate professionals."""

    def initialize_chat_history(self):
        """Initialize chat history in Streamlit session state"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_messages = []

    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_history.append({"role": role, "content": content})
        st.session_state.chat_messages.append({"role": role, "content": content})

    def clear_chat(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.session_state.chat_messages = []

    def render_chatbot_interface(self):
        """Render the advanced chatbot interface in Streamlit"""
        st.markdown("### ARIA - Advanced Real Estate Intelligence Assistant")
        st.markdown("*Your expert advisor for Indian real estate markets, investment strategies, and property transactions*")
        
        # Initialize chat history
        self.initialize_chat_history()
        
        # Advanced chat container with context awareness
        chat_container = st.container()
        
        # Display enhanced chat history with context indicators
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                        
                        # Show context analysis for recent messages
                        if i >= len(st.session_state.chat_messages) - 4:
                            context = self.extract_user_context(message["content"])
                            sentiment = self.analyze_sentiment_and_urgency(message["content"])
                            
                            context_info = []
                            if context.get('budget_mentioned'):
                                context_info.append("Budget mentioned")
                            if context.get('location_mentioned'):
                                context_info.append(f"Location: {context.get('mentioned_location', 'N/A')}")
                            if context.get('investment_goal'):
                                context_info.append(f"Goal: {context['investment_goal']}")
                            if sentiment['urgency'] != 'low':
                                context_info.append(f"Urgency: {sentiment['urgency']}")
                            
                            if context_info:
                                st.caption(" • ".join(context_info))
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Add smart suggestions for recent assistant messages
                        if i == len(st.session_state.chat_messages) - 1 and len(st.session_state.chat_messages) > 1:
                            last_user_msg = st.session_state.chat_messages[-2]["content"]
                            context = self.extract_user_context(last_user_msg)
                            suggestions = self.get_smart_suggestions(context)
                            
                            if suggestions:
                                st.markdown("**Quick Actions:**")
                                for j, suggestion in enumerate(suggestions):
                                    if st.button(suggestion, key=f"suggestion_{i}_{j}"):
                                        # Auto-fill the suggestion as user input
                                        self.add_message("user", suggestion)
                                        st.rerun()
        
        # Enhanced chat input with context hints
        placeholder_text = "Ask about properties, investments, market trends, legal advice..."
        
        # Dynamic placeholder based on conversation context
        if st.session_state.chat_messages:
            last_msg = st.session_state.chat_messages[-1]["content"]
            if "budget" in last_msg.lower():
                placeholder_text = "Tell me your budget range or ask about financing options..."
            elif "location" in last_msg.lower():
                placeholder_text = "Which areas are you considering or want to know about..."
            elif "investment" in last_msg.lower():
                placeholder_text = "Ask about ROI, rental yields, or investment strategies..."
        
        user_input = st.chat_input(placeholder_text)
        
        if user_input:
            # Add user message to chat
            self.add_message("user", user_input)
            
            # Display user message with enhanced formatting
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get advanced AI response with enhanced processing
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your query and market data..."):
                    ai_response = self.get_response(user_input, st.session_state.chat_history[:-1])
                    st.write(ai_response)
                    
                    # Add AI response to chat history
                    self.add_message("assistant", ai_response)
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            self.clear_chat()
            st.rerun()

    def get_suggested_questions(self) -> List[str]:
        """Get advanced suggested questions based on current market context"""
        return [
            "What's the investment potential of emerging micro-markets in 2024?",
            "How do interest rate changes affect property valuations?",
            "Which tier-2 cities offer the best ROI for real estate investment?",
            "How to evaluate RERA compliance and developer credibility?",
            "What are the tax implications of property investment vs REITs?",
            "How does the new Digital Property ID system work?",
            "What's the impact of infrastructure projects on property values?",
            "How to assess rental yield potential in different localities?",
            "What are the key factors for commercial real estate investment?",
            "How to leverage home loan tax benefits effectively?",
            "What's the outlook for affordable housing schemes?",
            "How to evaluate pre-launch vs ready-to-move properties?"
        ]

    def render_suggested_questions(self):
        """Render suggested questions as clickable buttons"""
        st.markdown("#### Popular Questions:")
        
        suggestions = self.get_suggested_questions()
        
        # Display questions in columns
        cols = st.columns(2)
        for i, question in enumerate(suggestions[:6]):  # Show first 6 questions
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    # Add question to chat
                    self.add_message("user", question)
                    
                    # Get AI response
                    ai_response = self.get_response(question, st.session_state.chat_history[:-1])
                    self.add_message("assistant", ai_response)
                    
                    # Rerun to update chat
                    st.rerun()