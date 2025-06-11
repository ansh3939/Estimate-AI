import os
import streamlit as st
from openai import OpenAI
from typing import List, Dict
import json

class RealEstateChatbot:
    def __init__(self):
        """Initialize the Real Estate Chatbot with OpenAI"""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.fallback_knowledge = self._build_knowledge_base()
        
        # System prompt for real estate expertise
        self.system_prompt = """You are an expert real estate advisor and consultant with deep knowledge of the Indian property market. You specialize in:

1. Property investment strategies and analysis
2. Market trends and price predictions
3. Legal aspects of property transactions
4. Home buying and selling processes
5. Property valuation and appraisal
6. Rental market insights
7. Construction and development advice
8. Property tax and financial planning
9. Location analysis and neighborhood insights
10. Commercial and residential real estate

Provide accurate, helpful, and professional advice. Always consider the Indian real estate context, regulations, and market conditions. When discussing specific investments or legal matters, remind users to consult with qualified professionals for personalized advice.

Keep responses concise but informative, and ask clarifying questions when needed to provide better assistance."""

    def get_response(self, user_message: str, chat_history: List[Dict] = None) -> str:
        """Get response from OpenAI GPT-4o for real estate questions"""
        try:
            # Prepare messages with system prompt and chat history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add chat history if available
            if chat_history is not None:
                messages.extend(chat_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_content = response.choices[0].message.content
            return response_content if response_content is not None else "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                # Use fallback knowledge base when quota exceeded
                fallback_response = self.get_fallback_response(user_message)
                return f"📋 **Real Estate Knowledge Base Response:**\n\n{fallback_response}\n\n💡 *Note: Advanced AI assistant temporarily unavailable due to quota limits. The above information is from our curated real estate knowledge base.*"
            else:
                return f"I'm experiencing technical difficulties connecting to the AI service. Please try again later. Error details: {error_msg}"

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

**🏠 Buying Process** - Steps to purchase property in India
**💰 Investment Analysis** - ROI calculation and market factors  
**📋 Legal Aspects** - Documentation and compliance requirements
**🏦 Financing** - Home loans and EMI calculations
**📈 Market Trends** - Current real estate market insights

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
        """Render the chatbot interface in Streamlit"""
        st.markdown("### 🏠 Real Estate AI Assistant")
        st.markdown("Ask me anything about real estate, property investment, market trends, or buying/selling properties!")
        
        # Initialize chat history
        self.initialize_chat_history()
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me about real estate...")
        
        if user_input:
            # Add user message to chat
            self.add_message("user", user_input)
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_response = self.get_response(user_input, st.session_state.chat_history[:-1])
                    st.write(ai_response)
                    
                    # Add AI response to chat history
                    self.add_message("assistant", ai_response)
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            self.clear_chat()
            st.rerun()

    def get_suggested_questions(self) -> List[str]:
        """Get suggested real estate questions for users"""
        return [
            "What factors should I consider when buying my first home?",
            "How do I determine if a property is a good investment?",
            "What are the current real estate market trends in India?",
            "How much should I budget for property taxes and maintenance?",
            "What documents do I need for a property purchase?",
            "Should I buy or rent in the current market?",
            "How do I negotiate the best price for a property?",
            "What are the legal aspects I should know about property buying?",
            "How do I calculate property ROI for investment purposes?",
            "What are the best locations for real estate investment?"
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