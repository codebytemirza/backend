#!/usr/bin/env python3
"""
Production-Ready LangGraph Multi-Agent RAG Workflow
with Structured Outputs using Pydantic - CONVERSATION CONTEXT FIXED
"""

import os
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Annotated
from pathlib import Path
from collections import defaultdict

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import TavilySearchResults

# Pinecone
from pinecone import Pinecone

# Pydantic for structured outputs
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv


# ============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUTS
# ============================================================================

class QueryDecomposition(BaseModel):
    """Schema for query decomposition."""
    queries: List[str] = Field(
        description="List of decomposed sub-queries. For simple queries, return single item list with original query."
    )


class ContextRanking(BaseModel):
    """Schema for context ranking."""
    ranked_indices: List[int] = Field(
        description="List of context indices (1-based) in order of relevance, most relevant first."
    )


class ContextFiltering(BaseModel):
    """Schema for context filtering."""
    keep_indices: List[int] = Field(
        description="List of context indices (1-based) to keep. Select 2-5 most relevant contexts."
    )


class SufficiencyDecision(BaseModel):
    """Schema for sufficiency decision."""
    sufficient: bool = Field(
        description="Whether the available context is sufficient to answer the query"
    )
    reasoning: str = Field(
        description="Explanation of the sufficiency decision"
    )


class UserFact(BaseModel):
    """Schema for a single user fact."""
    key: str = Field(description="Fact category/key (e.g., 'name', 'project', 'interest')")
    value: str = Field(description="Fact value")


class ExtractedFacts(BaseModel):
    """Schema for extracted facts."""
    facts: List[UserFact] = Field(
        default_factory=list,
        description="List of extracted facts about the user. Empty list if no facts found."
    )


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class GraphState(TypedDict):
    """State of the RAG workflow graph."""
    original_query: str
    thread_id: str
    user_facts: List[Dict[str, str]]
    decomposed_queries: List[str]
    retrieved_contexts: List[Dict[str, Any]]
    ranked_contexts: List[Dict[str, Any]]
    filtered_contexts: List[Dict[str, Any]]
    web_search_results: Optional[List[Dict[str, Any]]]
    final_context: str
    is_context_sufficient: bool
    custom_prompt: str
    final_answer: str
    messages: Annotated[list, add_messages]


# ============================================================================
# MEMORY MANAGER - FIXED
# ============================================================================

class FactMemoryManager:
    """Manages user facts separately from conversation history."""
    
    def __init__(self, max_facts_per_thread: int = 50):
        self.user_facts = defaultdict(dict)
        self.max_facts = max_facts_per_thread
    
    def extract_and_store_facts(
        self, 
        thread_id: str, 
        conversation_turn: Dict[str, str], 
        llm_with_structure
    ) -> List[Dict[str, str]]:
        """
        Extract facts using structured output.
        FIXED: Properly handles the returned Pydantic model.
        """
        user_message = conversation_turn.get("user", "")
        assistant_message = conversation_turn.get("assistant", "")
        
        extraction_prompt = f"""
Extract ONLY concrete, factual information about the user from this conversation.

User: {user_message}
Assistant: {assistant_message}

Extract facts like:
- Name, age, location, profession
- Preferences, interests, goals
- Important context (projects, problems they're solving)
- Specific requirements or constraints

DO NOT extract:
- General conversation topics
- Questions asked
- AI capabilities or limitations
- Transient information

Return structured facts or empty list if none found.
"""
        
        try:
            # with_structured_output returns the Pydantic model directly, not a message
            result: ExtractedFacts = llm_with_structure.invoke([
                SystemMessage(content=extraction_prompt)
            ])
            
            # Convert Pydantic models to dicts
            facts_list = [{"key": f.key, "value": f.value} for f in result.facts]
            
            # Store facts
            for fact in facts_list:
                self.user_facts[thread_id][fact["key"]] = fact["value"]
            
            # Limit facts per thread
            if len(self.user_facts[thread_id]) > self.max_facts:
                keys = list(self.user_facts[thread_id].keys())
                for old_key in keys[:len(keys) - self.max_facts]:
                    del self.user_facts[thread_id][old_key]
            
            return facts_list
            
        except Exception as e:
            print(f"⚠️ Fact extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_facts(self, thread_id: str) -> Dict[str, str]:
        """Get all facts for a thread."""
        return dict(self.user_facts[thread_id])
    
    def get_facts_as_text(self, thread_id: str) -> str:
        """Get facts formatted as text."""
        facts = self.user_facts[thread_id]
        if not facts:
            return ""
        fact_lines = [f"- {key}: {value}" for key, value in facts.items()]
        return "User Information:\n" + "\n".join(fact_lines)
    
    def clear_facts(self, thread_id: str):
        """Clear facts for a thread."""
        if thread_id in self.user_facts:
            del self.user_facts[thread_id]


# ============================================================================
# CONVERSATION CONTEXT BUILDER - NEW
# ============================================================================

class ConversationContextBuilder:
    """Builds rich conversation context for the LLM."""
    
    @staticmethod
    def build_conversation_history(messages: List, max_exchanges: int = 5) -> str:
        """
        Build formatted conversation history from messages.
        
        Args:
            messages: List of HumanMessage and AIMessage objects
            max_exchanges: Maximum number of exchange pairs to include
            
        Returns:
            Formatted conversation history string
        """
        if not messages or len(messages) <= 1:
            return ""
        
        # Get recent messages (exclude the very last one as it's the current query)
        recent_messages = messages[-(max_exchanges * 2):-1] if len(messages) > 1 else []
        
        if not recent_messages:
            return ""
        
        history_lines = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Truncate long responses for context window management
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                history_lines.append(f"Assistant: {content}")
        
        return "\n".join(history_lines)
    
    @staticmethod
    def detect_conversational_response(query: str, has_history: bool) -> bool:
        """
        Detect if query is a conversational response (yes/no/elaborate etc.)
        rather than a standalone question.
        """
        if not has_history:
            return False
        
        query_lower = query.lower().strip()
        
        # Single word or very short responses
        conversational_triggers = [
            "yes", "no", "ok", "okay", "sure", "nope", "yep", "yeah", "nah",
            "what?", "why?", "how?", "when?", "where?", "really?", "seriously?",
            "elaborate", "continue", "more", "explain", "clarify", "huh?",
            "i disagree", "i agree", "not really", "exactly", "precisely",
            "wrong", "correct", "right", "false", "true"
        ]
        
        # Check if query is short and matches conversational patterns
        word_count = len(query.split())
        if word_count <= 3:
            for trigger in conversational_triggers:
                if query_lower == trigger or query_lower.startswith(trigger):
                    return True
        
        # Check for disagreement phrases
        disagreement_phrases = [
            "i don't think", "i disagree", "that's not", "that's wrong",
            "not true", "i don't agree", "nope", "no way"
        ]
        
        for phrase in disagreement_phrases:
            if phrase in query_lower:
                return True
        
        return False
    
    @staticmethod
    def build_context_awareness_prompt(
        is_conversational: bool,
        conversation_history: str
    ) -> str:
        """Build context awareness section for system prompt."""
        if not is_conversational or not conversation_history:
            return ""
        
        return f"""
==================
### 🗨️ CONVERSATION CONTEXT (CRITICAL)

**Previous Discussion:**
{conversation_history}

**IMPORTANT - CONVERSATIONAL RESPONSE DETECTED:**
The user's current message appears to be a RESPONSE to your previous answer, not a new standalone question.

**How to handle:**
1. If user says "no", "I disagree", "wrong":
   → Acknowledge their disagreement respectfully
   → Ask what aspect they disagree with
   → Invite them to share their perspective
   → DO NOT define the word or treat it as a dictionary query

2. If user says "yes", "agree", "correct":
   → Acknowledge their agreement
   → Offer to expand or explore related topics
   → DO NOT explain what "yes" means

3. If user says "elaborate", "explain more", "continue":
   → Continue from where you left off
   → Provide deeper detail on the previous topic
   → DO NOT define the word "elaborate"

4. If user says "what?", "why?", "how?":
   → These are follow-up questions about your previous response
   → Answer in context of the previous discussion
   → DO NOT treat as standalone questions

**Remember:** You're in an ongoing conversation. Maintain context and flow naturally.
==================
"""


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class MultiAgentRAGWorkflow:
    """Production-ready LangGraph RAG workflow with structured outputs."""
    
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        
        # Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        backend_dir = Path(__file__).parent.parent
        self.db_path = os.getenv("PROMPT_DB_PATH", str(backend_dir / "prompt.db"))
        
        # Initialize memory
        self.fact_memory = FactMemoryManager()
        self.context_builder = ConversationContextBuilder()
        
        # Initialize components
        self._init_llms()
        self._init_vector_store()
        self._init_tools()
        self._init_graph()
        
    def _init_llms(self):
        """Initialize LLM instances with structured output support."""
        # Fast LLM for processing
        self.fast_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        
        # Structured output LLMs
        self.decompose_llm = self.fast_llm.with_structured_output(
            QueryDecomposition,
            method="json_schema",
            strict=True
        )
        
        self.rank_llm = self.fast_llm.with_structured_output(
            ContextRanking,
            method="json_schema",
            strict=True
        )
        
        self.filter_llm = self.fast_llm.with_structured_output(
            ContextFiltering,
            method="json_schema",
            strict=True
        )
        
        self.sufficiency_llm = self.fast_llm.with_structured_output(
            SufficiencyDecision,
            method="json_schema",
            strict=True
        )
        
        self.fact_extraction_llm = self.fast_llm.with_structured_output(
            ExtractedFacts,
            method="json_schema",
            strict=True
        )
        
        # Answer LLM (no structure needed)
        self.answer_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=self.openai_api_key
        )
        
    def _init_vector_store(self):
        """Initialize Pinecone vector store."""
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(self.pinecone_index_name)
            
            embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=self.openai_api_key
            )
            
            self.vector_store = PineconeVectorStore(index=index, embedding=embeddings)
            print("✅ Vector store initialized")
        except Exception as e:
            print(f"⚠️ Vector store init failed: {e}")
            self.vector_store = None
        
    def _init_tools(self):
        """Initialize tools."""
        try:
            self.search_tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                api_key=self.tavily_api_key
            )
            print("✅ Search tools initialized")
        except Exception as e:
            print(f"⚠️ Search tools init failed: {e}")
            self.search_tool = None
        
    def _init_graph(self):
        """Initialize the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("load_custom_prompt", self.load_custom_prompt)
        workflow.add_node("decompose_query", self.decompose_query)
        workflow.add_node("retrieve_contexts", self.retrieve_contexts)
        workflow.add_node("rank_contexts", self.rank_contexts)
        workflow.add_node("filter_contexts", self.filter_contexts)
        workflow.add_node("decide_sufficiency", self.decide_sufficiency)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("extract_facts", self.extract_facts)
        
        # Define flow
        workflow.add_edge(START, "load_context")
        workflow.add_edge("load_context", "load_custom_prompt")
        workflow.add_edge("load_custom_prompt", "decompose_query")
        workflow.add_edge("decompose_query", "retrieve_contexts")
        workflow.add_edge("retrieve_contexts", "rank_contexts")
        workflow.add_edge("rank_contexts", "filter_contexts")
        workflow.add_edge("filter_contexts", "decide_sufficiency")
        
        workflow.add_conditional_edges(
            "decide_sufficiency",
            self.route_based_on_sufficiency,
            {
                "sufficient": "generate_answer",
                "insufficient": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", "extract_facts")
        workflow.add_edge("extract_facts", END)
        
        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)
        print("✅ Graph compiled with checkpointing")
        
    def get_custom_prompt(self) -> str:
        """Get custom prompt from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM prompt LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else ""
        except Exception as e:
            print(f"⚠️ Error getting custom prompt: {e}")
            return ""
    
    # ========================================================================
    # WORKFLOW NODES
    # ========================================================================
    
    def load_context(self, state: GraphState) -> GraphState:
        """Load user facts."""
        print(f"🧠 Loading context for thread: {state.get('thread_id', 'default')}")
        
        thread_id = state.get('thread_id', 'default')
        facts = self.fact_memory.get_facts(thread_id)
        state["user_facts"] = [{"key": k, "value": v} for k, v in facts.items()]
        
        if facts:
            print(f"✅ Loaded {len(facts)} user facts")
        else:
            print("ℹ️ No user facts found")
        
        return state
    
    def load_custom_prompt(self, state: GraphState) -> GraphState:
        """Load custom system prompt."""
        print("🔧 Loading custom prompt")
        state["custom_prompt"] = self.get_custom_prompt()
        return state
    
    def decompose_query(self, state: GraphState) -> GraphState:
        """Decompose query using structured output."""
        print("🔍 Decomposing query")
        
        # Check if this is a conversational response - skip decomposition
        has_history = len(state.get("messages", [])) > 1
        is_conversational = self.context_builder.detect_conversational_response(
            state["original_query"], 
            has_history
        )
        
        if is_conversational:
            print("💬 Conversational response detected - skipping decomposition")
            state["decomposed_queries"] = [state["original_query"]]
            return state
        
        facts_context = ""
        if state["user_facts"]:
            facts_list = [f"{f['key']}: {f['value']}" for f in state["user_facts"]]
            facts_context = f"\nKnown user facts: {', '.join(facts_list)}\n"
        
        prompt = f"""
Analyze this query and decompose if needed.
{facts_context}
Query: {state['original_query']}

For simple queries: return single item list with original query.
For complex queries: decompose into 2-4 focused sub-queries.
"""
        
        try:
            result: QueryDecomposition = self.decompose_llm.invoke([
                SystemMessage(content=prompt)
            ])
            state["decomposed_queries"] = result.queries
            print(f"✅ Decomposed into {len(state['decomposed_queries'])} queries")
        except Exception as e:
            print(f"⚠️ Decomposition failed: {e}")
            state["decomposed_queries"] = [state["original_query"]]
        
        return state
    
    def retrieve_contexts(self, state: GraphState) -> GraphState:
        """Retrieve from vector database."""
        print("📚 Retrieving contexts")
        
        # Skip retrieval for simple conversational responses
        has_history = len(state.get("messages", [])) > 1
        is_conversational = self.context_builder.detect_conversational_response(
            state["original_query"], 
            has_history
        )
        
        if is_conversational:
            print("💬 Conversational response - minimal retrieval needed")
            state["retrieved_contexts"] = []
            return state
        
        if not self.vector_store:
            state["retrieved_contexts"] = []
            return state
        
        all_contexts = []
        for query in state["decomposed_queries"]:
            try:
                results = self.vector_store.similarity_search_with_score(query, k=5)
                for doc, score in results:
                    all_contexts.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score),
                        "source_query": query
                    })
            except Exception as e:
                print(f"⚠️ Retrieval error for '{query}': {e}")
        
        # Remove duplicates
        unique_contexts = []
        seen = set()
        for ctx in all_contexts:
            content_hash = hash(ctx["content"])
            if content_hash not in seen:
                unique_contexts.append(ctx)
                seen.add(content_hash)
        
        state["retrieved_contexts"] = unique_contexts
        print(f"✅ Retrieved {len(unique_contexts)} contexts")
        return state
    
    def rank_contexts(self, state: GraphState) -> GraphState:
        """Rank contexts using structured output."""
        print("📊 Ranking contexts")
        
        if not state["retrieved_contexts"]:
            state["ranked_contexts"] = []
            return state
        
        facts_text = self.fact_memory.get_facts_as_text(state.get('thread_id', 'default'))
        
        context_list = []
        for i, ctx in enumerate(state['retrieved_contexts']):
            content_preview = ctx['content'][:300].replace('\n', ' ')
            context_list.append(f"{i+1}. {content_preview}...")
        
        prompt = f"""
Rank these contexts by relevance to the query.

{facts_text}

Query: {state['original_query']}

Contexts:
{chr(10).join(context_list)}

Return indices (1-based) in order of relevance, most relevant first.
"""
        
        try:
            result: ContextRanking = self.rank_llm.invoke([
                SystemMessage(content=prompt)
            ])
            
            ranked = []
            for idx in result.ranked_indices:
                if 1 <= idx <= len(state["retrieved_contexts"]):
                    ctx = state["retrieved_contexts"][idx - 1].copy()
                    ctx["relevance_rank"] = len(ranked) + 1
                    ranked.append(ctx)
            
            state["ranked_contexts"] = ranked
            print(f"✅ Ranked {len(state['ranked_contexts'])} contexts")
        except Exception as e:
            print(f"⚠️ Ranking failed: {e}")
            state["ranked_contexts"] = state["retrieved_contexts"]
        
        return state
    
    def filter_contexts(self, state: GraphState) -> GraphState:
        """Filter contexts using structured output."""
        print("🔍 Filtering contexts")
        
        if not state["ranked_contexts"]:
            state["filtered_contexts"] = []
            state["final_context"] = ""
            return state
        
        facts_text = self.fact_memory.get_facts_as_text(state.get('thread_id', 'default'))
        
        context_list = []
        for i, ctx in enumerate(state['ranked_contexts']):
            content_preview = ctx['content'][:400].replace('\n', ' ')
            context_list.append(f"{i+1}. {content_preview}...")
        
        prompt = f"""
Select ONLY directly relevant contexts (2-5 max).

{facts_text}

Query: {state['original_query']}

Contexts:
{chr(10).join(context_list)}

Return indices of contexts to keep.
"""
        
        try:
            result: ContextFiltering = self.filter_llm.invoke([
                SystemMessage(content=prompt)
            ])
            
            filtered = []
            for idx in result.keep_indices:
                if 1 <= idx <= len(state["ranked_contexts"]):
                    filtered.append(state["ranked_contexts"][idx - 1])
            
            state["filtered_contexts"] = filtered
            print(f"✅ Filtered to {len(state['filtered_contexts'])} contexts")
        except Exception as e:
            print(f"⚠️ Filtering failed: {e}")
            state["filtered_contexts"] = state["ranked_contexts"][:3]
        
        # Build final context string
        context_parts = []
        for i, ctx in enumerate(state["filtered_contexts"]):
            source = ctx["metadata"].get("title") or ctx["metadata"].get("source", "Unknown")
            context_parts.append(f"Context {i+1} (Source: {source}):\n{ctx['content']}")
        
        state["final_context"] = "\n\n".join(context_parts)
        return state
    
    def decide_sufficiency(self, state: GraphState) -> GraphState:
        """Decide sufficiency using structured output."""
        print("🤔 Checking context sufficiency")
        
        # For conversational responses, context is usually sufficient
        has_history = len(state.get("messages", [])) > 1
        is_conversational = self.context_builder.detect_conversational_response(
            state["original_query"], 
            has_history
        )
        
        if is_conversational:
            print("💬 Conversational response - context sufficient from history")
            state["is_context_sufficient"] = True
            return state
        
        facts_text = self.fact_memory.get_facts_as_text(state.get('thread_id', 'default'))
        
        prompt = f"""
Is this context sufficient to answer the query?

{facts_text}

Query: {state['original_query']}

Context:
{state['final_context']}

Evaluate:
- Does context address the query directly?
- Is information complete?
- Would web search add significant value?
"""
        
        try:
            result: SufficiencyDecision = self.sufficiency_llm.invoke([
                SystemMessage(content=prompt)
            ])
            state["is_context_sufficient"] = result.sufficient
            print(f"✅ Sufficiency: {state['is_context_sufficient']} - {result.reasoning}")
        except Exception as e:
            print(f"⚠️ Decision failed: {e}")
            state["is_context_sufficient"] = False
        
        return state
    
    def route_based_on_sufficiency(self, state: GraphState) -> str:
        """Route based on context sufficiency."""
        return "sufficient" if state["is_context_sufficient"] else "insufficient"
    
    def web_search(self, state: GraphState) -> GraphState:
        """Perform web search."""
        print("🌐 Performing web search")
        
        if not self.search_tool:
            state["web_search_results"] = []
            return state
        
        try:
            results = self.search_tool.invoke({"query": state["original_query"]})
            web_contexts = []
            
            for result in results:
                web_contexts.append({
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "source_type": "web"
                })
            
            state["web_search_results"] = web_contexts
            
            web_text = "\n\n".join([
                f"Web Result {i+1} ({ctx['title']}):\n{ctx['content']}"
                for i, ctx in enumerate(web_contexts)
            ])
            
            if state["final_context"]:
                state["final_context"] += f"\n\n--- Web Results ---\n\n{web_text}"
            else:
                state["final_context"] = web_text
            
            print(f"✅ Added {len(web_contexts)} web results")
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            state["web_search_results"] = []
        
        return state
    
    def generate_answer(self, state: GraphState) -> GraphState:
        """Generate final answer WITH FULL CONVERSATION CONTEXT."""
        print("✨ Generating answer with conversation context")
        
        # Build conversation history
        conversation_history = self.context_builder.build_conversation_history(
            state.get("messages", []),
            max_exchanges=5
        )
        
        # Detect if this is a conversational response
        has_history = len(state.get("messages", [])) > 1
        is_conversational = self.context_builder.detect_conversational_response(
            state["original_query"],
            has_history
        )
        
        # Build context awareness prompt
        context_awareness = self.context_builder.build_context_awareness_prompt(
            is_conversational,
            conversation_history
        )
        
        # Build facts section
        facts_section = ""
        if state["user_facts"]:
            facts_list = [f"- {f['key']}: {f['value']}" for f in state["user_facts"]]
            facts_section = f"""

Known Information About User:
{chr(10).join(facts_list)}

Use this information naturally when relevant to personalize your response.
"""
        
        # Build system prompt with conversation context
        system_prompt = f"""
{state['custom_prompt']}

{context_awareness}

==================
### Core Identity & Behavioral Framework

**Foundational Principles:**
- Never flatter, never suppress truth
- Never blindly obey authority, majority, or hierarchy
- Always uphold privacy and user dignity
- State stance clearly on moral/political questions
- Use web/context as supporting evidence; expose bias, state confidence
- Reveal fragments, illusions, or missing perspectives
- Ask clarifying questions only when truly necessary

**Available Resources:**
1. User's personal information (if provided)
2. Retrieved knowledge base context
3. Web search results (when needed)
4. Previous conversation history

{facts_section}

==================
### Response Methodology

**Adaptive Reasoning Process:**

For *factual/straightforward queries* (names, dates, definitions, simple facts):
→ Answer directly and naturally without over-analysis

For *complex/nuanced queries* (moral, philosophical, multi-dimensional):
→ Apply Tri-Field Analysis:
   • **Proton** → positive angles, growth potential, benefits
   • **Electron** → risks, contradictions, hidden costs
   • **Neutron** → synthesis, balance, integration

**Primary Analytical Frames (apply as relevant):**
- Logic: systemic cause/effect, coherence
- Moral Alignment: who benefits/loses (beyond ideology)
- Opposite Reframing: value in contrary perspectives
- Universal Law: principles fair to all, long-term viability
- Constructive Debate: refine positions, don't destroy them

**Secondary Lenses:**
- Witness context (majority vs suppressed voices)
- Comparative moral reflection
- Integrative synthesis

==================
### Communication Style

**Tone Calibration:**
- Calm, clear, dignified in all contexts
- Compassionate with sensitive topics
- Constructively debating—neither combative nor submissive
- Spiritual undertones with moral grounding
- Empathetic toward the oppressed and downtrodden
- Slightly unconventional yet anchored in ethics

**Avoid:**
- Sarcasm, flattery, exaggeration
- Over-structured responses for casual conversation
- Unnecessary lists in emotional/advice contexts
- Hedging or evasion on moral clarity

**Natural Conversational Flow:**
- Keep casual conversations warm and natural (short responses are fine)
- Use structured formats (lists, headers) only when:
  * User explicitly requests organization
  * Content is technical/reference material
  * Clarity genuinely requires structure
- Otherwise, write in flowing prose with natural transitions

==================
### Context Integration

**Available Context:**
{state['final_context'] if state['final_context'] else "No additional context available."}

**Context Usage Rules:**
- Weave in relevant previous discussion naturally
- Use context as supporting material, never final authority
- Explicitly state when context is insufficient
- Reference user information organically when relevant
- Maintain reasoning independence while respecting evidence

**Limitations:**
- If context doesn't adequately address the query, clearly acknowledge gaps
- Don't fabricate connections between context and query
- Balance retrieved knowledge with critical analysis

==================
### Response Execution

**Current User Message:** {state['original_query']}

Now respond as *Neutral Intelligence*:
- Match complexity to question type
- Prioritize helpfulness, balance, and truth
- Let personality and reasoning emerge naturally
- Structure only where genuinely needed
- Maintain dignity and moral clarity throughout
- CRITICAL: If this is a conversational response (yes/no/elaborate), respond in context of the ongoing dialogue

When answering, be the conversation partner who thinks deeply when needed, speaks simply when appropriate, and always keeps human dignity at the center.
"""
        
        try:
            # Build message list with full conversation history
            conversation_messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history (last 10 messages for context window management)
            if state.get("messages") and len(state["messages"]) > 0:
                # Include recent messages but not the very last one if it's the current query
                recent_messages = state["messages"][-10:]
                
                for msg in recent_messages:
                    # Skip if this message is the current query (we'll add it separately)
                    if isinstance(msg, HumanMessage) and msg.content == state["original_query"]:
                        continue
                    conversation_messages.append(msg)
            
            # Add current query
            conversation_messages.append(HumanMessage(content=state["original_query"]))
            
            # Log conversation structure for debugging
            print(f"📝 Conversation structure: {len(conversation_messages)} messages")
            print(f"   - System: 1")
            print(f"   - History: {len(conversation_messages) - 2}")
            print(f"   - Current: 1")
            
            # Generate response with full context
            response = self.answer_llm.invoke(conversation_messages)
            
            state["final_answer"] = response.content
            state["messages"].append(AIMessage(content=response.content))
            
            print("✅ Answer generated with full conversation context")
            print(f"   Response length: {len(response.content)} chars")
            
        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            error_message = "I apologize, but I encountered an error while generating a response. Please try rephrasing your question."
            state["final_answer"] = error_message
            state["messages"].append(AIMessage(content=error_message))
        
        return state
    
    def extract_facts(self, state: GraphState) -> GraphState:
        """Extract and store user facts using structured output."""
        print("💾 Extracting user facts")
        
        thread_id = state.get("thread_id", "default")
        
        conversation_turn = {
            "user": state["original_query"],
            "assistant": state["final_answer"]
        }
        
        new_facts = self.fact_memory.extract_and_store_facts(
            thread_id, 
            conversation_turn, 
            self.fact_extraction_llm
        )
        
        if new_facts:
            print(f"✅ Extracted {len(new_facts)} new facts")
            for fact in new_facts:
                print(f"   📌 {fact['key']}: {fact['value']}")
        else:
            print("ℹ️ No new facts extracted")
        
        return state
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def process_query(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Process a query through the workflow."""
        print(f"\n{'='*80}")
        print(f"🚀 Processing query: '{query}'")
        print(f"🧵 Thread: {thread_id}")
        print(f"{'='*80}")
        
        initial_state = {
            "original_query": query,
            "thread_id": thread_id,
            "user_facts": [],
            "decomposed_queries": [],
            "retrieved_contexts": [],
            "ranked_contexts": [],
            "filtered_contexts": [],
            "web_search_results": None,
            "final_context": "",
            "is_context_sufficient": False,
            "custom_prompt": "",
            "final_answer": "",
            "messages": [HumanMessage(content=query)]
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            result = {
                "success": True,
                "query": query,
                "answer": final_state["final_answer"],
                "thread_id": thread_id,
                "metadata": {
                    "user_facts_count": len(final_state["user_facts"]),
                    "contexts_used": len(final_state["filtered_contexts"]),
                    "web_search_used": final_state["web_search_results"] is not None,
                    "context_sufficient": final_state["is_context_sufficient"],
                    "conversation_length": len(final_state["messages"])
                },
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"\n{'='*80}")
            print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"📊 Stats:")
            print(f"   - Answer length: {len(result['answer'])} chars")
            print(f"   - Facts stored: {result['metadata']['user_facts_count']}")
            print(f"   - Contexts used: {result['metadata']['contexts_used']}")
            print(f"   - Conversation length: {result['metadata']['conversation_length']} messages")
            print(f"   - Web search used: {result['metadata']['web_search_used']}")
            print(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ WORKFLOW FAILED")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            
            return {
                "success": False,
                "query": query,
                "thread_id": thread_id,
                "error": str(e),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
    
    def get_user_facts(self, thread_id: str) -> Dict[str, str]:
        """Get stored facts for a user."""
        return self.fact_memory.get_facts(thread_id)
    
    def get_conversation_summary(self, thread_id: str) -> Dict[str, Any]:
        """Get conversation summary for a thread."""
        facts = self.fact_memory.get_facts(thread_id)
        
        return {
            "thread_id": thread_id,
            "facts": facts,
            "fact_count": len(facts),
            "summary": self.fact_memory.get_facts_as_text(thread_id) or "No facts stored yet."
        }
    
    def clear_thread(self, thread_id: str):
        """Clear all data for a thread."""
        self.fact_memory.clear_facts(thread_id)
        print(f"✅ Cleared thread: {thread_id}")


# ============================================================================
# FASTAPI SERVICE INTEGRATION
# ============================================================================

class RAGService:
    """Production service for FastAPI integration."""
    
    def __init__(self, env_path: str = ".env"):
        self.workflow = MultiAgentRAGWorkflow(env_path)
        print("✅ RAG Service initialized")
    
    async def answer_query(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Process query with automatic thread ID generation."""
        if thread_id is None:
            thread_id = f"session_{datetime.now(timezone.utc).timestamp()}"
        
        return await self.workflow.process_query(query, thread_id)
    
    def get_user_profile(self, thread_id: str) -> Dict[str, Any]:
        """Get user's stored facts and conversation summary."""
        return self.workflow.get_conversation_summary(thread_id)
    
    def clear_conversation(self, thread_id: str):
        """Clear conversation and facts."""
        self.workflow.clear_thread(thread_id)
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "vector_store": self.workflow.vector_store is not None,
                "search_tool": self.workflow.search_tool is not None,
                "llm": True
            }
        }


# ============================================================================
# COMPREHENSIVE TESTING
# ============================================================================

async def test_conversation_flow():
    """Test conversation flow with context awareness."""
    from pathlib import Path
    
    backend_env_path = Path(__file__).parent.parent / ".env"
    workflow = MultiAgentRAGWorkflow(str(backend_env_path))
    
    thread_id = "test_conversation_001"
    
    print("\n" + "="*100)
    print("🧪 TESTING CONVERSATION FLOW WITH CONTEXT AWARENESS")
    print("="*100)
    
    test_scenarios = [
        {
            "name": "Initial Question",
            "queries": [
                "Hi, my name is Abdullah and I'm working on a RAG chatbot project",
                "What is the purpose of life?"
            ]
        },
        {
            "name": "Disagreement Test",
            "queries": [
                "no"
            ],
            "expected_behavior": "Should acknowledge disagreement, not define 'no'"
        },
        {
            "name": "Follow-up Elaboration",
            "queries": [
                "elaborate on that"
            ],
            "expected_behavior": "Should continue previous topic, not define 'elaborate'"
        },
        {
            "name": "Agreement Test",
            "queries": [
                "Should we help others?",
                "yes"
            ],
            "expected_behavior": "Should acknowledge agreement, not define 'yes'"
        },
        {
            "name": "Context Recall",
            "queries": [
                "Do you remember my name?",
                "What project am I working on?"
            ],
            "expected_behavior": "Should recall facts from earlier conversation"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*100}")
        print(f"📋 SCENARIO: {scenario['name']}")
        if 'expected_behavior' in scenario:
            print(f"Expected: {scenario['expected_behavior']}")
        print(f"{'='*100}\n")
        
        for query in scenario['queries']:
            print(f"\n{'>'*50}")
            print(f"👤 USER: {query}")
            print(f"{'>'*50}")
            
            result = await workflow.process_query(query, thread_id)
            
            if result["success"]:
                print(f"\n🤖 ASSISTANT:\n{result['answer']}\n")
                
                # Show facts
                facts = workflow.get_user_facts(thread_id)
                if facts:
                    print(f"\n📚 Stored Facts: {facts}")
            else:
                print(f"\n❌ Error: {result.get('error')}")
            
            print(f"\n{'<'*50}\n")
            
            # Small delay between queries
            await asyncio.sleep(1)
    
    print(f"\n{'='*100}")
    print("✅ CONVERSATION FLOW TEST COMPLETED")
    print(f"{'='*100}")
    
    # Final summary
    summary = workflow.get_conversation_summary(thread_id)
    print(f"\n📊 Final Summary:")
    print(f"   - Thread: {summary['thread_id']}")
    print(f"   - Facts collected: {summary['fact_count']}")
    print(f"   - Facts: {summary['facts']}")


async def test_edge_cases():
    """Test edge cases and error handling."""
    from pathlib import Path
    
    backend_env_path = Path(__file__).parent.parent / ".env"
    workflow = MultiAgentRAGWorkflow(str(backend_env_path))
    
    thread_id = "test_edge_cases_001"
    
    print("\n" + "="*100)
    print("🧪 TESTING EDGE CASES")
    print("="*100)
    
    edge_cases = [
        ("Empty-like query", "   "),
        ("Single character", "?"),
        ("Very long query", "What is " * 100 + "the meaning of life?"),
        ("Special characters", "!@#$%^&*()"),
        ("Mixed languages", "What is जीवन का purpose?"),
    ]
    
    for name, query in edge_cases:
        print(f"\n{'='*80}")
        print(f"Test: {name}")
        print(f"Query: '{query[:100]}...' " if len(query) > 100 else f"Query: '{query}'")
        print(f"{'='*80}")
        
        try:
            result = await workflow.process_query(query, thread_id)
            print(f"✅ Handled successfully")
            print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        except Exception as e:
            print(f"⚠️ Exception: {str(e)}")
        
        await asyncio.sleep(0.5)
    
    print(f"\n{'='*100}")
    print("✅ EDGE CASE TESTING COMPLETED")
    print(f"{'='*100}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    import asyncio
    
    async def run_all_tests():
        print("\n" + "🎯"*50)
        print("STARTING COMPREHENSIVE RAG WORKFLOW TESTS")
        print("🎯"*50 + "\n")
        
        # Test 1: Conversation Flow
        await test_conversation_flow()
        
        # Small break
        await asyncio.sleep(2)
        
        # Test 2: Edge Cases
        await test_edge_cases()
        
        print("\n" + "🎉"*50)
        print("ALL TESTS COMPLETED")
        print("🎉"*50 + "\n")
    
    # Run tests
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()