import streamlit as st
import arxiv

from langchain_groq import ChatGroq

from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)

from langchain_community.utilities import (
    WikipediaAPIWrapper
)

from langchain_core.tools import tool

from langchain.agents import create_agent


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LangChain Chat with Search",
    page_icon="🔎",
    layout="wide"
)

st.title("LangChain - Chat with Search")

st.caption(
    "Search the Web, Wikipedia and arXiv using a Groq-powered AI agent."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Available Tools**

    🔎 Web Search

    📚 Wikipedia

    📄 arXiv Research Papers
    """
)


# =========================================================
# ARXIV TOOL
# =========================================================

@tool
def search_arxiv(query: str) -> str:
    """
    Search arXiv for academic research papers.

    Use this tool when the user asks for:
    - research papers
    - academic papers
    - AI papers
    - machine learning papers
    - deep learning papers
    - NLP papers
    - computer vision papers
    - LLM papers
    - Generative AI research
    - scientific research
    """

    try:

        client = arxiv.Client(
            page_size=3,
            delay_seconds=3,
            num_retries=2
        )

        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []

        for paper in client.results(search):

            # -------------------------------------------------
            # AUTHORS
            # -------------------------------------------------

            authors = ", ".join(
                author.name
                for author in paper.authors[:5]
            )

            # -------------------------------------------------
            # SHORT ABSTRACT
            # -------------------------------------------------

            abstract = paper.summary.replace("\n", " ").strip()

            # Only send a small portion to the LLM
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            paper_text = (
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.date()}\n"
                f"Abstract: {abstract}\n"
                f"arXiv URL: {paper.entry_id}"
            )

            results.append(paper_text)

        # -----------------------------------------------------
        # NO RESULTS
        # -----------------------------------------------------

        if not results:

            return (
                f"No arXiv papers were found for: {query}"
            )

        # -----------------------------------------------------
        # RETURN SMALL RESULT
        # -----------------------------------------------------

        return "\n\n---\n\n".join(results)

    except Exception as e:

        return (
            "arXiv search failed.\n"
            f"Error: {str(e)}"
        )


# =========================================================
# WIKIPEDIA TOOL
# =========================================================

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1200
)

wiki = WikipediaQueryRun(
    name="wikipedia_search",
    description=(
        "Search Wikipedia for general factual information, "
        "definitions, concepts, people, organizations, "
        "history and background information."
    ),
    api_wrapper=wiki_wrapper
)


# =========================================================
# DUCKDUCKGO WEB SEARCH TOOL
# =========================================================

search = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the internet for current information, "
        "recent events, news, websites and general information."
    )
)


# =========================================================
# TOOLS
# =========================================================

tools = [
    search,
    wiki,
    search_arxiv
]


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your research assistant.\n\n"
                "I can search:\n"
                "- 🔎 The Web\n"
                "- 📚 Wikipedia\n"
                "- 📄 arXiv research papers\n\n"
                "Ask me anything!"
            )
        }
    ]


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Ask something..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if prompt:

    # =====================================================
    # API KEY CHECK
    # =====================================================

    if not api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()


    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(prompt)


    # =====================================================
    # GROQ MODEL
    # =====================================================

    llm = ChatGroq(
        groq_api_key=api_key,

        model="openai/gpt-oss-20b",

        temperature=0,

        max_tokens=400,

        # Reduce reasoning overhead
        reasoning_effort="low",

        streaming=True
    )


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    system_prompt = """
You are a concise AI research assistant.

You have three tools.

1. search_arxiv
Use this for academic papers and scientific research.

2. wikipedia_search
Use this for general factual information, concepts,
definitions, history and background.

3. web_search
Use this for current information, websites, news
and recent events.

TOOL SELECTION:

If the user asks for research papers:
USE search_arxiv.

If the user asks about a general concept:
USE wikipedia_search when useful.

If the user asks for current information:
USE web_search.

IMPORTANT:

- Do not invent papers.
- Do not invent authors.
- Do not invent URLs.
- Keep answers concise.
- If using arXiv, provide the paper title,
  authors, short summary and URL.
- Do not reproduce long abstracts.
- Normally discuss no more than 2 papers.
"""


    # =====================================================
    # CREATE AGENT
    # =====================================================

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )


    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        try:

            # -------------------------------------------------
            # RUN AGENT
            # -------------------------------------------------

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            )


            # -------------------------------------------------
            # GET MESSAGES
            # -------------------------------------------------

            messages = result.get(
                "messages",
                []
            )


            if not messages:

                response = (
                    "The agent did not return a response."
                )

            else:

                # Find the last AI message
                response = None

                for message in reversed(messages):

                    if getattr(
                        message,
                        "type",
                        None
                    ) == "ai":

                        response = message.content

                        break


                if response is None:

                    response = str(
                        messages[-1].content
                    )


            # -------------------------------------------------
            # SAFE RESPONSE CONVERSION
            # -------------------------------------------------

            if not isinstance(response, str):

                response = str(response)


            # -------------------------------------------------
            # DISPLAY RESPONSE
            # -------------------------------------------------

            st.markdown(response)


            # -------------------------------------------------
            # SAVE RESPONSE
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )


        except Exception as e:

            error_text = str(e)

            st.error(
                "An error occurred while running the agent."
            )

            # -------------------------------------------------
            # TOKEN ERROR
            # -------------------------------------------------

            if (
                "Request too large" in error_text
                or
                "tokens per minute" in error_text
                or
                "rate_limit_exceeded" in error_text
            ):

                st.warning(
                    """
                    The request was too large for the
                    current Groq token limit.

                    Try asking for fewer papers or a
                    shorter answer.
                    """
                )

            # -------------------------------------------------
            # OTHER ERROR
            # -------------------------------------------------

            st.exception(e)