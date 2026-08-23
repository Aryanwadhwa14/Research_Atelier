"""
streamlit_app.py — Moody Chestnut Glassmorphism & High-Contrast Typography
Verified against Streamlit version: 1.62.0

Features:
1. Rich, moody roasted chestnut backdrop with warm ambient lighting (not too light)
2. Frosted glassmorphic espresso bubbles with subtle caramel borders & backdrop blur
3. Crystal-clear, high-contrast typography across all buttons, prompt examples, cards, and inputs
4. Elegant styling with Fraunces, Outfit, and Space Mono (no emojis)
"""

import json
import requests
import streamlit as st
from theme_tokens import TOKENS

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend URL ────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"


# ── Single-Point CSS Injection ────────────────────────────────────────────────
def inject_theme_css() -> None:
    """
    Renders a single CSS injection block using CSS custom properties
    derived directly from theme_tokens.py.
    Verified against Streamlit 1.62.0.
    """
    css = f"""
    <style>
    /* Google Fonts: Fraunces (stylish/wavy display serif), Outfit (matching content sans), Space Mono */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..800;1,9..144,400..700&family=Outfit:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@500;700;800&display=swap');

    :root {{
        --color-bg-canvas: {TOKENS['color_bg_canvas']};
        --color-bg-glass: {TOKENS['color_bg_glass']};
        --color-bg-glass-hover: {TOKENS['color_bg_glass_hover']};
        --color-bg-glass-subtle: {TOKENS['color_bg_glass_subtle']};
        --color-border-glass: {TOKENS['color_border_glass']};
        --color-border-glass-hover: {TOKENS['color_border_glass-hover'] if 'color_border_glass-hover' in TOKENS else 'rgba(235, 172, 120, 0.55)'};
        --color-bg-sidebar: {TOKENS['color_bg_sidebar']};
        --color-bg-sidebar-card: {TOKENS['color_bg_sidebar_card']};
        --color-border-sidebar: {TOKENS['color_border_sidebar']};
        --color-text-primary: {TOKENS['color_text_primary']};
        --color-text-secondary: {TOKENS['color_text_secondary']};
        --color-text-tertiary: {TOKENS['color_text_tertiary']};
        --color-text-inverse: {TOKENS['color_text_inverse']};
        --color-accent: {TOKENS['color_accent']};
        --color-accent-hover: {TOKENS['color_accent_hover']};
        --color-accent-subtle: {TOKENS['color_accent_subtle']};
        --color-success: {TOKENS['color_success']};
        --color-warning: {TOKENS['color_warning']};
        --color-error: {TOKENS['color_error']};
        --font-heading: {TOKENS['font_heading']};
        --font-sans: {TOKENS['font_sans']};
        --font-mono: {TOKENS['font_mono']};
        --radius-sm: {TOKENS['radius_sm']};
        --radius-md: {TOKENS['radius_md']};
        --radius-lg: {TOKENS['radius_lg']};
        --radius-full: {TOKENS['radius_full']};
        --shadow-glass: {TOKENS['shadow_glass']};
        --shadow-glass-hover: {TOKENS['shadow_glass_hover']};
    }}

    /* Global Body & Main Background (Moody Roasted Chestnut / Amber Canvas) */
    html, body, [class*="css"] {{
        font-family: var(--font-sans) !important;
        background-color: var(--color-bg-canvas) !important;
        color: var(--color-text-primary) !important;
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background: radial-gradient(circle at 50% -10%, #3D2B22 0%, #261A14 55%, #180F0A 100%) !important;
        color: var(--color-text-primary) !important;
    }}

    /* Top App Header & Container */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .block-container {{
        max-width: 920px !important;
        padding-top: 4.5rem !important;
        padding-bottom: 4rem !important;
    }}

    /* Deep Dark Espresso Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--color-bg-sidebar) !important;
        border-right: 1px solid var(--color-border-sidebar) !important;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: var(--color-border-sidebar) !important;
        margin: 1.25rem 0 !important;
    }}

    /* Glassmorphic Sidebar Cards */
    .sidebar-glass-card {{
        background: var(--color-bg-sidebar-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--color-border-glass);
        border-radius: var(--radius-lg);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow-glass);
        color: var(--color-text-primary);
    }}

    .sidebar-glass-title {{
        font-family: var(--font-heading) !important;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--color-accent);
        margin-bottom: 0.75rem;
    }}

    .status-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: var(--color-text-secondary);
        margin-bottom: 0.5rem;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--color-accent);
        box-shadow: 0 0 10px rgba(222, 156, 104, 0.8);
    }}

    .status-dot.offline {{
        background-color: var(--color-error);
        box-shadow: 0 0 10px rgba(224, 92, 92, 0.8);
    }}

    /* Hero Section */
    .hero-container {{
        margin-bottom: 2.2rem;
    }}

    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--color-bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--color-border-glass);
        color: var(--color-accent);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: var(--radius-full);
        margin-bottom: 14px;
        font-family: var(--font-mono);
        box-shadow: var(--shadow-glass);
    }}

    .hero-title {{
        font-family: var(--font-heading) !important;
        font-size: 2.65rem;
        font-weight: 700;
        font-style: italic;
        letter-spacing: -0.02em;
        line-height: 1.18;
        background: linear-gradient(135deg, #FFFFFF 20%, #F5E5D5 65%, #DE9C68 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }}

    .hero-subtitle {{
        color: var(--color-text-secondary) !important;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }}

    /* Section Headers */
    .section-label {{
        font-family: var(--font-heading) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: var(--color-accent) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 10px !important;
    }}

    /* Glassmorphic Prompt Buttons (High-Contrast Fix) */
    .stButton > button {{
        background: var(--color-bg-glass) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        color: #FAF5F0 !important;
        border: 1px solid var(--color-border-glass) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.2rem !important;
        box-shadow: var(--shadow-glass) !important;
        transition: all 0.25s ease !important;
        font-family: var(--font-sans) !important;
        text-align: left !important;
    }}

    .stButton > button * {{
        color: #FAF5F0 !important;
        font-weight: 600 !important;
    }}

    .stButton > button:hover {{
        background: var(--color-bg-glass-hover) !important;
        border-color: rgba(235, 172, 120, 0.65) !important;
        box-shadow: var(--shadow-glass-hover) !important;
        transform: translateY(-2px) !important;
    }}

    .stButton > button:hover * {{
        color: #FFFFFF !important;
    }}

    /* Query Input (Glassmorphic Container with High Contrast Text) */
    div[data-baseweb="input"] {{
        background: var(--color-bg-glass) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--color-border-glass) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-glass) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        padding: 5px 8px !important;
    }}

    div[data-baseweb="input"]:focus-within {{
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 2px var(--color-accent-subtle), var(--shadow-glass-hover) !important;
    }}

    input.stTextInput {{
        color: #FAF5F0 !important;
        font-size: 0.95rem !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
    }}

    input.stTextInput::placeholder {{
        color: var(--color-text-tertiary) !important;
    }}

    /* Primary Submit Button (Glowing Caramel CTA) */
    div[data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(135deg, #EBAC78 0%, #DE9C68 50%, #C98A58 100%) !important;
        color: #160E0A !important;
        border: 1px solid #DE9C68 !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.25s ease !important;
        font-family: var(--font-heading) !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 6px 20px rgba(222, 156, 104, 0.45) !important;
        text-align: center !important;
    }}

    div[data-testid="stFormSubmitButton"] > button * {{
        color: #160E0A !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stFormSubmitButton"] > button:hover {{
        background: linear-gradient(135deg, #F5BA88 0%, #EBAC78 100%) !important;
        box-shadow: 0 8px 28px rgba(222, 156, 104, 0.65) !important;
        transform: translateY(-2px) !important;
    }}

    /* Glassmorphic Result Bubble Card */
    .result-glass-bubble {{
        background: var(--color-bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--color-border-glass);
        border-radius: var(--radius-lg);
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: var(--shadow-glass);
        color: #FAF5F0 !important;
    }}

    .result-glass-bubble p, .result-glass-bubble span, .result-glass-bubble li, .result-glass-bubble div {{
        color: #FAF5F0 !important;
        font-size: 0.95rem;
        line-height: 1.65;
    }}

    .result-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--color-border-glass);
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
        font-family: var(--font-heading) !important;
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--color-accent);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* Expanders (Reasoning Trace Glass Bubbles) */
    .streamlit-expanderHeader {{
        background: var(--color-bg-glass) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--color-border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: #FAF5F0 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        font-family: var(--font-heading) !important;
        box-shadow: var(--shadow-glass) !important;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--color-accent) !important;
        color: var(--color-accent) !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background: var(--color-bg-glass-subtle) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--color-border-glass) !important;
        border-top: none !important;
        border-bottom-left-radius: var(--radius-md) !important;
        border-bottom-right-radius: var(--radius-md) !important;
        padding: 1.2rem !important;
        color: #FAF5F0 !important;
    }}

    div[data-testid="stExpanderDetails"] p, div[data-testid="stExpanderDetails"] span {{
        color: #FAF5F0 !important;
    }}

    /* Code Blocks Inside Bubbles */
    pre, code {{
        font-family: var(--font-mono) !important;
        background-color: #0E0705 !important;
        border: 1px solid var(--color-border-glass) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.825rem !important;
        color: #F2DFC9 !important;
    }}

    /* Tool Badges inside Glass Bubbles */
    .tool-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-family: var(--font-mono);
        font-weight: 600;
    }}
    .badge-web {{ background: rgba(222, 156, 104, 0.15); border: 1px solid rgba(222, 156, 104, 0.4); color: #EBAC78; }}
    .badge-github {{ background: rgba(198, 134, 88, 0.15); border: 1px solid rgba(198, 134, 88, 0.4); color: #DE9C68; }}
    .badge-arxiv {{ background: rgba(235, 172, 120, 0.15); border: 1px solid rgba(235, 172, 120, 0.4); color: #F5BA88; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Render CSS Block ──────────────────────────────────────────────────────────
inject_theme_css()


# ── Check Backend Health ──────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def check_backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/docs", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


backend_online = check_backend_health()


# ── Deep Dark Espresso Sidebar with Glass Cards ───────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-family: var(--font-heading); font-weight: 700; font-size: 1.25rem; font-style: italic; color: #FAF5F0;">Research Atelier</div>
            <div style="font-size: 0.75rem; color: var(--color-text-tertiary); font-family: var(--font-mono); margin-top: 2px;">Autonomous Agent Environment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Health status card
    dot_class = "status-dot" if backend_online else "status-dot offline"
    status_text = "FastAPI Service Live" if backend_online else "FastAPI Service Offline"

    st.markdown(
        f"""
        <div class="sidebar-glass-card">
            <div class="sidebar-glass-title">System Status</div>
            <div class="status-item">
                <span class="{dot_class}"></span>
                <span>{status_text}</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--color-text-tertiary); font-family: var(--font-mono); margin-top: 4px;">
                port: 8000 &bull; env: local
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active tools card
    st.markdown(
        """
        <div class="sidebar-glass-card">
            <div class="sidebar-glass-title">Integrated Tools (3)</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <span class="tool-badge badge-web">web_search &bull; DuckDuckGo</span>
                <span class="tool-badge badge-github">github_repo_info &bull; GitHub REST</span>
                <span class="tool-badge badge-arxiv">arxiv_tool &bull; arXiv Papers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Architecture card
    st.markdown(
        """
        <div class="sidebar-glass-card">
            <div class="sidebar-glass-title">Architecture</div>
            <div style="font-size: 0.8125rem; color: var(--color-text-secondary); line-height: 1.7;">
                <div><b>LangGraph</b> ReAct Agent</div>
                <div><b>Groq</b> Llama 3.3 70B</div>
                <div><b>Glassmorphic</b> Espresso</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Content Area (Moody Roasted Chestnut Canvas) ─────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">Autonomous Research Assistant</div>
        <h1 class="hero-title">Ask deeply. Uncover truth.</h1>
        <p class="hero-subtitle">
            An intelligent research engine combining real-time web discovery, GitHub repositories, and arXiv academic papers with transparent reasoning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Suggested Prompt Bubbles (High-Contrast Glass Buttons) ─────────────────────
st.markdown("<p class='section-label'>Curated Prompts</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

selected_prompt = ""
with col1:
    if st.button("langchain-ai/langchain stars", key="chip1", use_container_width=True):
        selected_prompt = "How many stars does langchain-ai/langchain have?"
with col2:
    if st.button("Retrieval Augmented Generation papers", key="chip2", use_container_width=True):
        selected_prompt = "Find recent papers on retrieval augmented generation"
with col3:
    if st.button("FastAPI repository details", key="chip3", use_container_width=True):
        selected_prompt = "Give me repository details and link for tiangolo/fastapi"

# ── Query Input Form ──────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Query",
        value=selected_prompt if selected_prompt else "",
        placeholder="Explore academic papers, search the web, or analyze GitHub repositories...",
        label_visibility="collapsed",
    )
    submit_pressed = st.form_submit_button("Inquire & Execute", use_container_width=True)

# ── State Mapping & Execution ─────────────────────────────────────────────────
query_to_run = user_query.strip() if user_query.strip() else selected_prompt

if (submit_pressed or selected_prompt) and query_to_run:
    if not backend_online:
        st.markdown(
            """
            <div style="background: var(--color-bg-glass); backdrop-filter: blur(16px); border: 1px solid #E05C5C; border-radius: 12px; padding: 14px; margin-top: 1rem; color: #FAF5F0; font-size: 0.875rem;">
                <span style="color: #E05C5C; font-weight: 700; font-family: var(--font-heading);">Service Notice:</span> Cannot connect to the FastAPI backend at <code>http://localhost:8000</code>.
                <div style="margin-top: 6px; color: #D8C7B8; font-size: 0.8125rem;">
                    Please run: <code>uvicorn main:app --reload</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Loading State
        with st.spinner("Synthesizing knowledge and invoking agent tools..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": query_to_run},
                    timeout=120,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    steps = data.get("steps", [])

                    # Glassmorphic Answer Bubble
                    st.markdown(
                        f"""
                        <div class="result-glass-bubble">
                            <div class="result-header">
                                <span>Synthesized Answer</span>
                                <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--color-accent);">{len(steps)} tool(s) invoked</span>
                            </div>
                            <div style="color: #FAF5F0; font-size: 0.95rem; line-height: 1.65;">
                                {answer}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Reasoning Steps Trace (Glassmorphic Expanders)
                    if steps:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("<p class='section-label'>Reasoning Trace</p>", unsafe_allow_html=True)

                        for idx, step in enumerate(steps, 1):
                            t_name = step.get("tool", "unknown")

                            with st.expander(f"Step {idx}: {t_name}", expanded=True):
                                st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: var(--color-accent); text-transform: uppercase;'>Query Parameters</p>", unsafe_allow_html=True)
                                st.code(json.dumps(step.get("query", {}), indent=2), language="json")

                                st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: var(--color-accent); text-transform: uppercase; margin-top: 10px;'>Output</p>", unsafe_allow_html=True)
                                raw_result = step.get("result", "")
                                if len(raw_result) > 3000:
                                    st.markdown(raw_result[:3000] + "\n\n*...[output truncated]*")
                                else:
                                    st.markdown(raw_result)
                else:
                    err_msg = res.json().get("detail", "An unexpected error occurred.")
                    st.error(f"Backend returned error {res.status_code}: {err_msg}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The agent reasoning took longer than 120 seconds.")
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")

elif submit_pressed and not query_to_run:
    st.warning("Please enter a research question or select one of the curated prompt suggestions above.")
