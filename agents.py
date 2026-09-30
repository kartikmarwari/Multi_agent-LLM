import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url, multi_search
import os
from dotenv import load_dotenv

# =========================
# SETUP
# =========================
load_dotenv()

def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429

# Mistral free tier allows ~1 request/sec; pace every call so a single
# pipeline run (agents make several calls in a loop) doesn't trip 429s.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

base_model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.2,   # slight creativity boost
    max_tokens=2000,
    rate_limiter=rate_limiter,
)

# Optional backup provider: set GROQ_API_KEY to fall back to Groq
# when Mistral keeps rate-limiting.
fallback_model = (
    ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_tokens=2000)
    if os.getenv("GROQ_API_KEY") else None
)

# Used by the plain chains (splitter / planner / writer / critic)
model = base_model.with_retry(
    retry_if_exception_type=(httpx.HTTPStatusError,),
    wait_exponential_jitter=True,
    stop_after_attempt=4,
)
if fallback_model:
    model = model.with_fallbacks([fallback_model])

# Used by the tool-calling agents (create_agent needs a real chat model,
# so retry/fallback go in as middleware instead). Fallback must come first
# (outermost) so Mistral is retried before switching providers.
agent_middleware = [ModelFallbackMiddleware(fallback_model)] if fallback_model else []
agent_middleware.append(
    ModelRetryMiddleware(
        max_retries=3,
        retry_on=_is_rate_limit,
        initial_delay=2.0,
        backoff_factor=2.0,
        on_failure="error",
    )
)

# =========================
# QUESTION SPLITTER AGENT (NEW)
# =========================
# Job: take the raw topic and break it into 3-4 focused
# sub-questions so research agents have more to dig into.
splitter_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""You are an elite research planning assistant (2026-level intelligence).

Your job:

- Deeply understand the user's intent (not just the topic)
- Break the topic into 5–6 HIGHLY DISTINCT sub-questions
- Each sub-question must explore a DIFFERENT dimension:
  (example: definition, current trends, real-world use, challenges, future, comparisons, etc.)

STRICT RULES:

- NO two questions should overlap in meaning
- NO reworded duplicates
- Each question must unlock NEW information
- Questions must be specific, not generic
- Avoid vague phrasing like "what is", "explain", unless necessary

QUALITY CHECK BEFORE OUTPUT:

- Ask yourself: "Will each question produce different answers?"
- If NO → rewrite

OUTPUT FORMAT:

- Return ONLY sub-questions
- Separate using '|'
- No numbering, no explanation, no extra text

 
"""),
    ("human", "Topic: {topic}")
])

splitter_chain = splitter_prompt | model | StrOutputParser()

def build_question_splitter():
    return splitter_chain


# =========================
# SEARCH AGENT
# =========================
def build_search_agent():
    return create_agent(
        model=base_model,
        middleware=agent_middleware,
        tools=[web_search, multi_search],
        system_prompt="""You are a high-precision web research assistant.

GOAL:
Return ONLY high-value, non-redundant information.

STRICT ANTI-REPETITION RULES:

- NEVER include duplicate insights
- If multiple sources say the same thing → keep ONLY the most informative one
- DO NOT paraphrase duplicates
- Each result must add NEW knowledge

QUALITY RULES:

- Prefer recent (2024–2026), credible sources
- Avoid generic/blog-level content
- Prioritize depth, not quantity

BEHAVIOR:

- If multiple queries → MUST use multi_search
- If single query → use web_search

OUTPUT RULES:

- 5–6 UNIQUE results per sub-question
- Each result must include:
  - Title
  - URL
  - Summary (must contain a UNIQUE insight)

SELF-CHECK BEFORE OUTPUT:

- Remove repeated ideas
- Remove weak/generic summaries
- Ensure every point adds value

FINAL RULE:
If two points are similar → DELETE one."""
    )

# =========================
# READER AGENT
# =========================
def build_reader_agent():
    return create_agent(
        model=base_model,
        middleware=agent_middleware,
        tools=[scrape_url],
        system_prompt="""You are a deep research extraction agent.

MANDATORY:

- ALWAYS use scrape_url tool
- NEVER answer without scraping
- Use 4–5 HIGH-QUALITY URLs

ANTI-REPETITION RULES:

- If same idea appears multiple times → MERGE into ONE strong insight
- NEVER restate the same fact
- NO paraphrased duplicates
- Each bullet must be UNIQUE

EXTRACTION STRATEGY:

- Extract only:
  - unique facts
  - key arguments
  - strong insights
- Ignore fluff, ads, navigation text

PRIORITY:

1. Unique insights
2. Data/statistics
3. Expert opinions

OUTPUT FORMAT:

Key Facts:
- Only DISTINCT facts

Important Data:
- Only numbers, stats (NO repetition)

Key Arguments:
- Unique viewpoints only

FINAL CHECK:

- Remove duplicates
- Merge similar ideas
- Keep only highest-value insights
"""
    )

# =========================
# PLANNER AGENT (NEW)
# =========================
# Job: look at search_result + scraped_content and decide
# if any sub-question is still weakly covered, and suggest
# one more focused follow-up search query. Keeps the
# research loop "smarter" without adding heavy complexity.
planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""You are a research gap-analysis expert.

GOAL:
Find what is TRULY missing — not what is already covered.

STRICT RULES:

- DO NOT repeat any existing topic
- DO NOT suggest broader/general queries
- Suggest ONLY something NEW and UNEXPLORED

DEEP THINKING:

- Look for:
  - missing perspectives
  - ignored edge cases
  - deeper technical angles
  - future implications not covered

OUTPUT:

- ONE highly focused follow-up query
- Must unlock NEW information

SELF-CHECK:

- If answer already exists in research → REJECT it
- If too generic → REWRITE it
"""),
    ("human",
"""Topic: {topic}

Research so far:
{research}

What is the one follow-up search query needed?""")
])

planner_chain = planner_prompt | model | StrOutputParser()

def build_planner_agent():
    return planner_chain


# =========================
# WRITER CHAIN (UPGRADED)
# =========================
writer_prompt = ChatPromptTemplate.from_messages([
   ("system",
"""You are a senior research analyst writing a publication-quality report.

MISSION:
Produce a CLEAN, INSIGHT-DENSE, NON-REPETITIVE report.

STRICT ANTI-REPETITION RULES:

- NEVER repeat ideas across sections
- NEVER rephrase the same point
- Each section must contain COMPLETELY NEW information
- Merge duplicate ideas into ONE strong insight

SECTION DIFFERENTIATION (VERY IMPORTANT):

- Insights → Core takeaways ONLY
- Trends → What is happening NOW (2026)
- Challenges → Real-world problems ONLY
- Opportunities → Future potential ONLY
- Detailed Analysis → NEW deep reasoning (NOT repetition)

WRITING RULES:

- Be precise, not verbose
- Avoid generic statements
- Use meaningful, information-dense bullets

QUALITY CHECK:

- If a point appears twice → REMOVE one
- If a section overlaps → FIX it

FORMAT:

# 📊 Research Report: {topic}

## Key Insights
- Unique takeaways only

## Trends (2026)
- Current developments only

## Challenges
- Real problems only

## Opportunities
- Future-focused only

## Detailed Analysis
- Add NEW insights only

## Conclusion
- Summarize without repeating exact phrasing

## Sources
- Unique URLs only

FINAL RULE:
If repetition exists → output is INVALID → fix before returning.
"""
),
("human",
"""Topic: {topic}

Research:
{research}
""")

])

writer_chain = writer_prompt | model | StrOutputParser()

# =========================
# CRITIC CHAIN (UPGRADED)
# =========================
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a brutally honest PhD-level research reviewer.

DO NOT be polite.

EVALUATE:

- Clarity
- Depth
- Originality (very important)
- Non-repetition
- Logical structure
- Use of sources

STRICT:

- Penalize repetition heavily
- Penalize shallow insights
- Penalize generic writing

ALSO ANSWER:

- What is missing?
- Where is reasoning weak?
- What would make this publishable?

OUTPUT FORMAT (STRICT):

Score: X/10

Strengths:
- Point 1
- Point 2

Weaknesses:
- Point 1
- Point 2

Critical Improvements:
- Point 1
- Point 2

One-line verdict:"""
    ),
    (
        "human",
        """Review the report below.

Report:
{report}

Respond EXACTLY in this format:

Score: X/10

Strengths:
- Point 1
- Point 2

Areas to Improve:
- Point 1
- Point 2

One line verdict:
<short summary>"""
    ),
])

critic_chain = critic_prompt | model | StrOutputParser()
