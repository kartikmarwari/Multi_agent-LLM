from langchain.agents import create_agent
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

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.2,   # slight creativity boost
    max_tokens=2000
)

# =========================
# QUESTION SPLITTER AGENT (NEW)
# =========================
# Job: take the raw topic and break it into 3-4 focused
# sub-questions so research agents have more to dig into.
splitter_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""You are a best and super research planning assistant of 2026.
  dont make all question similar 
  Use tools ONLY when needed.
DO NOT treat tool outputs as function calls.
Tool outputs are plain text, not JSON.
Your job:
- Take a broad topic
- Break it into 5-6 focused sub-questions by seeply analyze what user want to know  that revolve around the question asked from user so that user will satistfied with answer , and the main question of user can be answered
- Sub-questions should cover different angles so that we can research deeply on topic , and cover it from all angles
- Keep each sub-question short and searchable

STRICT OUTPUT FORMAT:
Return ONLY the sub-questions, one per line, separated by '|'.
No numbering, no extra text.

 
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
        model=model,
        tools=[web_search, multi_search],
        system_prompt="""You are a web research assistant.

Your job:
- The current year is 2026
- Find recent and reliable information

STRICT RULES:
- Every result MUST be UNIQUE (no repeated ideas)
- If multiple sources say the same thing → keep ONLY the best one
- Do NOT repeat similar summaries across results
- Each point must add NEW information

BEHAVIOR:
- If given multiple sub-questions → use multi_search
- If single query → use web_search
- Return 5-6 DISTINCT results per sub-question

OUTPUT:
- Include title, URL, summary
- Each summary must highlight a DIFFERENT insight

Avoid redundancy at all costs."""
    )

# =========================
# READER AGENT
# =========================
def build_reader_agent():
    return create_agent(
        model=model,
        tools=[scrape_url],
        system_prompt="""You are a research extraction agent.

STRICT RULES:
- ALWAYS call scrape_url tool
- NEVER answer without tool usage
- Use 4-5 URLs

ANTI-REPETITION RULES:
- If multiple sources repeat the same idea → merge into ONE point
- DO NOT restate the same fact in different sections
- Each bullet must contain NEW information
- Avoid paraphrasing duplicates

YOUR JOB:
- Extract only UNIQUE insights
- Combine overlapping ideas into one strong point
- Prioritize depth over repetition

OUTPUT FORMAT:

Key Facts:
- (only distinct facts)

Important Data:
- (only numbers/statistics, no repetition)

Key Arguments:
- (unique perspectives only)

No duplication. No fluff. No reworded repetition.
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
"""You are a research gap-analysis assistant.

STRICT RULES:
- Do NOT repeat any already covered topic
- Identify ONLY a truly missing angle
- Avoid suggesting anything already present in research

Your job:
- Find ONE completely new gap
- Suggest ONE focused follow-up query

Output must be concise and unique.
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
"""You are a senior research analyst.

Write a HIGH QUALITY professional report in clean MARKDOWN.

STRICT ANTI-REPETITION RULES:
- NEVER repeat the same idea across sections
- Each section must contain COMPLETELY NEW information
- If an idea is used once → DO NOT reuse it anywhere else
- Do NOT rephrase or paraphrase repeated points
- Merge duplicate insights into one strong statement
- Avoid overlap between "Insights", "Trends", and "Analysis"

STRUCTURE RULES:
- Insights = key takeaways ONLY
- Trends = what is happening NOW (2026)
- Challenges = real-world problems ONLY
- Opportunities = future potential ONLY
- Detailed Analysis = deep explanation (NO repetition of above sections)

FORMAT:

# 📊 Research Report: {topic}

## Key Insights
- Only the most important UNIQUE takeaways

## Trends (2026)
- Only current developments (no overlap with insights)

## Challenges
- Real-world issues (not repeated elsewhere)

## Opportunities
- Future possibilities (not repeated elsewhere)

## Detailed Analysis
- Deep reasoning
- Add NEW insights only (do NOT repeat above points)

## Conclusion
- Final answer (summarize without repeating exact wording)

## Sources
- Unique URLs only (no duplicates)

FINAL RULE:
If any point is repeated → REMOVE it.

DO NOT OUTPUT JSON.
USE clean markdown.
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
        """You are a strict research reviewer (PhD level). 
Be critical. Do not be polite.

Evaluate with:
- Clarity
- Depth
- Accuracy
- Structure
- Use of sources
Also answer:
- What is missing?
- Where is reasoning weak?
- What would make this publishable?

Do NOT be polite.
Be honest and critical."""
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
