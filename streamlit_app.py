import streamlit as st
from agents import (
    build_question_splitter,
    build_search_agent,
    build_reader_agent,
    build_planner_agent,
    writer_chain,
    critic_chain
)

st.set_page_config(page_title="ResearchMind", layout="wide")

st.title("⚡ ResearchMind")
st.caption("Multi-Agent Research System")

topic = st.text_input("Enter Topic")
run_btn = st.button("Run Research")

if run_btn:
    if not topic.strip():
        st.warning("Enter a topic")
        st.stop()

    state = {}

    progress = st.progress(0)

    # ---------------- STEP 0 ----------------
    with st.expander("🧠 Step 0: Question Splitter", expanded=True):
        splitter = build_question_splitter()
        sub_q = splitter.invoke({"topic": topic})
        state['sub_questions'] = sub_q

        st.success("Sub-questions generated")
        st.code(sub_q)

    progress.progress(15)

    # ---------------- STEP 1 ----------------
    with st.expander("🌐 Step 1: Search Agent", expanded=True):
        search_agent = build_search_agent()

        res = search_agent.invoke({
            "messages": [(
                "user",
                f"Topic: {topic}\nSub-questions: {state['sub_questions']}\nUse multi_search"
            )]
        })

        state['search_result'] = res["messages"][-1].content

        st.success("Search completed")
        st.markdown(state['search_result'])

    progress.progress(35)

    # ---------------- STEP 2 ----------------
    with st.expander("📖 Step 2: Reader Agent", expanded=False):
        reader = build_reader_agent()

        res = reader.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze:\n{state['search_result'][:1000]}"
            }]
        })

        state['scraped_content'] = res["messages"][-1].content

        st.success("Content extracted")
        st.markdown(state['scraped_content'])

    progress.progress(55)

    # ---------------- STEP 3 ----------------
    with st.expander("🧠 Step 3: Planner Agent", expanded=False):
        planner = build_planner_agent()

        followup = planner.invoke({
            "topic": topic,
            "research": state['scraped_content'][:1200]
        })

        state['followup_query'] = followup

        st.success("Gap identified")
        st.info(followup)

    progress.progress(70)

    # ---------------- STEP 4 ----------------
    with st.expander("✍️ Step 4: Writer Agent", expanded=True):

        research = f"""
        {state['search_result']}
        {state['scraped_content']}
        """

        report = writer_chain.invoke({
            "topic": topic,
            "research": research
        })

        state['report'] = report

        st.success("Report generated")

        # 🔥 BEAUTIFUL RENDER
        st.markdown(report)

    progress.progress(90)

    # ---------------- STEP 5 ----------------
    with st.expander("🧪 Step 5: Critic Agent", expanded=False):
        feedback = critic_chain.invoke({
            "report": state['report']
        })

        state['feedback'] = feedback

        st.success("Review completed")
        st.markdown(feedback)

    progress.progress(100)

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "📥 Download Report (Markdown)",
        data=state["report"],
        file_name=f"{topic.replace(' ', '_')}.md",
        mime="text/markdown"
    )

    st.success("🎉 Pipeline Completed Successfully!")