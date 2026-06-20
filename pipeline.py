from agents import (
    build_question_splitter,
    build_search_agent,
    build_reader_agent,
    build_planner_agent,
    writer_chain,
    critic_chain
)


def run_pipeline(topic: str) -> str:
    state = {}

    # step 0 question splitter agent working
    print("\n" + "=" * 50)
    print("step 0 question splitter agent is working")
    print("_____________________________")

    splitter = build_question_splitter()
    sub_questions = splitter.invoke({"topic": topic})
    state['sub_questions'] = sub_questions
    print("\n sub questions generated:", state['sub_questions'])

    # step 1 search agent working
    print("\n" + "=" * 50)
    print("Step 1: Search Agent is running")
    print("_____________________________")

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {
            "messages": [(
                "user",
                f"Topic: {topic}\n\nSub-questions to research:\n{state['sub_questions']}\n\n"
                f"Use multi_search tool with these sub-questions separated by '|' to find recent and reliable information."
            )]
        }
    )
    state['search_result'] = search_result["messages"][-1].content
    print("\n search results", state['search_result'])

    # reader agent
    print("\n" + "=" * 50)
    print("Step 2: Reader Agent is running")
    print("_____________________________")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""Based on the following search results about "{topic}",

            Pick the most relevant URL and scrape it for deeper content.

            Search Results:
        {state['search_result'][:800]}
            """
            }
        ]
    })

    state['scraped_content'] = reader_result["messages"][-1].content
    print("\nscraped content", state['scraped_content'])

    # step 2.5 planner agent checks for gaps and does one more focused search
    print("\n" + "=" * 50)
    print("Step 3: Planner Agent is analyzing research gaps")
    print("_____________________________")

    research_so_far = (
        f"Search result:\n {state['search_result']}\n\n"
        f"Scraped content:\n {state['scraped_content']}"
    )

    planner = build_planner_agent()
    followup_query = planner.invoke({
        "topic": topic,
        "research": research_so_far[:1500]
    })
    state['followup_query'] = followup_query
    print("\n follow-up query suggested:", state['followup_query'])

    followup_result = search_agent.invoke(
        {
            "messages": [("user", f"find recent and reliable information on: {state['followup_query']}")]
        }
    )
    state['followup_result'] = followup_result["messages"][-1].content
    print("\n follow-up search result:", state['followup_result'])

    # step 3 writer chian
    print("\n" + "=" * 50)
    print("Step 4: Writer Agent is generating the report")
    print("_____________________________")

    research_combined = (
        f"Sub-questions:\n {state['sub_questions']}\n\n"
        f"Search result:\n {state['search_result']}]\n\n"
        f"detailed scraped result:\n {state['scraped_content']}]\n\n"
        f"follow-up research:\n {state['followup_result']}]"
    )

    state['report'] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    print("\nfinal report\n", state['report'])

    # step 4 writer chian
    print("\n" + "=" * 50)
    print("Step 5: Critic Agent is reviewing the report")
    print("_____________________________")

    state['feedback'] = critic_chain.invoke({
        "report": state['report']
    })
    print("\n critic report\n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("enter a topic \n")
    print(run_pipeline(topic))