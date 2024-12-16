import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor)
from langchain import hub
from tools.tools import get_profile_url_tavily



def lookup(name: str) -> str:
    #llm = ChatOllama(temperature= 0, model = "mistral")
    llm = ChatOllama(temperature=0, model="llama3.3")
    template = """ given the fullname {name_of_person} I want you to get it me a link to their Linkedin profile page.
                   Your answer should contain only a URL
        """
    prompt_templte = PromptTemplate(template=template, input_variables= ["name_of_person"])
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description = "Useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input = {"input": prompt_templte.format_prompt(name_of_person = name)}
    )

    linkedin_profile_url = result["output"]

    return linkedin_profile_url


if __name__ == "__main__":
    linkedin_url = lookup(name = "Charan Kumar Raghupatruni AI?ML Engineer")
    print(linkedin_url)