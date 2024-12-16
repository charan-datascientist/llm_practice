from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile

import psutil
import torch
import time

def system_stats():
    """
    Prints current CPU, Memory, and GPU utilization details.
    """
    # CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")

    # Memory usage
    memory_info = psutil.virtual_memory()
    print(f"Memory Usage: {memory_info.percent}% (Total: {memory_info.total // (1024**3)} GB, Used: {memory_info.used // (1024**3)} GB, Available: {memory_info.available // (1024**3)} GB)")

    # GPU usage (if MPS is available)
    if torch.backends.mps.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()} bytes")
    else:
        print("GPU (MPS) backend is not available.")

if __name__ == "__main__":
    # load_dotenv()

    print("Hello LangChain")

    # Display system stats at the start
    print("\n=== System Utilization Before Processing ===")
    system_stats()

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm = ChatOllama(model="llama3.3")
    #llm = ChatOllama(model="mistral", device="mps")

    chain = summary_prompt_template | llm | StrOutputParser()

    # Simulate LinkedIn data scraping
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",
        mock=True
    )

    # Run the chain and time the process
    start_time = time.time()
    res = chain.invoke(input={"information": linkedin_data})
    end_time = time.time()

    # Display the result
    print("\n=== Generated Summary and Facts ===")
    print(res)

    # Display system stats after processing
    print("\n=== System Utilization After Processing ===")
    system_stats()

    # Execution time
    print(f"\nProcessing Time: {end_time - start_time:.2f} seconds")
