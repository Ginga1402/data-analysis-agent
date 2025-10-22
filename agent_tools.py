import os,io,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List,Any, Optional
import torch
from langchain.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser



from configuration import *

print(torch.cuda.is_available())
print("*"*100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
print("*"*100)


query_classification_llm = ChatOllama(base_url = BASEURL ,model= query_classification_model_name, format= "json", temperature = 0)
pandas_writing_llm = ChatOllama(model = pandas_writing_model, base_url= BASEURL, temperature=0)
reasoning_llm = ChatOllama(model = reasoning_agent_model, base_url= BASEURL, temperature=0.3)



@tool
def query_classification_agent(query: str):

    "Classifies true if query needs visualization else false"

    prompt = PromptTemplate(template = 
        """You are a query classifier. Your task is to determine if a user query is requesting a data visualization.

    IMPORTANT: Respond with ONLY 'true' or 'false' (lowercase, no quotes, no punctuation).

    Classify as 'true' ONLY if the query explicitly asks for:
    - A plot, chart, graph, visualization, or figure
    - To "show" or "display" data visually
    - To "create" or "generate" a visual representation
    - Words like: plot, chart, graph, visualize, show, display, create, generate, draw

    Classify as 'false' for:
    - Data analysis without visualization requests
    - Statistical calculations, aggregations, filtering, sorting
    - Questions about data content, counts, summaries
    - Requests for tables, dataframes, or text results

    User query: {query}""",
    input_variables = ["query"]
    )

    classification_router = prompt | query_classification_llm | JsonOutputParser()

    classification_output = classification_router.invoke({"query":query})

    # print(classification_output)

    return classification_output['classification']


@tool
def pandas_writing_agent(cols: List[str], query: str):


    """Generate pandas snippet for data query"""

    prompt = PromptTemplate(
    template = 
    f"""
    You are a Python data analyst working with pandas.

    Given a pandas DataFrame named `df` with the following columns:
    {', '.join(cols)}
    

    Your task: Write **only Python code** (using pandas) to answer this question:
    "{query}"

    ### Rules
    1. Use pandas operations on `df` only.
    2. Use **only** the listed columns.
    3. Assign your final output to a variable named `result`.
    4. Return your code inside a single markdown code fence that starts with ```python and ends with ```.
    5. Do **not** include any explanations, comments, or text outside the code block.
    6. Do **not** import or read external data (pandas is already imported as `pd`).
    7. Handle missing values appropriately using `dropna()` before aggregations or calculations.
    8. Prefer readable, concise one-liners where possible.

    ### Example
    ```python
    result = df.groupby("region")["sales"].mean().sort_values(ascending=False)
    ```

    """,
    input_variables= ["cols","query"]
    )

    pandas_query_writer = prompt | pandas_writing_llm | StrOutputParser()

    pandas_query = pandas_query_writer.invoke({"cols":cols , "query" : query})

    # print(pandas_query)

    snippet = re.search(r'```python\n(.*?)```', pandas_query, re.DOTALL)
    final_snippet = snippet.group(1).strip() if snippet else pandas_query.strip()

    # print(final_snippet)
    return final_snippet # pandas_query


@tool
def execution_tool(code: str, df: pd.DataFrame, should_plot: str):

    "Code Execution Tool"

    env = {
        "pd":pd,
        "df": df
    }

    if should_plot == "true":
        plt.rcParams["figure.dpi"] = DEFAULT_DPI
        env["plt"] = plt
        env["io"] = io

    try:

        exec(code,{},env)
        result = env.get("result",None)

        if result is None:
            # if "result" not in env:
            return "No result was assigned to 'result' variable"
            
        return result

    except Exception as e:
        return f"Error executing code: {e.args}"
    

@tool
def curate_reason_tool(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:MAX_RESULT_DISPLAY_LENGTH]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt



@tool
def reasoning_agent(query: str, result: Any):
    """Generates LLM reasoning about the result (plot or value)."""

    prompt = curate_reason_tool.invoke({"query": query, "result": result})

    print("*"*25)
    print(prompt)
    print("*"*25)

    system_prompt = "You are an insightful data analyst. Provide clear, concise explanations."
    

    reasoning_prompt = PromptTemplate(
        template= "{system_prompt}\n\n{user_prompt}"
    )


    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()


    response = reasoning_chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": prompt
    })

    if "</think>" in response:


        reasoning = response.split("</think>")[0].strip()
        answer = response.split("</think>")[-1].strip("</s>").strip()

        print("#"*25)
        print("reasoning: " + str(reasoning))
        print("\n")
        print("\n")
        print("answer: " + str(answer))
    

    else:
        reasoning = ""
        answer = response.strip("</s>").strip()

        print("#"*25)
        print("reasoning: " + str(reasoning))
        print("\n")
        print("\n")
        print("answer: " + str(answer))



    return answer

    

@tool
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt



@tool
def data_insight_agent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
    
    summary_info = DataFrameSummaryTool.invoke({"df":df})

    print("*"*25)
    print("Summary Info: ")
    print(summary_info)
    property("*"*25)
    

    # Create insight prompt
    insight_prompt = PromptTemplate(
        template="""You are a data analyst providing brief, focused insights.
        
        Analyze this dataset and provide:
        1. A brief summary (2-3 sentences)
        2. 3-4 interesting questions that could be explored with this data
        
        Dataset Information:
        {dataset_info}
        
        Keep your response concise and actionable.""",
        input_variables=["dataset_info"]
    )
    
    # Create the chain
    insight_chain = insight_prompt | reasoning_llm | StrOutputParser()
    
    try:
        response = insight_chain.invoke({"dataset_info": summary_info})

        if "</think>" in response:

            reasoning = response.split("</think>")[0].strip()
            answer = response.split("</think>")[-1].strip("</s>").strip()

            print("#"*25)
            print("reasoning: " + str(reasoning))
            print("\n")
            print("\n")
            print("answer: " + str(answer))
    

        else:
            reasoning = ""
            answer = response.strip("</s>").strip()

            print("#"*25)
            print("reasoning: " + str(reasoning))
            print("\n")
            print("\n")
            print("answer: " + str(answer))
            
            
            return answer
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"





#############################################################
## Testing queries


###########
# print(query_classification_llm.invoke("How are you"))
###########
# print(query_classification_agent.invoke(input = "plot a pie chart showing the distribution of boys and girls in a class"))

###########
# print(pandas_writing_agent.invoke({
#     "cols": ["name", "age", "salary"],
#     "query": "Find the average salary by age group"
# }))


###########


# test_df = pd.DataFrame({
#     'name': ['Alice', 'Bob', 'Charlie'],
#     'age': [25, 30, 35],
#     'salary': [50000, 60000, 70000]
# })

# test_code = "result = df['salary'].mean()"

# print(execution_tool.invoke({
#     "code": test_code,
#     "df": test_df,
#     "should_plot": "false"
# }))

############



# result1 = 75.5
# prompt1 = curate_reason_tool.invoke({
#     "query": "What's the average age?", 
#     "result": result1
# })

# print(prompt1)

# fig, ax = plt.subplots()
# ax.bar(['A', 'B', 'C'], [1, 2, 3])
# ax.set_title("Sample Chart")
# prompt3 = curate_reason_tool.invoke({
#     "query": "Create a bar chart",
#     "result": ax
# })

# print(prompt3)

#################


# test_result = 75.5

# reasoning_response = reasoning_agent.invoke({
#     "query": "What's the average age?",
#     "result": test_result
# })

# print("#"*25)
# print(reasoning_response)


##################################


# # Test example
# test_df = pd.DataFrame({
#     'name': ['Alice', 'Bob', 'Charlie'],
#     'age': [25, 30, 35],
#     'salary': [50000, 60000, 70000]
# })

# insights = data_insight_agent(test_df)
# print(insights)

