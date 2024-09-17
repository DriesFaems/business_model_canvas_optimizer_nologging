from groq import Groq
import streamlit as st
import os
import tempfile
from crewai import Crew, Agent, Task, Process
import json
import os
import requests
from crewai_tools import tool
from crewai import Crew, Process
import tomllib
from langchain_groq import ChatGroq
import pandas as pd
import datetime
from streamlit_gsheets import GSheetsConnection
import time
from pyairtable import Table


# create title for the streamlit app

st.title('Business Model Canvas Evaluator')

# create a description

st.write(f"""This application will help you in evaluating a business model canvas. For more information, contact Dries Faems at https://www.linkedin.com/in/dries-faems-0371569/""")



groq_api_key = st.text_input('Please provide your Groq API key. If you do not have a Groq API key, please go to https://console.groq.com/playground', type='password')

# create a text input for the user to input the name of the customer

value_proposition = st.text_area('What is the value proposition for your business model')
customer_pofile = st.text_area('Please provide a description of the customer segment that you are targeting')
distribution_channel = st.text_area('Please provide a description of the distribution channel that you are using')
customer_relationship = st.text_area('Please provide a description of the customer relationship that you are building')
revenue_streams = st.text_area('Please provide a description of the revenue streams that you are generating')
key_resources = st.text_area('Please provide a description of the key resources that you are using')
key_activities = st.text_area('Please provide a description of the key activities that you are performing')
key_partners = st.text_area('Please provide a description of the key partners that you are working with')
cost_structure = st.text_area('Please provide a description of the cost structure that you are facing')

initial_business_model_canvas = "Value proposition: " + value_proposition + "\n" + "Customer profile: " + customer_pofile + "\n" + "Distribution channel: " + distribution_channel + "\n" + "Customer relationship: " + customer_relationship + "\n" + "Revenue streams: " + revenue_streams + "\n" + "Key resources: " + key_resources + "\n" + "Key activities: " + key_activities + "\n" + "Key partners: " + key_partners + "\n" + "Cost structure: " + cost_structure

# create a button to start the generation of the business model canvas

if st.button('Start Business Model Evaluation'):
    os.environ["GROQ_API_KEY"] = groq_api_key
    client = Groq()
    GROQ_LLM = ChatGroq(
            # api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )


    #create crew to optimize the business model canvas

    business_model_canvas_criticizer = Agent(
        role='Critiquing the business model canvas',
        goal=f"""Critique the business model canvas to identify areas for improvement and optimization. Pay special attention to inconsistencies between different parts of the business model. Try to identify room for improvement in terms of the uniqueness of the business model canvas""",
        backstory = """You have more than 20 years of experience in evaluating business models. You are great in spotting inconsistencies in business models and identifying the weaknesses in particular components of the business model canvas.""",
        verbose = True,
        llm = GROQ_LLM,
        allow_delegation = False,
        max_iter=5,
        memory=True,
    )

    business_model_canvas_optimizer = Agent(
        role='Optimize the business model canvas',
        goal=f"""Optimize the business model canvas by giving advice on how the issues identified by the business_model_canvas_criticizer can be addressed.""",
        backstory="""You are an expert in optimizing the business model canvas. You find creative ideas to address the issues raised by the business_model_canvas_criticizer.
        Your main function is to provide advice on how to improve the business model canvas. You have a pedagogical approach to your work, and you are able to explain complex concepts in a simple way.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )
    

    # Create tasks for the agents
    critique_business_model_canvas = Task(
        description=f"""Critique the business model canvas to identify areas for improvement and optimization. You can make critical comments regarding inidividual components. In addition, critically evaluate the overall coherence and consistency of the business model canvas. Ask yourself how the components of the business model canvas could be more unique and could be used as a stepstone for building an unfair advantage. The initial business model canvas is: {initial_business_model_canvas}.""",
        expected_output='As output, provide a critical analysis of the initial business model canvas, highlighting the main issues and inconsistencies.',
        agent=business_model_canvas_criticizer
    )

    optimize_business_model_canvas = Task(
        description=f"""Provide advice on how to optimize the business model canvas to ensure that you adress the critical issues raised.""",
        expected_output='As output, provide specific advice on how the critical comments can be addressed. The focus should be advice on potential solutions for addressing the critical comments.',
        agent=business_model_canvas_optimizer
    )

    # Instantiate the second crew with a sequential process

    second_crew = Crew(
        agents=[business_model_canvas_criticizer, business_model_canvas_optimizer],
        tasks=[critique_business_model_canvas, optimize_business_model_canvas],
        verbose=2,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work
    second_results = second_crew.kickoff()

    st.markdown("**Optimizing Business Model Canvas**")
    st.write(f"""{critique_business_model_canvas.output.raw_output}""")
    st.write(f"""{optimize_business_model_canvas.output.raw_output}""")
else:
    st.write('Please click the button to start the interview')
