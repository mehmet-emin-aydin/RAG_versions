import streamlit as st
import json
import os

# Define the path to the agents.json file
AGENTS_FILE_PATH = 'agents.json'

# Load agent data from the JSON file if it exists, otherwise return an empty list
def load_agents(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return []

# Set the number of columns per row
num_columns = 3

# Initialize session state for storing selected spaces if not already set
if 'spaces' not in st.session_state:
    st.session_state.spaces = []

# Load agents from the JSON file
agents = load_agents(AGENTS_FILE_PATH)

# Helper function to handle agent selection
def select_agent(agent_index):
    selected_agent = agents[agent_index]
    st.session_state.spaces = selected_agent['selected_space_key']
    st.session_state.agent_selected = True  # Mark agent as selected
    st.query_params.page='chatbot'  # Redirect to chatbot page
    st.rerun()

# Helper function to create a Bootstrap card for each agent
def create_agent_card(agent, index):
    space_keys = ', '.join(agent['selected_space_key'])  # Join space keys into a string
    return f"""
    <div class="card" style="width: 18rem; margin: 10px; ">
      <div class="card-body">
        <h5 class="card-title" >{agent['name']}</h5>
        <p class="card-text">
            <strong>Created Date:</strong> {agent['created_date']}<br>
            <strong>Created By:</strong> {agent['created_by']}<br>
            <strong>Selected Spaces:</strong> {space_keys}
        </p>
      </div>
    </div>
    """

# Check if there are any agents to display
if not agents:
    st.warning("There are no created agents.")
else:
    # Divide agents into chunks of the specified number of columns
    rows = [agents[i:i + num_columns] for i in range(0, len(agents), num_columns)]

    # Display agents in a grid layout
    for row in rows:
        cols = st.columns(num_columns)
        for col, (index, agent) in zip(cols, enumerate(row)):
            with col:
                # Render the Bootstrap card for the agent
                st.markdown(create_agent_card(agent, index), unsafe_allow_html=True)
                
                # Streamlit button handling instead of HTML button click
                if st.button(f"Select Agent", key=f"select_agent_{index}"):
                    select_agent(index)

# Check URL query params for agent selection
query_params = st.query_params.to_dict()
if 'select' in query_params:
    agent_index = int(query_params['select'][0])
    select_agent(agent_index)
