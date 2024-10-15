import streamlit as st
import sys
import json
from pathlib import Path

# backend reposu üst dizinde oldugu icin path ini systeme eklemek gerekiyor
sys.path.insert(0, '/home/mehmet/Desktop/Softtech/RAG/RAG_versions/demo_prototip/backend')
from updater import update_given_spaces

# Define the space dictionary
space_dict = {
    "AILab": "YPZ",
    "Mimari Yetkinlik Merkezi": "MYM",
    "Sermaye Piyasaları İşlem Sonrası Sistemleri Direktörlüğü": "SPISS",
}

# Path to the agents.json file
AGENTS_FILE_PATH = Path('agents.json')

# Function to load agents from the JSON file
def load_agents():
    if AGENTS_FILE_PATH.exists():
        with open(AGENTS_FILE_PATH, 'r') as file:
            return json.load(file)
    return []

# Function to save agents to the JSON file
def save_agents(agents):
    with open(AGENTS_FILE_PATH, 'w') as file:
        json.dump(agents, file, indent=4)

# --- Agent creation section ---
col1, col2 = st.columns(2, gap="small")
with col1:
    if 'selected_spaces' not in st.session_state:
        st.session_state['selected_spaces'] = []
    if 'agent_name' not in st.session_state:
        st.session_state['agent_name'] = ''

    st.title("Space Selection")
    options = [*space_dict]
    selected_spaces = st.multiselect("Select the Spaces you want to retrieve:", options=options)
    agent_name = st.text_input("Agent Name:")

with col2:
    if 'update_spaces' not in st.session_state:
        st.session_state['update_spaces'] = []
    st.title("Update Spaces", anchor=False)
    selected_options = selected_spaces
    update_spaces = st.multiselect("Select the Spaces you want to update:", options=selected_options)

if st.button("Create Agent"):
    if not len(selected_spaces):
        st.write("You must select at least one Space.")
    elif not agent_name.strip():
        st.write("You must name the Agent.")
    else:
        st.session_state.selected_spaces = selected_spaces
        st.session_state.agent_name = agent_name
        
        # Update spaces if selected
        if len(update_spaces):
            st.session_state.update_spaces = update_spaces
            update_given_spaces([space_dict[space] for space in update_spaces])

        # Load existing agents
        agents = load_agents()
        
        # Create the new agent data
        new_agent = {
            "name": agent_name,
            "created_date": st.date_input("Date").strftime('%Y-%m-%d'),  # You can adjust the source of the date
            "created_by": "mehmet.aydin2@softtech.com.tr",  # Replace with actual creator info if available
            "selected_space_key": [space_dict[space] for space in selected_spaces]
        }
        
        # Add the new agent to the list
        agents.append(new_agent)
        
        # Save updated agents back to the file
        save_agents(agents)
        
        st.success(f"Agent '{agent_name}' has been created and saved successfully!")

# Display agent details if an agent has been created
if len(st.session_state['selected_spaces']) and st.session_state['agent_name'].strip():
    st.write(f"Agent Name: {st.session_state['agent_name']}")
    st.write("Selected Spaces:")
    for space in st.session_state['selected_spaces']:
        st.write(f"- {space_dict[space]}")
    st.write("Update Spaces:")
    st.write(st.session_state.update_spaces)
