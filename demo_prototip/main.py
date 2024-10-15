import streamlit as st

# Sayfa Ayarları
agents_page = st.Page(
    page = "views/agents.py",
    title = "Previous Agents",
    icon = ":material/account_circle:",
    default = True,
)
create_agent_page = st.Page(
    page = "views/create_agent.py",
    title = "Create Agent",
    icon = ":material/account_circle:",
)
chatbot_page = st.Page(
    page = "views/chatbot.py",
    title = "Chatbot",
    icon = ":material/smart_toy:",
)

# Navigation setup [without sections]
# pg = st.navigation(pages = [create_agent_page, chatbot_page])


# Navigation setup [with sections]

pg = st.navigation(
    {
        "Home" : [agents_page],
        "PlayGround" : [create_agent_page, chatbot_page],
    }
)

#run navigator
pg.run()
