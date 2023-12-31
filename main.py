import json   
import os
import requests  

# LangChain 
from langchain import PromptTemplate   
from langchain.chat_models import ChatOpenAI  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate  
from langchain.document_loaders import YoutubeLoader  
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Streamlit 
import streamlit as st    
from streamlit_extras.customize_running import center_running
from annotated_text import annotated_text, annotation



import openai       
# YouTube  


# Scraping   
import requests   
from bs4 import BeautifulSoup   
from markdownify import markdownify as md    

 


# Get your API keys set
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PROXYCURL_API_KEY = st.secrets['PROXYCURL_API_KEY']
ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']



# Load up your LLM
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=.4, openai_api_key=openai_api_key, max_tokens=4000, model_name='gpt-3.5-turbo-16k')
    return llm

# A function that will be called only if the environment's openai_api_key isn't set
def get_openai_api_key():
    """Gets OpenAI API key from user input."""
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# Here we'll pull data from a website and return it's text
def pull_from_website(url):
    st.write("Getting webpages...")
    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text

# Pulling data from YouTube in text form
def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript

# Function to change our long text about a person into documents
def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=4000)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([user_information])

    return docs

#Function to scrape LI data via Proxycurl
def get_linkedin_data(api_key, linkedin_url, fallback_to_cache='on-error', use_cache='if-present',
                      skills='include', inferred_salary='include', personal_email='include',
                      personal_contact_number='include', twitter_profile_id='include',
                      facebook_profile_id='include', github_profile_id='include', extra='include'):
    st.write("Getting LinkedIn Data...")
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    header_dic = {'Authorization': 'Bearer ' + api_key}
    params = {
        'url': linkedin_url,
        'fallback_to_cache': fallback_to_cache,
        'use_cache': use_cache,
        'skills': skills,
        'inferred_salary': inferred_salary,
        'personal_email': personal_email,
        'personal_contact_number': personal_contact_number,
        'twitter_profile_id': twitter_profile_id,
        'facebook_profile_id': facebook_profile_id,
        'github_profile_id': github_profile_id,
        'extra': extra,
    }
    
    response = requests.get(api_endpoint, params=params, headers=header_dic)
    
    if response.status_code == 200:
        json_data = response.json()
        data_str = json.dumps(json_data)
        return data_str
    else:
        raise Exception(f'Request failed with status code {response.status_code}: {response.text}')

#Create GPT3.5-turbo completion helper function    
def get_gpt_response(prompt):
    gpt_response = openai.ChatCompletion.create(  
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",  
             "content": prompt}  
        ]
    )
    return gpt_response

#Create Claude completion helper function
def get_claude_response(prompt):
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=50000,
        temperature = 0.5,
        top_k = 1.0,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
    )
    return completion.completion


#Create JSON to text helper function for entire personal information section
def convert_personal_info_to_text(personal_info_keys, personal_info_json_str):
    personal_info_parts = []
    personal_info_json = json.loads(personal_info_json_str)  # Convert the string to a dictionary

    for key in personal_info_keys:
        personal_info_parts.append(f"{key} is {personal_info_json[key]}")

    return personal_info_parts

#Create JSON to text helper function for parsed linkedin data
def convert_json_to_text(json_str):
    data = json.loads(json_str)  # Parse the JSON string to a dictionary
    text_parts = []

    for key, value in data.items():
        text_parts.append(f"{key}: {value}")

    text = "\n".join(text_parts)
    return text


# Prompts
response = """
        Your goal is to generate a a concise summary that would prepare someone to talk to this person.
        Please limit your answer to 1 paragraph of 80-100 words.
        """

map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {persons_name}.
Information will include interview transcripts, and blog posts about {persons_name}
Use specifics from the research when possible. 

{response}

% START OF INFORMATION ABOUT {persons_name}:
{text}
% END OF INFORMATION ABOUT {persons_name}:

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name", "response"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given information about {persons_name}.
Do not make anything up, only use information which is in the person's context.

{response}

% PERSON CONTEXT
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name", "response"])

# Set the Streamlit page configuration
st.set_page_config(page_title="Meeting Bio Generator", page_icon=":robot:")

# Build the sidebar
add_sidebar = st.sidebar.title("Meeting Bio Generator")
output_type = st.sidebar.selectbox('Choose section', ('Personal Information', 'Meeting Bio'))

with st.sidebar:
        st.markdown("Have a meeting coming up? I bet they are on LinkedIn or YouTube or the web. This tool is meant to help you generate \
                a meeting bio based off of their data on the web, or topics they've recently talked about.\
                \n\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#) [markdownify](https://pypi.org/project/markdownify/), [LangChain](https://langchain.com/) and [OpenAI](https://openai.com). \
                \n\nForked from [@GregKamradt's](https://twitter.com/GregKamradt) repo on [LLM Interview Research Assistants](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

# Initialize session state variables
if 'personal_info_keys' not in st.session_state:
    st.session_state.personal_info_keys = {
        'name': '', 'email': '', 'linkedin_url': '', 'current_city': '', 'angellist_url': '', 'company_name': '',
        'company_linkedin_url': '', 'other_website_urls': '', 'undergraduate_school_name': '',
        'undergraduate_school_location': '', 'undergraduate_school_year': '',
        'undergraduate_school_area_of_study': '', 'undergraduate_school_linkedin': '',
        'highschool_name': '', 'highschool_location': '', 'highschool_year': '', 'youtube_videos': ''
    }

# Initialize personal_info_json in session state if not present
if 'personal_info_json' not in st.session_state:
    st.session_state.personal_info_json = ''

# Initialize personal_linkedin_json in session state if not present
if 'personal_linkedin_data_json' not in st.session_state:
    st.session_state.personal_linkedin_json = ''

# Personal Information section
if output_type == 'Personal Information':
    st.markdown("# Personal Information")
    st.markdown("### Enter your client's information here:")
    annotated_text(annotation("Hit the 'Save Personal Information' button below once you finish entering all client information", color="#8ef", border="1px dashed red"))
    
    col1, col2 = st.columns(2)

    with col1:
        for i, key in enumerate(st.session_state.personal_info_keys):
            if i < len(st.session_state.personal_info_keys) / 2:
                label = key.replace('_', ' ').title()
                st.session_state.personal_info_keys[key] = st.text_input(f"Enter your {label}", st.session_state.personal_info_keys[key])

    with col2:
        for i, key in enumerate(st.session_state.personal_info_keys):
            if i >= len(st.session_state.personal_info_keys) / 2:
                label = key.replace('_', ' ').title()
                st.session_state.personal_info_keys[key] = st.text_input(f"Enter your {label}", st.session_state.personal_info_keys[key])

    # Add a save button
    save_button = st.button("Save Personal Information")

    # If save button is clicked, save the data to session state
    if save_button:
        # Display confirmation message
        st.success("Personal information saved!")

        # Convert session state to JSON
        st.session_state.personal_info_json = json.dumps(st.session_state.personal_info_keys)
        
        
        linkedin_url = st.session_state.personal_info_keys['linkedin_url']  
        personal_linkedin_data = get_linkedin_data(api_key=PROXYCURL_API_KEY, linkedin_url=linkedin_url) if linkedin_url else ""  
        personal_linkedin_data_json = json.dumps(personal_linkedin_data)
        personal_linkedin_data = json.loads(personal_linkedin_data_json)
        st.session_state.personal_linkedin_data_json = convert_json_to_text(personal_linkedin_data)
        
        # Show success message instead of "Getting LinkedIn Data"
        center_running()
        st.success("Successfully extracted LinkedIn data!") 



# Meeting Bio section
elif output_type == 'Meeting Bio':
    st.markdown("# Meeting Bio")
    st.markdown("### Enter the details of the person you are preparing a bio for:")
    
    col1, col2 = st.columns(2)

    with col1:
        person_name = st.text_input(label="Person's Name",  placeholder="Ex: Chris York", key="persons_name")

    with col2:    
        linkedin_profile_url = st.text_input(label="LinkedIn Profile", placeholder="https://www.linkedin.com/in/chris-york-9bb05a11/", key="linkedin_url_input")

    youtube_videos = st.text_input(label="YouTube URLs (Use commas to seperate videos)",  placeholder="E.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ", key="youtube_user_input")
    webpages = st.text_input(label="Web Page URLs (Use commas to seperate urls. Won't work with sites that require logins. Must include https://)",  placeholder="https://chrisyork.co/", key="webpage_user_input")

# Output
    st.markdown(f"### {output_type}:")

# Get URLs from a string
    def parse_urls(urls_string):
        """Split the string by comma and strip leading/trailing whitespaces from each URL."""
        return [url.strip() for url in urls_string.split(',')]

# Get information from those URLs
    def get_content_from_urls(urls, content_extractor):
        """Get contents from multiple URLs using the provided content extractor function."""
        contents = []
        for url in urls:
            content = content_extractor(url)
            if content is not None:
                contents.append(content)
        return "\n".join(contents)


    button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate output based on information")

# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
    if button_ind:    
        if not (linkedin_profile_url or youtube_videos or webpages):
            st.warning('Please provide links to parse', icon="⚠️")
            st.stop()

        if not OPENAI_API_KEY:
            st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
            st.stop()

        if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
            # If the openai key isn't set in the env, put a text box out there
            OPENAI_API_KEY = get_openai_api_key()

    # Go get your data
        video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
        website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else ""
        linkedin_data = get_linkedin_data(api_key=PROXYCURL_API_KEY, linkedin_url=linkedin_profile_url) if linkedin_profile_url else ""
         
        user_information = "\n".join([linkedin_data, video_text, website_data])

        user_information_docs = split_text(user_information)

    # Calls the function above
        llm = load_LLM(openai_api_key=OPENAI_API_KEY)

        chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 # verbose=True
                                 )
    
        st.write("Sending to LLM...")

    # Here we will pass our user information we gathered, the persons name and the response type from the radio button
        output = chain({"input_documents": user_information_docs,
                    "persons_name": person_name,
                    "response" : response
                    })

        st.markdown(f"### Output:")

    # Extract and transform a few of the relevant data points    
        data_dict = json.loads(linkedin_data)
        company = data_dict["experiences"][0]["company"]
        company_site = data_dict["experiences"][0]["company_linkedin_profile_url"]

    # Summarize the 'About' section further and turn it into bullet points.
        linkedin_summary_prompt = f"""You are provided with a long description of a person, delimited by triple backticks.
                Extract the most important details about the person and turn this into 3-5 bullet points. Do not add any paragraphs. Only the bullet points with your answers.

                Description: ```{data_dict['summary']}```"""

        linkedin_response = get_claude_response(linkedin_summary_prompt)

    # Initialize the session state if not set
        if 'personal_info_keys' not in st.session_state:
            st.session_state.personal_info_keys = [
                'name', 'email', 'linkedin_url', 'current_city', 'angellist_url', 'company_name',
                'company_linkedin_url', 'other_website_urls', 'undergraduate_school_name',
                'undergraduate_school_location', 'undergraduate_school_year',
                'undergraduate_school_area_of_study', 'undergraduate_school_linkedin',
                'highschool_name', 'highschool_location', 'highschool_year', 'youtube_videos'
            ]

    # Convert Personal Info JSON to text in order to use it in the prompt
        converted_personal_info = convert_personal_info_to_text(st.session_state.personal_info_keys, st.session_state.personal_info_json)


# Output the text
    

    # OpenAI prompt to extract shared background
        school_commonalities_prompt = f"""You are given two sets of data delimited by triple backticks. The first called 'personal information' provides my own personal details.
                 The second set, called 'researched person information', provides data of a person I will be meeting with.
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, {st.session_state.personal_linkedin_data_json}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any shared school/education connections. Look for details like if both of us have went to the same university, highschool, etc. Also check if we have the same field of study and anything relevant to our educational backgrounds.
                 2. If there are any shared school/education connections, provide a bullet point summary describing each connection. Only add bullet points with your answers. Write only the word 'None' if you cannot find any relevant info.
                 3. Follow ONLY the format of the sample response below. Limit to 3-5 bullet points only. Do not add anything else to your response:

                 SAMPLE RESPONSE:
                 * You both studied in Stanford
                 * You both had highschool in New York
                 * You both have Chemical Engineering degrees
                 
                RESPONSE:"""
    

        school_response = get_claude_response(school_commonalities_prompt)


        work_commonalities_prompt = f"""You are given two sets of data delimited by triple backticks. The first called 'personal information' provides my own personal details.
                 The second set, called 'researched person information', provides data of a person I will be meeting with.
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, "\n\n", {st.session_state.personal_linkedin_data_json}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any shared work/company connections. Look for details like the current company, previous companies, industries, etc. 
                 2. If there are any shared work/company connections, provide a bullet point summary describing each connection. Only add bullet points with your answers. Write 'None' if you cannot find any relevant info.
                 3. Follow ONLY the format of the sample response below. Limit to 3-5 bullet points only. Do not add anything else to your response:

                 SAMPLE RESPONSE:
                 * You both worked at Google
                 * You both have experience as a junior developer
                 * You both worked in the Health industry for more than 5 years
                 
                RESPONSE:"""

        work_response = get_claude_response(work_commonalities_prompt)

        investment_commonalities_prompt = f"""You are given two sets of data delimited by triple backticks. The first called 'personal information' provides my own personal details.
                 The second set, called 'researched person information', provides data of a person I will be meeting with.
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, "\n\n", {st.session_state.personal_linkedin_data_json}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any commonalities between me and the researched person's investments/advising gigs. Do not include work or school similarities e.g. going to the same schools, working in the same companies, etc. Focus on our investments, as well as instances where we advised startups/early-stage companies.
                 2. Provide a bullet point describing each connection. Only add bullet points with your answers. Write 'None' if you cannot find any relevant info.
                 3. Follow ONLY the format of the sample response below. Limit to 3-5 bullet points only Do not add anything else to your response:

                 SAMPLE RESPONSE:
                 * You both invested in MasterClass
                 * You both prefer to invest in pre-seed/seed stage startups
                 * You have both invested in the FinTech industry

                RESPONSE:"""


        investment_response = get_claude_response(investment_commonalities_prompt)

        other_commonalities_prompt = f"""You are given two sets of data delimited by triple backticks. The first called 'personal information' provides my own personal details.
                 The second set, called 'researched person information', provides data of a person I will be meeting with.
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, "\n\n", {st.session_state.personal_linkedin_data_json}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any commonalities between me and the researched person. Do not include work or school similarities e.g. going to the same schools, working in the same companies, etc. Focus on other non work/education similarities, as well as non-obvious similarities that we may have. An example of this is if we share similar interests, or if we have both visited a specific country before.
                 2. If there are any other non work/school commonalities, provide a bullet point describing each connection. Only add bullet points with your answers. Write 'None' if you cannot find any relevant info.
                 3. Follow ONLY the format of the sample response below. Limit to 3-5 bullet points only. Do not add anything else to your response:

                 SAMPLE RESPONSE:
                 * You both are interested in AI and Automations
                 * You both love to travel to Europe
                 * You have both hiked Mt. Everest

                RESPONSE:"""

    
        other_response = get_claude_response(other_commonalities_prompt)
    

        col3, col4 = st.columns(2)
    

        with col3:
            st.markdown(f"##### 📋 Basic Information")
            st.markdown(f"###### Name")
            st.write(data_dict.get("full_name", ""))
            st.markdown(f"###### Location")
            st.write((str(data_dict.get("city", ""))  + ", " + str(data_dict.get("state", "")) + ", " + str(data_dict.get("country", ""))).strip(", "))
            st.markdown(f"###### Occupation")
            st.write(data_dict.get("occupation", ""))
            st.markdown(f"###### LinkedIn Bio")
            st.write(data_dict.get("headline", ""))
        
        with col4:
            st.image(data_dict["profile_pic_url"])


        st.markdown(f"##### 📖 Summary")
        st.write(linkedin_response if linkedin_response is not None else "")
        st.write(output['output_text'])

        st.markdown(f"##### 👥 Commonalities")
        st.markdown(f"###### Shared School Connections")
        st.write(school_response)
        st.markdown(f"###### Shared Work Connections")
        st.write(work_response)
        st.markdown(f"###### Similar Investments")
        st.write(investment_response)
        st.markdown(f"###### Other Commonalities")
        st.write(other_response)

        # Add the corresponding links
        st.markdown(f"##### 🌐 Links")
      
        def get_value(data, default=""):
            if data is None:
                return default
            elif isinstance(data, int):
                return str(data)
            else:
                return str(data).strip(", ")

        # Personal Links

        st.markdown(f"###### Personal Links")
        st.markdown(f"* [LinkedIn](https://linkedin.com/in/{get_value(data_dict['public_identifier'], ' ')})")
        st.markdown(f"* [Twitter](https://www.twitter.com/)")

        # Company Links
        st.markdown("###### Company Links")
        st.markdown(f"* [{get_value(company, ' ')} LinkedIn]({get_value(company_site, 'https://www.linkedin.com/')})")

        # Work History
        st.markdown(f"##### 💼 Work History")
        st.markdown(f"###### Current")
        try:
            st.write(f"* {get_value(data_dict['experiences'][0]['title'], ' ')} @ {get_value(data_dict['experiences'][0]['company'], ' ')} ({get_value(data_dict['experiences'][0]['starts_at']['month'], ' ')}/{get_value(data_dict['experiences'][0]['starts_at']['day'], ' ')}/{get_value(data_dict['experiences'][0]['starts_at']['year'], ' ')}) ")
        except IndexError:
             st.write("No current work experience")
        st.markdown(f"###### Previous")
        try:
            st.write(f"* {get_value(data_dict['experiences'][1]['title'], ' ')} @ {get_value(data_dict['experiences'][1]['company'], ' ')} ({get_value(data_dict['experiences'][1]['starts_at']['month'], ' ')}/{get_value(data_dict['experiences'][1]['starts_at']['day'], ' ')}/{get_value(data_dict['experiences'][1]['starts_at']['year'], ' ')}) ")
        except IndexError:
            st.write("No previous work experience")
   
        # School History
        st.markdown(f"##### 🎓 Education")
        try:
            st.write(f"* {get_value(data_dict['education'][0]['degree_name'], ' ')}, {get_value(data_dict['education'][0]['field_of_study'], ' ')} @ {get_value(data_dict['education'][0]['school'], ' ')} ({get_value(data_dict['education'][0]['starts_at']['month'], ' ')}/{get_value(data_dict['education'][0]['starts_at']['day'], ' ')}/{get_value(data_dict['education'][0]['starts_at']['year'], ' ')}) ")
        except IndexError:
            st.write("No educational background provided")
