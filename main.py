# NLP and ML  
from langchain import PromptTemplate   
from langchain.chat_models import ChatOpenAI  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate   

# Streamlit   
import streamlit as st     

# APIs
# OpenAI
import openai       
# YouTube  
from langchain.document_loaders import YoutubeLoader   

# Scraping   
import requests   
from bs4 import BeautifulSoup   
from markdownify import markdownify as md    

# Environment Variables   
import json   
import requests   
import os   
from dotenv import load_dotenv  


load_dotenv()


# Get your API keys set
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
if "openai_api_key" in st.secrets:
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    openai.api_key = st.secrets["openai_api_key"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")
if "proxycurl_api_key" in st.secrets:
    PROXYCURL_API_KEY = st.secrets["proxycurl_api_key"]
else:
    PROXYCURL_API_KEY = os.getenv('PROXYCURL_API_KEY')

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

#Create GPT4 completion helper function    
def get_gpt4_response(prompt):
    gpt4_response = openai.ChatCompletion.create(  
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",  
             "content": prompt}  
        ]
    )
    return gpt4_response

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
        Your goal is to generate a 2 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
        On the second page, transform the LinkedIn data that you have into a list of bullet points about the person."""

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
Do not make anything up, only use information which is in the person's context. Limit your answer to 200-400 words.

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

    # Create input boxes for each key
    for key in st.session_state.personal_info_keys:
        st.session_state.personal_info_keys[key] = st.text_input(f"Enter your {key}", st.session_state.personal_info_keys[key])

    # Add a save button
    save_button = st.button("Save Personal Information")

    # If save button is clicked, save the data to session state
    if save_button:
        # Display confirmation message
        st.success("Personal information saved!")

        # Convert session state to JSON
        st.session_state.personal_info_json = json.dumps(st.session_state.personal_info_keys)
        st.write("PersonalInfo JSON:", st.session_state.personal_info_json)  # Debug line
        
        linkedin_url = st.session_state.personal_info_keys['linkedin_url']
        personal_linkedin_data = get_linkedin_data(api_key=PROXYCURL_API_KEY, linkedin_url=linkedin_url) if linkedin_url else ""  
        personal_linkedin_data = json.loads(personal_linkedin_data)
        st.session_state.personal_linkedin_data_json = convert_json_to_text(personal_linkedin_data)
        st.write(st.session_state.personal_linkedin_data_json)  # Display the text output


# Meeting Bio section
elif output_type == 'Meeting Bio':
    st.markdown("# Meeting Bio")

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
                    "response" : response[output_type]
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

        linkedin_response = get_gpt4_response(linkedin_summary_prompt)
        linkedin_content = linkedin_response['choices'][0]['message']['content']

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
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, "\n\n", {st.session_state.personal_linkedin_data}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any shared school/education connections. Look for details like the university name, highschool name, field of study, etc. 
                 2. If there are any shared school/education connections, provide a bullet point of each connection. Limit to 3-5 bullet points only. Do not add any paragraphs. Only the bullet points with your answers. Write 'None' if you cannot find any relevant info."""

        school_response = get_gpt4_response(school_commonalities_prompt)
        school_content = school_response['choices'][0]['message']['content']


        work_commonalities_prompt = f"""You are given two sets of data delimited by triple backticks. The first called 'personal information' provides my own personal details.
                 The second set, called 'researched person information', provides data of a person I will be meeting with.
         
                 PERSONAL INFORMATION: ```{converted_personal_info}, "\n\n", {st.session_state.personal_linkedin_data}```
                 RESEARCHED PERSON INFORMATION: ```{user_information}```
        
                 Perform the following action:
                 1. Help me to prepare for this meeting by checking if there are any shared work/company connections. Look for details like the current company, previous companies, industries, etc. 
                 2. If there are any shared work/company connections, provide a bullet point of each connection. Limit to 3-5 bullet points only. Do not add any paragraphs. Only the bullet points with your answers. Write 'None' if you cannot find any relevant info.""" 
        

        work_response = get_gpt4_response(work_commonalities_prompt)
        work_content = work_response['choices'][0]['message']['content']


        col3, col4 = st.columns(2)

        with col3:
            st.markdown(f"##### 📋 Basic Information")
            st.markdown(f"###### Name")
            st.write(data_dict.get("full_name", ""))
            st.markdown(f"###### Location")
            st.write((data_dict.get("city", "")  + ", " + data_dict.get("state", "") + ", " + data_dict.get("country", "")).strip(", "))
            st.markdown(f"###### Occupation")
            st.write(data_dict.get("occupation", ""))
            st.markdown(f"###### LinkedIn Bio")
            st.write(data_dict.get("headline", ""))
        
        with col4:
            st.image(data_dict["profile_pic_url"])


        st.markdown(f"##### 📖 Summary")
        st.write(linkedin_content if linkedin_content is not None else "")
        st.write(output['output_text'] if output.get['output_text'] is not None else "")

        st.markdown(f"##### 👥 Commonalities")
        st.markdown(f"###### Shared School Connections")
        st.write(school_content)
        st.markdown(f"###### Shared Company Connections")
        st.write(work_content)   

        # Add the corresponding links
        st.markdown(f"##### 🌐 Links")
      
        def get_value(data, default=""):
            return data.strip(", ") if data else default

        # Personal Links

        st.markdown("###### Personal Links")
        st.markdown(f"* [LinkedIn](https://linkedin.com/in/{get_value(data_dict['public_identifier'])})")
        st.markdown("* [Twitter](https://www.twitter.com)")

        # Company Links
        st.markdown("###### Company Links")
        st.markdown(f"* [{get_value(company)} LinkedIn]({get_value(company_site)})")

        # Work History
        st.markdown(f"##### 💼 Work History")
        st.markdown(f"###### Current")
        st.write(f"* {get_value(data_dict['experiences'][0]['title'])} @ {get_value(data_dict['experiences'][0]['company'])} ({get_value(data_dict['experiences'][0]['starts_at']['month'])}/{get_value(data_dict['experiences'][0]['starts_at']['day'])}/{get_value(data_dict['experiences'][0]['starts_at']['year'])}) ")
        st.markdown(f"###### Previous")
        st.write(f"* {get_value(data_dict['experiences'][1]['title'])} @ {get_value(data_dict['experiences'][1]['company'])} ({get_value(data_dict['experiences'][1]['starts_at']['month'])}/{get_value(data_dict['experiences'][1]['starts_at']['day'])}/{get_value(data_dict['experiences'][1]['starts_at']['year'])}) ")
    
        # School History
        st.markdown(f"##### 🎓 Education")
        st.write(f"* {get_value(data_dict['education'][0]['field_of_study'])} @ {get_value(data_dict['education'][0]['school'])} ({get_value(data_dict['education'][0]['starts_at']['month'])}/{get_value(data_dict['education'][0]['starts_at']['day'])}/{get_value(data_dict['education'][0]['starts_at']['year'])}) ")
        st.write(f"* {get_value(data_dict['education'][1]['field_of_study'])} @ {get_value(data_dict['education'][1]['school'])} ({get_value(data_dict['education'][1]['starts_at']['month'])}/{get_value(data_dict['education'][1]['starts_at']['day'])}/{get_value(data_dict['education'][1]['starts_at']['year'])}) ")

        

