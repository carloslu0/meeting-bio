# LLMs
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Streamlit
import streamlit as st

# Twitter
import tweepy
    
# Scraping
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# YouTube
from langchain.document_loaders import YoutubeLoader
# !pip install youtube-transcript-api

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
else:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# We'll query 80 tweets because we end up filtering out a bunch
def get_original_tweets(screen_name, tweets_to_pull=80):
    print("Getting Tweets...")
    
    # Tweepy set up
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    # Holder for the tweets
    tweets = []

    # Pull the tweets
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended',
                                   exclude_replies=True).items(tweets_to_pull)
    
    # Run through tweets and remove retweets and quote tweets
    for status in tweepy_results:
        if hasattr(status, 'retweeted_status') or hasattr(status, 'quoted_status'):
            # Skip if it's a retweet or quote tweet
            continue
        else:
            tweets.append(status.full_text)

    # Convert the list of tweets into a string of tweets
    user_tweets = "\n\n".join(tweets)
    
    return user_tweets

# We'll get the latest 3 tweets
def get_latest_tweets(screen_name, tweets_to_pull=3):
    print("Getting Latest Tweets...")
    
    # Tweepy set up
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    # Holder for the tweets
    latest_tweets = []

    # Pull the tweets
    tweepy_results = tweepy.Cursor(api.user_timeline,
                                   screen_name=screen_name,
                                   tweet_mode='extended').items(tweets_to_pull)

    # Collect the last 3 tweets
    for status in tweepy_results:
        latest_tweets.append(status.full_text)

    # Convert the list of tweets into a string of tweets
    user_latest_tweets = "\n\n".join(latest_tweets)
    
    return user_latest_tweets

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

# Prompts
response = {
    'Meeting Bio' : """
        Your goal is to generate a 2 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
        On the second page, transform the LinkedIn data that you have into a list of bullet points about the person.
    """,
    'Client Data' : """
        Your goal is to generate a 1 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
    """
}

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
Do not make anything up, only use information which is in the person's context

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

# Initialize session state variables if not already set
if 'personal_info' not in st.session_state:
    st.session_state['personal_info'] = {}

# Personal Information section
if output_type == 'Personal Information':
    st.markdown("# Personal Information")

    # Input fields for personal information with session state
    personal_info_keys = [
        'name', 'email', 'linkedin_url', 'current_city', 'angellist_url', 'company_name',
        'company_linkedin_url', 'other_website_urls', 'undergraduate_school_name',
        'undergraduate_school_location', 'undergraduate_school_year',
        'undergraduate_school_area_of_study', 'undergraduate_school_linkedin',
        'highschool_name', 'highschool_location', 'highschool_year', 'youtube_videos'
    ]

    for key in personal_info_keys:
        st.session_state.personal_info[key] = st.text_input(key.capitalize().replace('_', ' '))

# Meeting Bio section
elif output_type == 'Meeting Bio':
    st.markdown("# Meeting Bio")

    person_name = st.text_input(label="Person's Name",  placeholder="Ex: Chris York", key="persons_name")
    linkedin_profile_url = st.text_input(label="LinkedIn Profile", placeholder="https://www.linkedin.com/in/chris-york-9bb05a11/", key="linkedin_url_input")
    youtube_videos = st.text_input(label="YouTube URLs (Use commas to seperate videos)",  placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk, https://www.youtube.com/watch?v=c_hO_fjmMnk", key="youtube_user_input")
    webpages = st.text_input(label="Web Page URLs (Use commas to seperate urls. Sites that require sign-in to view user data like LinkedIn, Crunchbase don't work yet. Must include https://)",  placeholder="https://chrisyork.co/", key="webpage_user_input")

# Check to see if there is an @ symbol or not on the user name
#if twitter_handle and twitter_handle[0] == "@":
#    twitter_handle = twitter_handle[1:]

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
     #   if not (twitter_handle or youtube_videos or webpages):
     #       st.warning('Please provide links to parse', icon="⚠️")
     #       st.stop()
        
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
    #    user_tweets = get_original_tweets(twitter_handle) if twitter_handle else ""
        video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
        website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else ""
        linkedin_data = get_linkedin_data(api_key=PROXYCURL_API_KEY, linkedin_url=linkedin_profile_url) if linkedin_profile_url else ""
   
    #    user_information = "\n".join([user_tweets, video_text, website_data])
      
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
        data_dict = json.loads(linkedin_data)
        col1, col2 = st.columns(2)

        st.markdown(f"##### 📋 Basic Information")

        with col1:
            st.write(f"Name")

        with col2:
            st.image(data_dict["profile_pic_url"])

        st.write(data_dict["full_name"])
        st.write(f"Email")
        st.write(data_dict["email"])
        st.write(f"Location")
        st.write(data_dict["city"] + ", " + data_dict["state"] + ", " + data_dict["country"])


        st.markdown(f"##### 🌐 Links")
        # Add the corresponding links
        st.markdown(f"###### Personal")

        st.write(f"LinkedIn")
        st.write(f"Twitter")
        st.write(f"Company")

        st.write(f"Company Site")
        st.write(f"Company LinkedIn")
        st.write(f"Pave Twitter")

        st.write(f"📖 Bio")
        st.write(f"Twitter Bio")
        st.write(data_dict["twitter_bio"])
        st.write(f"LinkedIn Bio")
        st.write(data_dict["linkedin_bio"])
        st.write(output['output_text'])

