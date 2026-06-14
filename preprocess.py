import json
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Clean texts in the json
def clean_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")

# Function to load the raw json file and convert into a preprocessed one
def process_posts(input_path, output_path):
    with open(input_path, encoding='utf-8') as file:
        posts = json.load(file)
        preprocessed_posts =[]
        for post in posts:
            clean = clean_text(post["text"])
            metadata = extract_metadata(clean)
            post_with_metadata = post | metadata
            preprocessed_posts.append(post_with_metadata)
    
    unified_tags = get_unified_tags(preprocessed_posts)
    
    for post in preprocessed_posts:
        new_tag = {unified_tags[tag] for tag in post['tags']}
        post['tags'] = list(new_tag)

    with open(output_path, encoding='utf-8', mode='w') as outfile:
        json.dump(preprocessed_posts, outfile, indent=4)
        
    
    

# Function to create more metadata on the post using the text key
def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res


# Function to unify the tag metadata into a comprehensive one     
def get_unified_tags(preprocessed_posts):
    unique_tags = set()
    for post in preprocessed_posts:
        unique_tags.update(post['tags'])
    
    unique_tags_list= ', '.join(unique_tags) # Not actually a list
    
    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={'tags': str(unique_tags_list)})
    
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res



if __name__ == "__main__":
   input_path = "data\\raw_post.json"
   output_path = "data\\prerocessed_post.json"
   process_posts(input_path, output_path)