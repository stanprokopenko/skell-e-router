from skell_e_router import ask_ai, RouterError


# MAIN EXECUTION
#---------------

if __name__ == "__main__":

    temp_system_message = """# MISSON

You are an expert assistant for $domain. Respond based on the transcript.

Visit the page url provided and extract the content about the course in a clean markdown format. Follow the template exactly and complete it with the data from the page.

Do not include any of your own additional preamble or conclusion statements in your response. Respond ONLY with the completed template.

You must ONLY use the visit_website tool to extract the content from the page. DO NOT SEARCH THE WEB.

# TEMPLATE

## Meta

Course Title:
Instructor:
Labels: (comma separated list of labels shown above the price, such as 'course', 'in progress', 'n free lessons', etc..)
Full Course Price:
Price in Parts: (comma separated part names and price in parenthesis, if course includes more than 1 part)
Sale: ("On Sale" or  "Presale" if applicable. "bundle discount" doesn't count as a sale.)
Discount: (discount percentage if on sale or on presale)
Sale Ends: (date of sale ending if applicable)
<property name>: (list out key and value of all the properties of the course such as lessons, duration, skill level, views, captions. if you see the exact phrase 'produced by proko' include it here too.): (list out key and value of all the properties of the course such as lessons, duration, skill level, views, captions. if you see the exact phrase 'produced by proko' include it here too.)

* Don't include blank items that don't have a value.

## Overview

(Full text of the Overview section. This section doesn't end untill the What You Will Learn Section begins. The overview itself is structured as markdown so it will have subsections within it.)

## What You Will Learn

(List of all the learning points with their names bolded and descriptions following a -)

## Premium Benefits

(List of all the premium benefits with their names bolded and descriptions following a -)

## Testimonials

(Full text of the What Others Are Saying section)

## FAQ

(Full text of the FAQ section)

*** Lesson list and comment sections can be ignored.
    
    """

    SYSTEM_MESSAGE = "You are funny."
    SYSTEM_MESSAGE = temp_system_message
    PROMPT = "write a joke about Proko."
    PROMPT = "https://www.proko.com/course/visual-storytelling-painting-light-103"
    MODEL = "groq-compound"

    try:
        ask_ai(
            MODEL,
            PROMPT, # accepts str OR list[dict] for conversation history
            SYSTEM_MESSAGE,
            verbosity='debug', # 'none', 'response', 'info', 'debug'
            # KWARGS
            temperature=1, 
            max_tokens=5000,
            max_completion_tokens=2048,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "text"},
            candidate_count=1,
            reasoning_effort="high",
            # compound_custom={"tools": {"enabled_tools": ["visit_website"]}},   # Used only for Groq's compound model.
            #budget_tokens=8000   # Budget conversion: <=1024=low, <=2048=medium, >2048=high
            # Use reasoning_effort because most models suport it and litellm maps it to thinking_config
        )
    except RouterError as err:
        # Pass upstream to your service / translate to your API error shape
        print({"code": err.code, "message": err.message, "details": err.details})

    print("\n--- Tests finished ---")