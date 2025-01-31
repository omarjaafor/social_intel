import json
import re
import traceback
from loguru import logger
import google.generativeai as genai
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/omar/git/llm_4_quality/sa.json"
def extract_json(text, attempt=1):
    """Extract and validate JSON from a text string.

    Args:
        text (str): The text string containing JSON data.
        attempt (int): Number of attempts to fix and extract JSON.

    Returns:
        dict or str: The extracted JSON data if valid, otherwise attempts to correct and extract it.
    """
    logger.info(text)
    try:
        if len(text.strip()) == 0:
            return {}
        
        if attempt <= 0:
            logger.error("Maximum attempts reached. Returning raw text.")
            return {}
        text=text.replace("```json","").replace("```","").strip()

        if text.strip()[0]!="{":
            text="{"+text+"}"
        logger.info(text)
        try:
            return json.loads(text)
        except Exception as e:

            text=text.replace("['",'["')
            text=text.replace("']",'"]')
            text=text.replace("',",'",')
            text=text.replace(",'",',"')
            # Find the starting index of the JSON part
            start_index = text.find("{")
            # Find the ending index of the JSON part
            end_index = text.rfind("}") + 1

            if start_index == -1 or end_index == -1:
                logger.error("No JSON-like structure found in the text : \n" + text)
                #raise ValueError("Maximum attempts reached. Unable to extract valid JSON.")
                return {}
                
            # Extract the JSON part
            json_str = text[start_index:end_index]

            # Parse the JSON to ensure it is valid
            try:
                return json.loads(json_str)
            except Exception as e:
                #logger.error(f"JSON decoding failed on attempt {attempt}. Error: {e}")
                try:
                    # Attempt to fix and parse the JSON
                    fixed_json_str = "{" + json_str + "}"
                    return json.loads(fixed_json_str)
                except Exception as e:
                    #logger.error(
                    #    f"Fixed JSON decoding failed on attempt {attempt-1}. Error: {e}"
                    #)
                    # Recursively attempt to correct the JSON with a different method
                    text=re.sub(r"[^\x20-\x7E]", "", text.strip().replace("\n","\\n").replace('"',' '))
                    new_text = run_gemini(
                        "transform the text below into a valid json. Remove json special characters. Return nothing but one valid json.\n\n"
                        + text
                        + "\n\njson:"
                    )
                    
                    return extract_json(new_text, attempt - 1)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error extracting JSON: {str(e)}")
        return {}


def run_gemini(prompt: str) -> str:
    """
    Generate text using the Gemini model.
    
    Args:
        prompt: The prompt text to generate a response for
        
    Returns:
        str: The generated response or an empty string if an error occurs
    """
    try:
        if prompt == -1:
            return ""
            
        # Initialize model with fixed parameters
        model = "gemini-1.5-flash-002"
        temp = 0.7
        top_p = 0.8
        top_k = 40
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Initialize model
                model_instance = genai.GenerativeModel(model)

                # Generate content with specified parameters
                run_ = model_instance.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temp, top_p=top_p, top_k=top_k
                    ),
                )

                response = run_.text
                
                # Clean up potential model artifacts
                return response.replace("0" * 248, "")

            except Exception as e:
                # Log error
                traceback.print_exc()
                logger.error(f"Generation attempt {attempt + 1} failed. Error: {e}")
                try:
                    logger.info(run_.candidates[0].finish_reason)
                    logger.info(run_.candidates[0].safety_ratings)
                except:
                    pass
                    
                # On last attempt, raise the error
                if attempt == max_retries - 1:
                    raise ValueError(f"Maximum attempts ({max_retries}) reached. Unable to run Gemini: {e}")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error running Gemini: {str(e)}")
        raise
