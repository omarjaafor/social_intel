import os
import glob
from helpers import run_gemini, extract_json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from loguru import logger
import sys
import time
# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/analyze_comments_{time}.log", rotation="500 MB", level="INFO")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sa.json"

# Constants
MAX_CHARS = 7000
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 200

def read_file_content(file_path):
    """Read and return the content of a file."""
    try:
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path} ({len(content)} characters)")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def analyze_chunk(content):
    """Analyze a single chunk of content using Gemini."""
    logger.info(f"Analyzing chunk of size {len(content)} characters")
    prompt = """
    Analyze the following text and return a JSON with this structure:
    {
        "comments": [
            {
                "text": "actual comment text",
                "language": "language identification (Arabic/French/Arabic_Dialect)"
            }
        ]
    }
    
    Text to analyze:
    """ + content
    
    try:
        logger.debug("Sending request to Gemini API")
        response = run_gemini(prompt)
        result = extract_json(response)
        
        if not result or 'comments' not in result:
            logger.error("Invalid response format from API")
            return None
        
        logger.info(f"Successfully analyzed chunk, found {len(result['comments'])} comments")
        return result['comments']
    except Exception as e:
        logger.error(f"Error in API call: {e}")
        return None

def identify_topics(comments, is_chunk=False):
    """Identify top 15 topics from comments with examples."""
    logger.info(f"Identifying topics from {'chunk' if is_chunk else 'all'} comments")
    
    prompt = f"""
    Identify the top 15 topics from these comments. Topics should be precise and concern actions performed or to perform, policies, events, etc.
    For each topic, provide up to 15 representative examples at most from the comments.
    Return a JSON with this structure:
    {{
        "topics": [
            {{
                "topic": "topic name",
                "count": number_of_occurrences,
                "examples": ["example1", "example2", "example3"]
            }}
        ]
    }}

    Comments to analyze:
    {comments}
    """
    
    try:
        logger.debug("Sending topic identification request to Gemini API")
        response = run_gemini(prompt)
        result = extract_json(response)
        topics = result.get('topics', [])
        logger.info(f"Identified {len(topics)} topics")
        return topics
    except Exception as e:
        logger.error(f"Error in topic identification: {e}")
        return []

def merge_topics(all_topics):
    """Merge and consolidate topics, keeping top 15 in French with descriptions and examples."""
    logger.info(f"Merging {len(all_topics)} topics")
    prompt = f"""
    Merge these topics based on semantic meaning and return the top 15 topics in French.
    Topics are precise and should concern actions performed or to perform, policies, events, etc.
    For each merged topic, combine and select the most representative examples.
    Return a JSON with this structure:
    {{
        "merged_topics": [
            {{
                "topic": "topic in french",
                "count": total_count,
                "description": "short description in french",
                "examples": ["example1", "example2", "example3"]
            }}
        ]
    }}
    
    Topics to merge (with their examples):
    {all_topics}
    """
    
    try:
        logger.debug("Sending topic merging request to Gemini API")
        response = run_gemini(prompt)
        result = extract_json(response)
        merged = result.get('merged_topics', [])
        logger.info(f"Successfully merged topics into {len(merged)} topics")
        
        # Save topics with descriptions and examples to a file
        output_file = "topic_descriptions.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Topics, Descriptions et Examples:\n\n")
            for topic in merged:
                f.write(f"Topic: {topic['topic']}\n")
                f.write(f"Description: {topic.get('description', 'No description available')}\n")
                f.write(f"Count: {topic['count']}\n")
                f.write("Examples:\n")
                for example in topic.get('examples', []):
                    f.write(f"- {example}\n")
                f.write("-" * 50 + "\n")
        logger.info(f"Saved topic descriptions and examples to {output_file}")
        
        return merged
    except Exception as e:
        logger.error(f"Error in topic merging: {e}")
        return []

def analyze_sentiment(comments, batch_size=5):
    """Analyze sentiment of comments in Arabic and French in small batches."""
    filtered_comments = [c for c in comments if c['language'] in ['Arabic', 'French']]
    logger.info(f"Analyzing sentiment for {len(filtered_comments)} Arabic/French comments")
    
    all_sentiments = []
    
    # Process comments in batches
    for i in range(0, len(filtered_comments), batch_size):
        batch = filtered_comments[i:i + batch_size]
        logger.info(f"Processing sentiment batch {i//batch_size + 1} of {(len(filtered_comments) + batch_size - 1)//batch_size}")
        
        prompt = f"""
        Analyze the sentiment of these comments and return a JSON with this structure:
        {{
            "sentiments": [
                {{
                    "text": "comment text",
                    "language": "Arabic/French",
                    "polarity": "positive/negative/neutral",
                   
                }}
            ]
        }}
        
        Comments:
        {batch}
        """
        
        try:
            logger.debug(f"Sending sentiment analysis request to Gemini for batch of {len(batch)} comments")
            response = run_gemini(prompt)
            result = extract_json(response)
            
            if result and 'sentiments' in result:
                batch_sentiments = result['sentiments']
                all_sentiments.extend(batch_sentiments)
                logger.info(f"Successfully analyzed sentiment for batch: {len(batch_sentiments)} results")
            else:
                logger.error("Invalid response format from API for sentiment analysis")
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis batch: {e}")
            continue
            
        # Add a small delay between batches to avoid rate limiting
        time.sleep(0.5)
    
    logger.success(f"Completed sentiment analysis: {len(all_sentiments)} total results")
    return all_sentiments

def assign_topics_to_comments(comments, merged_topics, source_file, batch_size=5):
    """Assign topics to comments based on semantic matching."""
    logger.info(f"Assigning topics to {len(comments)} comments from {source_file}")
    
    all_comments = comments.copy()  # Make a copy to preserve original order
    
    # Create a list of topics with their descriptions
    topics_with_descriptions = [
        {
            "topic": topic["topic"],
            "description": topic.get("description", "No description available")
        }
        for topic in merged_topics
    ]
    
    # Process comments in batches
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        logger.info(f"Processing topic assignment batch {i//batch_size + 1} of {(len(comments) + batch_size - 1)//batch_size}")
        
        prompt = f"""
        For each comment, assign relevant topics from the list of merged topics.
        A comment can have multiple topics or no topics if none match.
        Use the topic descriptions to better understand the context and meaning of each topic.
        Return a JSON with this structure:
        {{
            "comments": [
                {{
                    "text": "comment text",
                    "source_file": "{source_file}",
                    "topics": ["topic1", "topic2"]  # or [] if no topics match
                }}
            ]
        }}
        
        Comments:
        {batch}
        
        Available Topics (with descriptions):
        {topics_with_descriptions}
        """
        
        try:
            logger.debug(f"Sending topic assignment request to Gemini for batch of {len(batch)} comments")
            response = run_gemini(prompt)
            result = extract_json(response)
            
            if result and 'comments' in result:
                assigned_comments = result['comments']
                # Update original comments with assigned topics
                for j, assigned in enumerate(assigned_comments):
                    idx = i + j
                    if idx < len(all_comments):
                        all_comments[idx]['topics'] = assigned.get('topics', [])
                        all_comments[idx]['source_file'] = source_file
                logger.info(f"Successfully assigned topics to batch: {len(assigned_comments)} comments")
            else:
                logger.error(f"Invalid response format from API for topic assignment batch {i//batch_size + 1}")
                # Assign empty topics to this batch
                for j in range(len(batch)):
                    idx = i + j
                    if idx < len(all_comments):
                        all_comments[idx]['topics'] = []
                        all_comments[idx]['source_file'] = source_file
                
        except Exception as e:
            logger.error(f"Error in topic assignment batch {i//batch_size + 1}: {e}")
            # Assign empty topics to this batch
            for j in range(len(batch)):
                idx = i + j
                if idx < len(all_comments):
                    all_comments[idx]['topics'] = []
                    all_comments[idx]['source_file'] = source_file
            continue
            
        # Add a small delay between batches to avoid rate limiting
        time.sleep(0.5)
    
    logger.success(f"Completed topic assignment: {len(all_comments)} total results")
    return all_comments

def create_visualizations(all_results):
    """Create and save visualization plots."""
    logger.info("Creating visualization plots")
    os.makedirs('figures', exist_ok=True)
    
    try:
        # Convert results to pandas DataFrame
        data = []
        for file_path, file_data in all_results.items():
            for sentiment in file_data['sentiments']:
                data.append({
                    'file': os.path.basename(file_path),
                    'language': sentiment.get('language', ''),
                    'polarity': sentiment.get('polarity', ''),
                    'topics': sentiment.get('topics', [])
                })
        
        if not data:
            logger.warning("No data available for visualization")
            return
            
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        # Create each visualization
        visualizations = [
            ('polarity_by_language.png', 'Sentiment Distribution by Language'),
            ('topics_distribution.png', 'Distribution of Topics'),
            ('overall_polarity.png', 'Overall Sentiment Distribution'),
            ('polarity_by_topic.png', 'Sentiment Distribution by Topic'),
            ('polarity_by_file.png', 'Sentiment Distribution by Source File'),
            ('language_by_file.png', 'Language Distribution by File')
        ]
        
        for filename, title in visualizations:
            try:
                logger.info(f"Creating visualization: {title}")
                plt.figure(figsize=(12, 8))
                
                if 'polarity_by_language' in filename:
                    sns.countplot(data=df, x='language', hue='polarity')
                elif 'topics_distribution' in filename:
                    topic_counts = Counter([t for topics in df['topics'] for t in topics])
                    if topic_counts:
                        plt.pie(topic_counts.values(), labels=topic_counts.keys(), autopct='%1.1f%%')
                    else:
                        logger.warning("No topics data available for pie chart")
                        continue
                elif 'overall_polarity' in filename:
                    polarity_counts = df['polarity'].value_counts()
                    if not polarity_counts.empty:
                        plt.pie(polarity_counts.values, labels=polarity_counts.index, autopct='%1.1f%%')
                    else:
                        logger.warning("No polarity data available for pie chart")
                        continue
                elif 'polarity_by_topic' in filename:
                    topic_sentiment = []
                    for _, row in df.iterrows():
                        for topic in row['topics']:
                            topic_sentiment.append({'topic': topic, 'polarity': row['polarity']})
                    if topic_sentiment:
                        topic_df = pd.DataFrame(topic_sentiment)
                        sns.countplot(data=topic_df, x='topic', hue='polarity')
                        plt.xticks(rotation=45, ha='right')
                    else:
                        logger.warning("No topic sentiment data available")
                        continue
                elif 'polarity_by_file' in filename:
                    sns.countplot(data=df, x='file', hue='polarity')
                    plt.xticks(rotation=45, ha='right')
                else:  # language_by_file
                    sns.countplot(data=df, x='file', hue='language')
                    plt.xticks(rotation=45, ha='right')
                
                plt.title(title)
                plt.tight_layout()
                plt.savefig(f'figures/{filename}')
                plt.close()
                logger.success(f"Saved visualization: figures/{filename}")
            except Exception as e:
                logger.error(f"Error creating visualization {filename}: {e}")
                plt.close()
                continue
            
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def write_results(results, output_file):
    """Write analysis results to the specified file."""
    logger.info(f"Writing results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Results - Generated on 2025-01-30T20:46:12+01:00\n")
            f.write("=" * 50 + "\n\n")
            
            for file_path, file_data in results.items():
                f.write(f"File: {file_path}\n")
                f.write("-" * len(f"File: {file_path}") + "\n\n")
                
                # Write topics
                f.write("Topics:\n")
                if 'topics' in file_data and file_data['topics']:
                    for topic in file_data['topics']:
                        if isinstance(topic, dict) and 'topic' in topic:
                            f.write(f"- {topic['topic']}: {topic.get('count', 'N/A')}\n")
                        else:
                            f.write(f"- {topic}\n")
                else:
                    f.write("No topics identified\n")
                f.write("\n")
                
                # Write comments with sentiment
                f.write("Comments:\n")
                if 'sentiments' in file_data and file_data['sentiments']:
                    for sentiment in file_data['sentiments']:
                        f.write(f"Text: {sentiment.get('text', 'N/A')}\n")
                        f.write(f"Language: {sentiment.get('language', 'N/A')}\n")
                        f.write(f"Polarity: {sentiment.get('polarity', 'N/A')}\n")
                        topics = sentiment.get('topics', [])
                        f.write(f"Topics: {', '.join(topics) if topics else 'No topics'}\n")
                        f.write(f"Source File: {sentiment.get('source_file', 'N/A')}\n")
                        f.write("---\n")
                else:
                    f.write("No comments analyzed\n")
                f.write("\n")
        logger.success(f"Successfully wrote results to {output_file}")
    except Exception as e:
        logger.error(f"Error writing results: {e}")
        logger.exception(e)

def save_topic_polarity_comments(results):
    """Save comments grouped by topic and polarity combinations."""
    logger.info("Saving comments grouped by topic and polarity")
    
    # Create directory for topic-polarity files
    output_dir = 'topic_polarity_comments'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all comments by topic and polarity
    topic_polarity_comments = {}
    
    for file_path, file_data in results.items():
        for comment in file_data['sentiments']:
            polarity = comment.get('polarity', 'unknown')
            topics = comment.get('topics', [])
            
            # If no topics, add to "no_topic" category
            if not topics:
                topics = ['no_topic']
            
            # Add comment to each topic-polarity combination
            for topic in topics:
                key = f"{topic}_{polarity}"
                if key not in topic_polarity_comments:
                    topic_polarity_comments[key] = []
                
                topic_polarity_comments[key].append({
                    'text': comment.get('text', ''),
                    'language': comment.get('language', ''),
                    'source_file': comment.get('source_file', '')
                })
    
    # Write files for each combination
    for key, comments in topic_polarity_comments.items():
        if not comments:  # Skip empty combinations
            continue
            
        # Create a valid filename
        filename = f"{key}.txt".replace('/', '_').replace('\\', '_')
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Comments for Topic-Polarity: {key}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, comment in enumerate(comments, 1):
                    f.write(f"Comment {i}:\n")
                    f.write(f"Text: {comment['text']}\n")
                    f.write(f"Language: {comment['language']}\n")
                    f.write(f"Source File: {comment['source_file']}\n")
                    f.write("-" * 30 + "\n")
            
            logger.info(f"Saved {len(comments)} comments for {key}")
        except Exception as e:
            logger.error(f"Error saving comments for {key}: {e}")
    
    logger.success(f"Saved comments for {len(topic_polarity_comments)} topic-polarity combinations")

def analyze_comments(content):
    """Analyze comments, splitting into chunks if necessary."""
    if len(content) <= MAX_CHARS:
        logger.info("Content within size limit, analyzing as single chunk")
        return analyze_chunk(content)
    
    logger.info(f"Content size ({len(content)} chars) exceeds limit, splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_text(content)
    logger.info(f"Split content into {len(chunks)} chunks")
    
    all_comments = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)}")
        chunk_comments = analyze_chunk(chunk)
        if chunk_comments:
            all_comments.extend(chunk_comments)
    
    return all_comments

def main():
    logger.info("Starting comment analysis")
    comment_files = glob.glob('comments/*')
    
    if not comment_files:
        logger.error("No comment files found in the comments directory")
        return
    
    results = {}
    all_topics = []
    
    for file_path in comment_files:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get comments with language identification
            comments = analyze_comments(content)
            
            # Get topics for this file
            file_topics = identify_topics(comments)
            all_topics.extend(file_topics)
            
            # Analyze sentiment
            sentiments = analyze_sentiment(comments)
            
            results[file_path] = {
                'topics': file_topics,
                'sentiments': sentiments
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    
    # Merge all topics
    merged_topics = merge_topics(all_topics)
    logger.info(f"Merged into {len(merged_topics)} topics")
    
    # Assign topics to comments after sentiment analysis
    for file_path, file_data in results.items():
        file_data['sentiments'] = assign_topics_to_comments(
            file_data['sentiments'], 
            merged_topics,
            os.path.basename(file_path)
        )
        file_data['topics'] = merged_topics
    
    # Create visualizations
    create_visualizations(results)
    logger.success("Created visualization plots in the 'figures' directory")
    
    # Write results to file
    write_results(results, 'prep/list.txt')
    
    # Save comments by topic and polarity
    save_topic_polarity_comments(results)
    
    logger.success("Analysis completed successfully")

if __name__ == "__main__":
    main()
