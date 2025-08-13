import requests
import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from urllib.parse import quote
from flask import Flask, request, jsonify, Blueprint
from dotenv import load_dotenv
import google.generativeai as genai

from flask_cors import CORS





# --- Configuration and Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)


gemini_supabase_multi_new_bp = Blueprint("thematic_analysis_new", __name__)

CORS(gemini_supabase_multi_new_bp)


# --- Constants ---
NCBI_API_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Suggestion 6: Add rate limiting to respect NCBI policies (10 req/sec with key, 3 without)
# A 0.4s delay is a safe choice, allowing ~2.5 requests/second.
NCBI_REQUEST_DELAY = 0.4

# --- Helper Functions ---

def _validate_and_parse_inputs(data):
    """Suggestion 3: Validates and extracts request parameters, with defaults."""
    if not data:
        raise ValueError("Invalid JSON format or empty body.")

    mesh_term = data.get("mesh_term")
    if not mesh_term:
        raise ValueError("mesh_term is a required field.")

    try:
        years = int(data.get("years", 5))
        limit = int(data.get("limit", 200))
        author_limit = int(data.get("author_limit", 25))
        if years <= 0 or limit <= 0 or author_limit <= 0:
            raise ValueError("Parameters 'years', 'limit', and 'author_limit' must be positive integers.")
    except (ValueError, TypeError):
        raise ValueError("Parameters 'years', 'limit', and 'author_limit' must be valid integers.")

    return {
        "mesh_term": mesh_term,
        "affiliation": data.get("affiliation"),
        "years": years,
        "limit": limit,
        "author_limit": author_limit,
        "sort_by": data.get("sort_by", "date"),
        "perform_thematic_analysis": data.get("perform_thematic_analysis", False),
    }

def _build_esearch_url(params, api_key_param):
    """Builds the eSearch URL with precise date filtering."""
    search_term = f"({params['mesh_term']})"
    if params['affiliation']:
        # Allow users to provide their own field tags like [ad] or [affiliation]
        if "[ad]" in params['affiliation'].lower() or "[affiliation]" in params['affiliation'].lower():
            affiliation_query = f"({params['affiliation']})"
        else:
            affiliation_query = f"({params['affiliation']}[Affiliation])"
        search_term += f" AND {affiliation_query}"

    encoded_search_term = quote(search_term)

    # Suggestion 7: Replace imprecise 'reldate' with precise 'mindate'/'maxdate'
    if params['sort_by'] == 'relevance':
        sort_param = "&sort=relevance"
        date_param = ""
    else:
        sort_param = ""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(params['years'] * 365.25))
        date_param = f"&datetype=pdat&mindate={start_date.strftime('%Y/%m/%d')}&maxdate={end_date.strftime('%Y/%m/%d')}"

    return (
        f"{NCBI_API_BASE_URL}esearch.fcgi?"
        f"db=pubmed&term={encoded_search_term}{date_param}"
        f"&retmax={params['limit']}&retmode=json{api_key_param}{sort_param}"
    )

def _add_author_to_category(author_data_dict, name, pmid, affiliation):
    """Suggestion 5: Consolidates the logic for adding author data to a category."""
    author_data_dict[name]['pmids'].append(pmid)
    author_data_dict[name]['affiliations'].append(affiliation)

def _process_article_element(article_element, author_data, first_author_data, last_author_data, middle_author_data):
    """
    Suggestion 4 (Helper): Processes a single <PubmedArticle> element from the stream.
    Returns a dictionary of the processed article data.
    """
    pmid_tag = article_element.find(".//PMID")
    if pmid_tag is None or not pmid_tag.text:
        return None
    pmid = pmid_tag.text

    title = (article_element.find(".//ArticleTitle").text or "No Title Found").strip()
    journal = (article_element.find(".//Title").text or "No Journal Found").strip()
    pub_year = (article_element.find(".//PubDate/Year").text or "N/A").strip()

    authors_xml = article_element.findall(".//Author")
    authors_with_affiliations = []

    for i, auth_element in enumerate(authors_xml):
        last_name = (auth_element.find("LastName").text or "").strip()
        initials = (auth_element.find("Initials").text or "").strip()

        if not last_name:
            continue

        # Suggestion 1: Safely parse affiliation to prevent AttributeError
        affiliation_text = "No affiliation listed"
        affiliation_tag = auth_element.find(".//Affiliation")
        if affiliation_tag is not None and affiliation_tag.text:
            affiliation_text = affiliation_tag.text.strip()

        authors_with_affiliations.append({"name": f"{last_name}, {initials}", "affiliation": affiliation_text})

        normalized_name = f"{last_name}, {initials[0]}" if initials else last_name

        # Suggestion 5 (Usage): Use the consolidated helper function
        _add_author_to_category(author_data, normalized_name, pmid, affiliation_text)
        if i == 0:
            _add_author_to_category(first_author_data, normalized_name, pmid, affiliation_text)
        elif i == len(authors_xml) - 1 and len(authors_xml) > 1:
            _add_author_to_category(last_author_data, normalized_name, pmid, affiliation_text)
        else:
            _add_author_to_category(middle_author_data, normalized_name, pmid, affiliation_text)


    # Extract Abstract
    abstract_tags = article_element.findall(".//Abstract/AbstractText")
    abstract = "\n".join([tag.text.strip() for tag in abstract_tags if tag.text]).strip()
    if not abstract:
        abstract = "No abstract available."

    return {"pmid": pmid, "title": title, "journal": journal, "pubYear": pub_year, "authors": authors_with_affiliations, "abstract": abstract}

def _get_thematic_analysis(articles, mesh_term):
    """
    Performs thematic analysis on a list of articles using the Gemini API.
    """
    if not gemini_api_key:
        return {"error": "Gemini API key is not configured."}

    # Prepare the content for the prompt
    content_to_analyze = ""
    # Limit to first 50 articles to keep the prompt concise and focused
    for article in articles[:50]:
        content_to_analyze += f"Title: {article['title']}\nAbstract: {article['abstract']}\n\n"

    if not content_to_analyze:
        return {"themes": ["No content available for analysis."]}

    # The model's context window is large, but let's be pragmatic.
    # A hard character limit is a safeguard.
    max_content_length = 30000
    if len(content_to_analyze) > max_content_length:
        content_to_analyze = content_to_analyze[:max_content_length]
        logger.warning(f"Content for thematic analysis was truncated to {max_content_length} characters.")

    prompt = f"""
    Based on the titles and abstracts from a literature search on '{mesh_term}', what are the 5-7 main research themes?
    For each theme, provide a one-sentence description.
    Please format the output as a numbered list. For example:
    1. Theme Name: [Brief description].

    ---
    ARTICLES:
    {content_to_analyze}
    ---
    """

    try:
        logger.info("Sending request to Gemini API for thematic analysis...")
        # Using a specific model known for this kind of task.
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # A more robust way to parse the response.
        themes = [theme.strip() for theme in response.text.split('\n') if theme.strip() and theme.strip()[0].isdigit()]

        if not themes:
             # If parsing fails, return the raw text, which is still useful.
             return {"themes": [response.text]}

        return {"themes": themes}

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return {"error": f"Failed to perform thematic analysis: {str(e)}"}


def format_author_counts(author_data, limit):
    """Formats author data for the final response."""
    sorted_authors = sorted(author_data.items(), key=lambda item: len(item[1]['pmids']), reverse=True)

    formatted_list = []
    for author, data in sorted_authors[:limit]:
        affiliation_counter = Counter(data['affiliations'])
        most_common_affiliation = affiliation_counter.most_common(1)[0][0] if affiliation_counter else "No affiliation listed"

        formatted_list.append({
            "author": author,
            "affiliation": most_common_affiliation,
            "count": len(data['pmids']),
            "pmids": ",".join(data['pmids'])
        })
    return formatted_list

# --- Main API Route (Refactored) ---
@gemini_supabase_multi_new_bp.route("/thematic_analysis", methods=["POST"])
def thematic_analysis():
    """
    Suggestion 2: The main function is now a coordinator, delegating tasks to helpers.
    """
    try:
        params = _validate_and_parse_inputs(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        logger.info(f"Starting analysis for MeSH: '{params['mesh_term']}', Affiliation: '{params.get('affiliation') or 'None'}'")

        session = requests.Session()
        session.headers.update({"User-Agent": "CodingFleet-Code-Enhancer/1.0"})
        nih_api_key = os.getenv("NIH_API_KEY")
        api_key_param = f"&api_key={nih_api_key}" if nih_api_key else ""

        # --- Step 1: eSearch to find PMIDs ---
        esearch_url = _build_esearch_url(params, api_key_param)
        logger.info(f"Executing eSearch: {esearch_url}")
        search_response = session.get(esearch_url)
        time.sleep(NCBI_REQUEST_DELAY) # Suggestion 6: Rate limiting
        search_response.raise_for_status()
        search_data = search_response.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        total_found = int(search_data.get("esearchresult", {}).get("count", 0))

        if not id_list:
            return jsonify({"message": "No publications found for the given criteria."}), 200

        # --- Step 2 & 3: eFetch and Process with Streaming Parser ---
        pmids_str = ",".join(id_list)
        efetch_url = f"{NCBI_API_BASE_URL}efetch.fcgi?db=pubmed&id={pmids_str}&retmode=xml{api_key_param}"
        logger.info(f"Executing eFetch for {len(id_list)} PMIDs...")

        # Suggestion 4: Use a streaming parser for memory efficiency
        fetch_response = session.get(efetch_url, stream=True)
        time.sleep(NCBI_REQUEST_DELAY) # Suggestion 6: Rate limiting
        fetch_response.raise_for_status()
        fetch_response.encoding = 'utf-8' # Ensure correct encoding

        author_data = defaultdict(lambda: {'pmids': [], 'affiliations': []})
        first_author_data = defaultdict(lambda: {'pmids': [], 'affiliations': []})
        last_author_data = defaultdict(lambda: {'pmids': [], 'affiliations': []})
        middle_author_data = defaultdict(lambda: {'pmids': [], 'affiliations': []})
        processed_articles = []

        context = ET.iterparse(fetch_response.raw, events=("end",))
        for _, elem in context:
            if elem.tag == 'PubmedArticle':
                article_details = _process_article_element(elem, author_data, first_author_data, last_author_data, middle_author_data)
                if article_details:
                    processed_articles.append(article_details)
                elem.clear() # Clear element from memory after processing

        # --- Step 4: Perform Thematic Analysis (if requested) ---
        if params.get("perform_thematic_analysis"):
            thematic_analysis_results = _get_thematic_analysis(processed_articles, params['mesh_term'])
        else:
            thematic_analysis_results = {"message": "Thematic analysis was not requested."}

        # --- Step 5: Format the final response ---
        response_data = {
            "query": params,
            "total_found": total_found,
            "total_processed": len(processed_articles),
            "thematic_analysis": thematic_analysis_results,
            "author_counts": {
                "all_authors": format_author_counts(author_data, params['author_limit']),
                "first_authors": format_author_counts(first_author_data, params['author_limit']),
                "last_authors": format_author_counts(last_author_data, params['author_limit']),
                "middle_authors": format_author_counts(middle_author_data, params['author_limit'])
            },
            "articles": processed_articles
        }

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching PubMed data: {e}")
        return jsonify({"error": f"Failed to fetch data from NCBI: {str(e)}"}), 502
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {e}")
        return jsonify({"error": "Failed to parse the response from NCBI."}), 500
    except Exception as e:
        logger.exception("Critical error in thematic_analysis")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
