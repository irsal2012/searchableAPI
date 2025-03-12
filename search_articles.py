import os
from dotenv import load_dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime, timedelta
import mimetypes
import time
from googlesearch import search
import re
import random

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def get_file_type(url):
    """Determine the file type from URL."""
    content_type = None
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get('content-type', '').lower()
    except:
        # If request fails, try to guess from URL
        content_type = mimetypes.guess_type(url)[0]
    
    if content_type:
        if 'pdf' in content_type:
            return 'PDF'
        elif 'word' in content_type or 'msword' in content_type:
            return 'DOC'
        elif 'text' in content_type:
            return 'TEXT'
    return 'HTML'

def extract_metadata(soup, url):
    """Extract metadata from webpage."""
    metadata = {
        'title': None,
        'publication_date': None,
        'authors': None,
        'journal': None,
        'doi': None,
        'abstract': None
    }
    
    try:
        # Try to find title
        # First check meta tags
        title_meta = soup.find('meta', {'name': ['title', 'og:title', 'twitter:title', 'citation_title', 'dc.title']})
        if title_meta:
            metadata['title'] = title_meta.get('content')
        
        # If not found in meta tags, try title tag
        if not metadata['title'] and soup.title:
            metadata['title'] = soup.title.string
            
        # If still not found, try h1 tag
        if not metadata['title'] and soup.h1:
            metadata['title'] = soup.h1.get_text().strip()
            
        # Try to find publication date
        date_meta = soup.find('meta', {'name': ['date', 'citation_publication_date', 'dc.date', 'article:published_time']})
        if date_meta:
            metadata['publication_date'] = date_meta.get('content')
        
        # Try to find authors
        authors = []
        
        # Check meta tags with various author-related attributes
        author_meta = soup.find_all('meta', {'name': ['author', 'citation_author', 'dc.creator', 'article:author', 'DCSext.author']})
        for author in author_meta:
            authors.append(author.get('content'))
            
        # Check for OpenGraph author tags
        og_author = soup.find('meta', {'property': 'og:author'})
        if og_author and og_author.get('content'):
            authors.append(og_author.get('content'))
            
        # Check for schema.org author markup
        schema_authors = soup.find_all(['span', 'div', 'a'], {'itemprop': 'author'})
        for author in schema_authors:
            name_elem = author.find({'itemprop': 'name'})
            if name_elem:
                authors.append(name_elem.get_text().strip())
            else:
                authors.append(author.get_text().strip())
                
        if authors:
            # Remove duplicates while preserving order
            unique_authors = []
            for author in authors:
                if author and author not in unique_authors:
                    unique_authors.append(author)
            metadata['authors'] = '; '.join(unique_authors)
        
        # Try to find journal name
        journal_meta = soup.find('meta', {'name': ['citation_journal_title', 'dc.source']})
        if journal_meta:
            metadata['journal'] = journal_meta.get('content')
        
        # Try to find DOI
        doi_meta = soup.find('meta', {'name': ['citation_doi', 'dc.identifier']})
        if doi_meta:
            metadata['doi'] = doi_meta.get('content')
            
        # Try to find abstract
        # First check meta tags
        abstract_meta = soup.find('meta', {'name': ['description', 'og:description', 'twitter:description', 'citation_abstract', 'dc.description', 'abstract']})
        if abstract_meta:
            metadata['abstract'] = abstract_meta.get('content')
            
        # If not found in meta tags, look for common abstract elements
        if not metadata['abstract']:
            # Look for elements with abstract-related classes or IDs
            abstract_elements = soup.find_all(['div', 'p', 'section'], 
                                             {'class': ['abstract', 'article-abstract', 'summary', 'article-summary'],
                                              'id': ['abstract', 'article-abstract', 'summary']})
            if abstract_elements:
                metadata['abstract'] = abstract_elements[0].get_text().strip()
                
        # If still not found, look for abstract section with heading
        if not metadata['abstract']:
            abstract_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong'], 
                                             string=re.compile(r'abstract', re.IGNORECASE))
            for heading in abstract_headings:
                # Try to get the next sibling paragraph
                next_p = heading.find_next('p')
                if next_p:
                    metadata['abstract'] = next_p.get_text().strip()
                    break
                    
                # If no paragraph found, try parent's text
                parent = heading.parent
                if parent and parent.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Get text excluding the heading
                    heading_text = heading.get_text().strip()
                    parent_text = parent.get_text().strip()
                    if len(parent_text) > len(heading_text):
                        metadata['abstract'] = parent_text.replace(heading_text, '').strip()
                        break
            
        # If metadata not found in meta tags, try other common patterns
        if not metadata['publication_date']:
            # Look for common date patterns in the text
            date_patterns = soup.find_all(['time', 'span', 'div'], class_=['date', 'published', 'pub-date'])
            if date_patterns:
                metadata['publication_date'] = date_patterns[0].get_text().strip()
                
        if not metadata['authors']:
            # Look for author information in common patterns
            author_patterns = soup.find_all(['div', 'span', 'p', 'a'], class_=['author', 'authors', 'contributor', 'byline', 'meta-author', 'article-author'])
            if author_patterns:
                authors = [a.get_text().strip() for a in author_patterns]
                # Remove duplicates and empty strings
                authors = [a for a in authors if a and len(a) > 1]
                if authors:
                    metadata['authors'] = '; '.join(authors)
                    
            # Look for byline text
            if not metadata['authors']:
                byline = soup.find(['p', 'div', 'span'], class_=['byline', 'article-byline'])
                if byline:
                    text = byline.get_text().strip()
                    # Try to extract author names from byline text
                    if text.lower().startswith('by '):
                        metadata['authors'] = text[3:].strip()
    
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
    
    return metadata

def extract_text_from_url(url):
    """Extract text content and metadata from URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Create a session to maintain cookies
        session = requests.Session()
        
        # Some sites need an initial visit to set cookies
        session.get(url, headers=headers, timeout=5)
        
        # Then make the actual request
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # For HTML pages
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        # Try to find main content
        main_content = None
        for tag in ["article", "main", ".content", "#content", ".post", ".article"]:
            main_content = soup.select_one(tag)
            if main_content:
                break
        
        if main_content:
            text = main_content.get_text(separator='\n')
        else:
            text = soup.get_text(separator='\n')
            
        # Clean and process text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 20:  # Only keep meaningful lines
                lines.append(line)
        
        text = '\n'.join(lines)
        metadata = extract_metadata(soup, url)
        return text, metadata
    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return None, None

def generate_summary(text, url, metadata):
    """Generate summary using OpenAI's GPT model."""
    try:
        if not text:
            return "Could not access or extract content from the webpage."
            
        if len(text) > 15000:
            text = text[:15000]
        
        prompt = f"""
        URL: {url}
        
        Content: {text}
        
        Please provide a concise summary of this content, focusing specifically on:
        1. Key information about the topic
        2. Any clinical findings or research outcomes
        3. Development status or regulatory updates
        4. Technical details about methods or mechanisms (if mentioned)
        
        If the content is not relevant to the topic, indicate that clearly.
        """
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in medical and scientific literature. Provide accurate, technical summaries focusing on key findings and developments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_source_metadata(url):
    """Get source type and development phase based on URL and content."""
    source_type = "Other"
    
    if "clinicaltrials.gov" in url:
        source_type = "Clinical Trial"
    elif any(domain in url for domain in ['pubmed.ncbi.nlm.nih.gov', 'pmc.ncbi.nlm.nih.gov', 'nejm.org', 'thelancet.com', 'jamanetwork.com', 'diabetesjournals.org', 'onlinelibrary.wiley.com', 'academic.oup.com']):
        source_type = "Research Paper"
    elif any(domain in url for domain in ['fda.gov', 'ema.europa.eu', 'who.int']):
        source_type = "Regulatory Document"
    elif any(domain in url for domain in ['zealandpharma.com', 'novonordisk.com', 'lilly.com', 'sanofi.com']):
        source_type = "Company Document"
    elif any(domain in url for domain in ['biospace.com', 'fiercebiotech.com', 'medscape.com', 'globenewswire.com', 'prnewswire.com']):
        source_type = "News"
    
    return source_type

def extract_phase_info(text):
    """Extract development phase information from content."""
    phase = None
    study_type = None
    
    if text:
        # Look for phase information
        phase_patterns = [
            r"[Pp]hase (?:1|2|3|4|I|II|III|IV)",
            r"[Pp]hase-(?:1|2|3|4|I|II|III|IV)",
        ]
        
        # Look for study type
        study_patterns = {
            'RCT': r"randomized(?:\s+controlled)?(?:\s+trial)?",
            'Observational': r"observational study",
            'Meta-Analysis': r"meta-analysis",
            'Review': r"systematic review",
            'Case Study': r"case study|case report",
            'Preclinical': r"preclinical|in vitro|in vivo"
        }
        
        # Find phase
        for pattern in phase_patterns:
            match = re.search(pattern, text)
            if match:
                phase = match.group(0)
                break
        
        # Find study type
        for study_type_name, pattern in study_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                study_type = study_type_name
                break
    
    return phase, study_type

def get_search_domains():
    """Get list of medical and academic domains to search."""
    return {
        'Research Paper': [
            'pubmed.ncbi.nlm.nih.gov',
            'pmc.ncbi.nlm.nih.gov',
            'nejm.org',
            'thelancet.com',
            'jamanetwork.com',
            'diabetesjournals.org',
            'onlinelibrary.wiley.com',
            'academic.oup.com',
            'sciencedirect.com'
        ],
        'Clinical Trial': [
            'clinicaltrials.gov',
            'clinicaltrialsregister.eu'
        ],
        'Regulatory Document': [
            'fda.gov',
            'ema.europa.eu',
            'who.int'
        ],
        'Company Document': [
            'zealandpharma.com',
            'novonordisk.com',
            'lilly.com',
            'sanofi.com'
        ],
        'News': [
            'biospace.com',
            'fiercebiotech.com',
            'medscape.com',
            'globenewswire.com',
            'prnewswire.com'
        ]
    }

def get_pubmed_results(query, max_results=5):
    """Get results directly from PubMed's E-utilities."""
    try:
        # Add API key if available
        api_key = os.getenv('NCBI_API_KEY')
        
        # First get IDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'sort': 'date',
            'retmode': 'json',
            'usehistory': 'y'
        }
        if api_key:
            params['api_key'] = api_key
            
        response = requests.get(esearch_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get WebEnv and QueryKey for subsequent requests
        webenv = data['esearchresult'].get('webenv')
        query_key = data['esearchresult'].get('querykey')
        ids = data['esearchresult'].get('idlist', [])
        
        if not ids:
            return []
            
        # Then get details using summary API
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'json',
            'webenv': webenv,
            'query_key': query_key
        }
        if api_key:
            fetch_params['api_key'] = api_key
            
        response = requests.get(efetch_url, params=fetch_params)
        response.raise_for_status()
        details = response.json()
        
        # Extract URLs, titles, and authors
        results = []
        for pmid in ids:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            # Try to get title and authors from the summary results
            title = None
            authors = None
            if 'result' in details and pmid in details['result']:
                title = details['result'][pmid].get('title', '')
                
                # Extract authors if available
                if 'authors' in details['result'][pmid]:
                    author_list = details['result'][pmid]['authors']
                    author_names = []
                    for author in author_list:
                        if 'name' in author:
                            author_names.append(author['name'])
                    if author_names:
                        authors = '; '.join(author_names)
            
            results.append({
                'url': url,
                'title': title,
                'authors': authors
            })
            
        return results
    except Exception as e:
        print(f"Error fetching from PubMed: {str(e)}")
        return []

def get_clinicaltrials_results(query, max_results=5):
    """Get results directly from ClinicalTrials.gov API."""
    try:
        # First try the new API endpoint
        api_url = "https://classic.clinicaltrials.gov/api/query/study_fields"
        params = {
            'expr': query,
            'fields': 'NCTId,BriefTitle,StartDate,LeadSponsorName,ResponsiblePartyInvestigatorFullName',
            'min_rnk': 1,
            'max_rnk': max_results,
            'fmt': 'json'
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for study in data.get('StudyFieldsResponse', {}).get('StudyFields', []):
                if 'NCTId' in study and study['NCTId']:
                    nct_id = study['NCTId'][0]
                    url = f"https://classic.clinicaltrials.gov/study/{nct_id}"
                    
                    # Extract title and authors if available
                    title = None
                    authors = None
                    
                    if 'BriefTitle' in study and study['BriefTitle']:
                        title = study['BriefTitle'][0]
                    
                    # Try to get investigator name first, then sponsor as fallback
                    if 'ResponsiblePartyInvestigatorFullName' in study and study['ResponsiblePartyInvestigatorFullName']:
                        authors = study['ResponsiblePartyInvestigatorFullName'][0]
                    elif 'LeadSponsorName' in study and study['LeadSponsorName']:
                        authors = f"Sponsor: {study['LeadSponsorName'][0]}"
                    
                    results.append({
                        'url': url,
                        'title': title,
                        'authors': authors
                    })
            return results
            
        except requests.exceptions.RequestException:
            # If API fails, try web search as fallback
            search_url = "https://classic.clinicaltrials.gov/ct2/results"
            params = {
                'term': query,
                'recrs': 'abdef',  # All studies except withdrawn
                'age_v': 'all',
                'gndr': 'all',
                'type': 'all',
                'rslt': 'all',
                'Search': 'Apply'
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find study links and titles
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/show/' in href and 'NCT' in href:
                    nct_id = href.split('/')[-1]
                    url = f"https://classic.clinicaltrials.gov/study/{nct_id}"
                    
                    # Try to extract title from link text
                    title = link.get_text().strip() if link.get_text().strip() else None
                    
                    results.append({
                        'url': url,
                        'title': title
                    })
                    
                    if len(results) >= max_results:
                        break
                        
            return results
            
    except Exception as e:
        print(f"Error fetching from ClinicalTrials.gov: {str(e)}")
        return []
    except Exception as e:
        print(f"Error fetching from ClinicalTrials.gov: {str(e)}")
        return []

def search_articles(query, num_results=10, years_back=5, source_types=None, status_callback=None):
    """Search and analyze articles based on query parameters."""
    results = []
    processed_urls = set()
    
    # Calculate date for years_back
    years_ago = (datetime.now() - timedelta(days=years_back*365)).year
    
    # Get domains to search based on source types
    domains = get_search_domains()
    if source_types:
        search_domains = [domain for type_ in source_types for domain in domains.get(type_, [])]
    else:
        search_domains = [domain for domains_list in domains.values() for domain in domains_list]
    
    # Shuffle domains to randomize access pattern
    random.shuffle(search_domains)
    
    print(f"\nSearching for: {query}")
    all_urls = []
    
    # First try direct API access for major sources
    if status_callback:
        status_callback("Searching PubMed...", 20)
    pubmed_results = get_pubmed_results(query, max_results=5)
    if pubmed_results:
        print(f"Found {len(pubmed_results)} PubMed results")
        all_urls.extend(pubmed_results)
        time.sleep(3)  # Respect API rate limits
    
    if status_callback:
        status_callback("Searching ClinicalTrials.gov...", 40)
    ct_results = get_clinicaltrials_results(query, max_results=5)
    if ct_results:
        print(f"Found {len(ct_results)} Clinical Trial results")
        all_urls.extend(ct_results)
        time.sleep(3)  # Respect API rate limits
    
    # Remove duplicates from all_urls while preserving order
    # Create a set of URLs to track duplicates
    seen_urls = set()
    unique_urls = []
    for item in all_urls:
        # Handle both string URLs and dictionary items
        if isinstance(item, dict):
            url = item['url']
        else:
            url = item
            
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(item)
    
    all_urls = unique_urls
            
    if status_callback:
        status_callback("Searching journal databases...", 60)
    
    # Try direct access to journal APIs
    journal_apis = {
        'diabetesjournals.org': {
            'url': 'https://diabetesjournals.org/api/search',
            'params': {'q': query, 'size': 2}
        },
        'nejm.org': {
            'url': 'https://www.nejm.org/api/search',
            'params': {'q': query, 'page': 1, 'size': 2}
        }
    }
    
    for domain, api_info in journal_apis.items():
        if domain in search_domains:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/json'
                }
                response = requests.get(api_info['url'], params=api_info['params'], headers=headers, timeout=30)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if isinstance(data, dict) and 'results' in data:
                            for result in data['results'][:2]:  # Limit to 2 results per domain
                                if 'url' in result:
                                    all_urls.append(result['url'])
                    except:
                        pass  # Skip if JSON parsing fails
            except Exception as e:
                print(f"Error accessing {domain} API: {str(e)}")
            time.sleep(5)  # Brief delay between API calls
    
    if status_callback:
        status_callback("Searching additional sources...", 80)
    
    # For remaining domains, try direct website search
    remaining_domains = [d for d in search_domains[:3]  # Only try top 3 remaining domains
                        if not any(x in d for x in ['pubmed', 'clinicaltrials.gov']) 
                        and d not in journal_apis]
    
    for domain in remaining_domains:
        try:
            search_url = f"https://{domain}/search"
            params = {'q': query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(x in href.lower() for x in ['/article/', '/full/', '/study/']):
                        full_url = f"https://{domain}{href}" if href.startswith('/') else href
                        all_urls.append(full_url)
                        if len(all_urls) >= 2:  # Limit to 2 results per domain
                            break
        except Exception as e:
            print(f"Error searching {domain}: {str(e)}")
        
        time.sleep(5)  # Brief delay between domains
        
    # Process all collected URLs
    for item in all_urls:
            # Handle both string URLs and dictionary items
            if isinstance(item, dict):
                url = item['url']
                api_title = item.get('title')
                api_authors = item.get('authors')
            else:
                url = item
                api_title = None
                api_authors = None
                
            if url not in processed_urls:
                processed_urls.add(url)
                try:
                    print(f"\nProcessing: {url}")
                    
                    # Get file type
                    file_type = get_file_type(url)
                    print(f"File type: {file_type}")
                    
                    # Extract text and metadata
                    content, metadata = extract_text_from_url(url)
                    if not content:
                        print("Could not extract content")
                        continue
                    
                    # Generate summary
                    print("Generating summary...")
                    summary = generate_summary(content, url, metadata)
                    
                    # Extract phase and study type from content
                    phase, study_type = extract_phase_info(content)
                    
                    # Get source type
                    source_type = get_source_metadata(url)
                    
                    # Use API-provided title and authors if available and metadata is not
                    title = metadata.get('title', '')
                    if not title and api_title:
                        title = api_title
                        
                    authors = metadata.get('authors', '')
                    if not authors and api_authors:
                        authors = api_authors
                    
                    result = {
                        'URL': url,
                        'Title': title,
                        'File_Type': file_type,
                        'Content': content,  # Store the full article content
                        'Summary': summary,
                        'Abstract': metadata.get('abstract', ''),  # Store the extracted abstract
                        'Publication_Date': metadata.get('publication_date', ''),
                        'Authors': authors,
                        'Journal': metadata.get('journal', ''),
                        'DOI': metadata.get('doi', ''),
                        'Source_Type': source_type,
                        'Development_Phase': phase,
                        'Study_Type': study_type
                    }
                    results.append(result)
                    
                    # Add delay between requests
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
    
    # Create DataFrame with explicit columns
    columns = ['URL', 'Title', 'File_Type', 'Content', 'Summary', 'Abstract', 'Publication_Date', 'Authors', 'Journal', 'DOI', 'Source_Type', 'Development_Phase', 'Study_Type']
    df = pd.DataFrame(results, columns=columns)
    
    # Sort by publication date if available
    if 'Publication_Date' in df.columns and not df.empty:
        try:
            df['Publication_Date'] = pd.to_datetime(df['Publication_Date'])
        except Exception as e:
            print(f"Warning: Could not parse some publication dates: {str(e)}")
        df = df.sort_values('Publication_Date', ascending=False, na_position='last')
    
    return df

def main():
    query = "Dasiglucagon delivery systems"
    print(f"Searching for: {query}")
    
    df = search_articles(query)
    
if __name__ == "__main__":
    main()
