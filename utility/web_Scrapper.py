import pandas as pd
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup


def crawl_website(start_url, max_links=50):
    visited = set()
    queue = [start_url]
    base_domain = urlparse(start_url).netloc
    processed_links = 0  # Counter to track the number of links processed

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    results = []  # To store dicts of page data

    while queue and processed_links < max_links:
        current_url = queue.pop(0)
        if current_url in visited:
            continue

        # Skip unwanted URLs
        if any(x in current_url for x in ["linkedin", "youtube", "instagram"]) or current_url.endswith(".pdf"):
            continue

        # Ensure the URL belongs to the same domain
        if urlparse(current_url).netloc != base_domain:
            continue

        visited.add(current_url)

        try:
            response = requests.get(current_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract data
            title = soup.find("title").get_text() if soup.find("title") else "No title"
            meta_desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
            meta_desc = meta_desc["content"] if meta_desc else "No description"
            headings = {f"h{i}": [tag.get_text(strip=True) for tag in soup.find_all(f"h{i}")] for i in range(1, 7)}
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            links = [urljoin(current_url, a["href"]) for a in soup.find_all("a", href=True) if urlparse(urljoin(current_url, a["href"])).netloc == base_domain]

            # Store current URL's data
            page_data = {
                "URL": current_url,
                "Title": title,
                "Meta Description": meta_desc,
                "Headings": headings,
                "Paragraphs": paragraphs,
                "Links": links
            }
            results.append(page_data)

            # Add new links to the queue (limit adding to 25 per page)
            count_links_added = 0
            for link in soup.find_all("a", href=True):
                full_url = urljoin(current_url, link["href"])
                if (urlparse(full_url).netloc == base_domain and
                    full_url not in visited and
                    count_links_added < 25):
                    queue.append(full_url)
                    count_links_added += 1

            processed_links += 1  # Increment processed links

        except requests.RequestException:
            continue

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)
    return df
