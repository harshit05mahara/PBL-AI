from crewai import Crew, Agent, Task
import requests
from bs4 import BeautifulSoup
import json


# Web Scraper Agent
class ArxivScraper:
    def scrape_arxiv(self, query, max_results=5):
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return "Failed to retrieve data"

        soup = BeautifulSoup(response.text, "xml")
        papers = []

        for entry in soup.find_all("entry"):
            title = entry.title.text.strip()
            authors = ", ".join([author.text for author in entry.find_all("author")])
            summary = entry.summary.text.strip()
            link = entry.id.text

            papers.append(
                {"title": title, "authors": authors, "summary": summary, "link": link}
            )

        return papers


# Data Storage Agent
class DataSaver:
    def save_data(self, papers):
        with open("scraped_papers.json", "w") as f:
            json.dump(papers, f, indent=4)
        return "Data saved successfully"


# Define Crew AI Agents
scraper_agent = Agent(
    role="Scraper",
    goal="Extract research papers from arXiv",
    backstory="A research assistant who finds academic papers",
    memory=True,
    verbose=True,
)

storage_agent = Agent(
    role="Data Manager",
    goal="Save scraped research papers in JSON format",
    backstory="A document handler that organizes research data",
    memory=True,
    verbose=True,
)

# Task Defintion
scraper_task = Task(
    description="Scrape arXiv for research papers on Machine Learning.",
    agent=scraper_agent,
    expected_output="A list of research papers with title, authors, summary, and link.",
)

storage_task = Task(
    description="Save the extracted research papers in JSON format.",
    agent=storage_agent,
    expected_output="A JSON file containing research paper details.",
)

# Create Crew
crew = Crew(agents=[scraper_agent, storage_agent], tasks=[scraper_task, storage_task])

# Run the crew
papers = ArxivScraper().scrape_arxiv("machine Learning", max_results=3)
DataSaver().save_data(papers)

print("\n✅ Scraping and saving completed successfully!")
