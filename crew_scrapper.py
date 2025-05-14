import streamlit as st
from crewai import Crew, Agent, Task
from crewai.tools import BaseTool
import requests
from bs4 import BeautifulSoup
import json
from crewai import BaseLLM
import re
from collections import Counter

# Custom Tools


class NullLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(model="null-llm", **kwargs)

    def call(self, prompt, **kwargs):
        return "LLM disabled: no response."


class ScrapeArxivTool(BaseTool):
    name: str = "Scrape arXiv papers"
    description: str = "Scrapes arXiv for research papers."

    def _run(self, query="quantum computing", max_results=3):
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Error fetching papers: {e}"

        soup = BeautifulSoup(response.text, "xml")
        papers = []

        for entry in soup.find_all("entry"):
            title = entry.title.text.strip()
            authors = ", ".join(
                [author.find("name").text for author in entry.find_all("author")]
            )
            summary = entry.summary.text.strip()
            link = entry.id.text
            published = entry.published.text[:4] if entry.published else "n.d."

            # Generate citation string
            citation = f"{authors} ({published}). {title}. arXiv preprint arXiv:{link.split('/')[-1]}."

            # Basic keyword extraction
            keywords = self.extract_keywords(title + " " + summary)

            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "link": link,
                    "citation": citation,
                    "keywords": keywords,
                }
            )

        return papers

    def extract_keywords(self, text, top_n=5):
        # Very basic tokenizer and cleaner
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        stopwords = set(
            [
                "the",
                "and",
                "for",
                "with",
                "that",
                "this",
                "from",
                "are",
                "was",
                "were",
                "have",
                "has",
                "had",
                "but",
                "not",
                "all",
                "can",
                "our",
                "their",
                "these",
                "those",
                "been",
                "also",
                "such",
                "into",
                "using",
                "use",
                "new",
                "results",
                "paper",
                "show",
                "propose",
                "based",
                "study",
                "methods",
                "approach",
                "data",
                "model",
                "models",
                "present",
            ]
        )
        filtered = [word for word in words if word not in stopwords]
        most_common = Counter(filtered).most_common(top_n)
        return [word for word, _ in most_common]


class SaveDataTool(BaseTool):
    name: str = "Save papers to JSON"
    description: str = "Saves research paper data to a local JSON file."

    def _run(self, papers):
        try:
            with open("scraped_papers.json", "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4, ensure_ascii=False)
            return "Data successfully saved to scraped_papers.json"
        except Exception as e:
            return f"Error saving data: {e}"


# Streamlit App

st.title("AI Research Assistant: arXiv Paper Scraper")

query = st.text_input("Enter research topic:", "quantum computing")
max_results = st.slider("Number of papers:", 1, 10, 3)

if st.button("Get Papers"):
    with st.spinner("Scraping papers..."):
        # Instantiate Tools
        scraper_tool = ScrapeArxivTool()
        saver_tool = SaveDataTool()

        papers_container = {}
        # define agents

        # Define Agents
        def scraper_logic(_: str = ""):
            papers = scraper_tool.run(query=query, max_results=max_results)
            papers_container["papers"] = papers
            return (
                f"Scraped {len(papers)} papers." if isinstance(papers, list) else papers
            )

        def saver_logic(_: str = ""):
            papers = papers_container.get("papers", [])
            if not papers:
                return "No papers to save."
            return saver_tool.run(papers)

        scraper_agent = Agent(
            role="Scraper",
            goal="Scrape research papers from arXiv",
            backstory="An assistant who finds cutting-edge research papers from arXiv.",
            llm=NullLLM(),
            verbose=False,
        )

        storage_agent = Agent(
            role="Data Manager",
            goal="Save scraped research papers to a file",
            backstory="An assistant who securely stores data provided by the scraper.",
            llm=NullLLM(),
            verbose=False,
        )
       # ya ya m
        # Assign functions to tasks
        scraper_task = Task(
            description=f"Scrape {max_results} papers on '{query}' from arXiv.",
            agent=scraper_agent,
            expected_output="A list of scraped papers.",
            callback=scraper_logic,
        )

        storage_task = Task(
            description="Save the papers to a JSON file.",
            agent=storage_agent,
            expected_output="Confirmation of saved data.",
            callback=saver_logic,
        )

        # Run Crew
        crew = Crew(
            agents=[scraper_agent, storage_agent], tasks=[scraper_task, storage_task]
        )
        crew.kickoff()

        # Display Results
        papers = papers_container.get("papers", [])
        if isinstance(papers, list) and papers:
            st.success(f"Found {len(papers)} papers!")
            for paper in papers:
                st.subheader(paper["title"])
                st.text(f"Authors: {paper['authors']}")
                st.text(f"Summary: {paper['summary'][:500]}...")
                st.text(f"Citation: {paper['citation']}")
                st.text(f"Keywords: {', '.join(paper['keywords'])}")
                st.markdown(f"[Read More]({paper['link']})")
                st.markdown("---")
        else:
            st.error(f"No papers found. {papers if isinstance(papers, str) else ''}")
