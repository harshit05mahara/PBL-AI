import streamlit as st
from crewai import Crew, Agent, Task
from crewai.tools import BaseTool
from crewai import BaseLLM
import requests
from bs4 import BeautifulSoup
import json
import re
from collections import Counter

# Custom Tools


class NullLLM(BaseLLM):
    def _init_(self, **kwargs):
        super()._init_(model="null-llm", **kwargs)

    def call(self, prompt, **kwargs):
        return "LLM disabled: no response."


class ScrapeArxivTool(BaseTool):
    name: str = "Scrape arXiv papers"
    description: str = "Scrapes arXiv for research papers."

    def _run(
        self,
        query="quantum computing",
        max_results=3,
        sort_by="relevance",
        sort_direction="Descending",
    ):
        base_url = "https://export.arxiv.org/api/query"
        fetch_limit = max_results * 3  # Overfetch to allow local filtering

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": fetch_limit,
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
            summary = entry.summary.text.strip()
            authors = ", ".join(
                [author.find("name").text for author in entry.find_all("author")]
            )
            link = entry.id.text
            published = entry.published.text if entry.published else "n.d."

            citation = f"{authors} ({published[:4]}). {title}. arXiv preprint arXiv:{link.split('/')[-1]}."
            keywords = self.extract_keywords(title + " " + summary)

            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "link": link,
                    "published": published,
                    "citation": citation,
                    "keywords": keywords,
                }
            )

        # Sort by publication date
        reverse = sort_direction.lower() == "descending"
        papers.sort(key=lambda x: x["published"], reverse=reverse)

        return papers[:max_results]

    def extract_keywords(self, text, top_n=5):
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
sort_order = st.selectbox("Sort papers by:", options=["Relevance", "Latest"], index=1)
sort_direction = st.selectbox(
    "Date sort direction (if 'Latest'):", ["Descending", "Ascending"], index=0
)
sort_by = "relevance"  # Always fetch by relevance

if st.button("Get Papers"):
    with st.spinner("Scraping papers..."):
        # Instantiate tools
        scraper_tool = ScrapeArxivTool()
        saver_tool = SaveDataTool()
        papers_container = {}

        # Define agent logic
        def scraper_logic(_: str = ""):
            papers = scraper_tool.run(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_direction=sort_direction
                if sort_order == "Latest"
                else "Descending",
            )
            papers_container["papers"] = papers
            return (
                f"Scraped {len(papers)} papers." if isinstance(papers, list) else papers
            )

        def saver_logic(_: str = ""):
            papers = papers_container.get("papers", [])
            if not papers:
                return "No papers to save."
            return saver_tool.run(papers)

        # Create agents
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

        # Assign tasks
        scraper_task = Task(
            description=f"Scrape {max_results} papers on '{query}' from arXiv and sort by {sort_order} ({sort_direction}).",
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

        # Run CrewAI
        crew = Crew(
            agents=[scraper_agent, storage_agent], tasks=[scraper_task, storage_task]
        )
        crew.kickoff()

        # Display results
        papers = papers_container.get("papers", [])
        if isinstance(papers, list) and papers:
            st.success(f"Found {len(papers)} papers!")
            for paper in papers:
                st.subheader(paper["title"])
                st.text(f"Authors: {paper['authors']}")
                st.text(f"Published: {paper['published']}")
                st.text(f"Summary: {paper['summary'][:500]}...")
                st.text(f"Citation: {paper['citation']}")
                st.text(f"Keywords: {', '.join(paper['keywords'])}")
                st.markdown(f"[Read More]({paper['link']})")
                st.markdown("---")
        else:
            st.error(f"No papers found. {papers if isinstance(papers, str) else ''}")
