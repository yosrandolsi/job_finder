"""
Generates a synthetic, text-based sample resume PDF for demoing the
pipeline without needing a real candidate's resume.

Not used by the app at runtime -- run this once manually:
    python sample_data/generate_sample_resume.py

Requires reportlab (see requirements-dev.txt).
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUTPUT_PATH = Path(__file__).parent / "sample_resume.pdf"

LINES = [
    "Alex Morgan",
    "Data Analyst | alex.morgan@example.com | (555) 010-2024",
    "",
    "SUMMARY",
    "Data analyst with 4 years of experience turning messy datasets into",
    "actionable dashboards and reports for product and marketing teams.",
    "",
    "SKILLS",
    "Python, SQL, Pandas, Tableau, Power BI, A/B testing, statistics,",
    "data cleaning, ETL pipelines, Excel, communication",
    "",
    "EXPERIENCE",
    "Data Analyst, Northwind Retail -- 2022 to Present",
    "  - Built weekly sales dashboards in Tableau used by 5 regional managers",
    "  - Automated a manual reporting process with Python, saving 6 hours/week",
    "",
    "Junior Data Analyst, Riverside Analytics -- 2020 to 2022",
    "  - Wrote SQL queries to support marketing campaign analysis",
    "  - Assisted in A/B test design and results reporting",
    "",
    "EDUCATION",
    "B.Sc. in Statistics, State University -- 2020",
    "",
    "CERTIFICATIONS",
    "Google Data Analytics Professional Certificate",
    "",
    "LANGUAGES",
    "English (native), Spanish (conversational)",
]


def generate() -> None:
    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=LETTER)
    _, height = LETTER
    text = c.beginText(50, height - 50)
    text.setFont("Helvetica", 11)
    for line in LINES:
        text.textLine(line)
    c.drawText(text)
    c.save()
    print(f"Sample resume written to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
