"""Universe definitions for the Frontline Shift research.

Everything that classifies a job, an industry or a stock as "frontline" vs
"knowledge / AI-exposed" lives here, so the classification is decided once,
a priori, and never tuned on returns.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

AI_ERA_START = "2022-11-30"   # public release of ChatGPT

# ---------------------------------------------------------------------------
# 1. Indeed job postings by occupation (FRED, daily, index Feb-1-2020 = 100)
#    Frontline = in-person, physical, hands-on work that LLMs cannot perform.
#    Knowledge = screen-based work where LLMs/agents substitute for tasks.
# ---------------------------------------------------------------------------
INDEED_FRONTLINE = {
    "CONS": "Construction", "CLSA": "Cleaning & Sanitation", "DRIV": "Driving",
    "FOPRSE": "Food Prep & Service", "LOST": "Loading & Stocking",
    "LOGISUPP": "Logistic Support", "NURS": "Nursing", "DENT": "Dental",
    "THER": "Therapy", "MEDITECH": "Medical Technician", "VETE": "Veterinary",
    "PRMA": "Production & Manufacturing", "SEPUSA": "Security & Public Safety",
    "HOTO": "Hospitality & Tourism", "RETA": "Retail",
}
INDEED_KNOWLEDGE = {
    "SOFTDEVE": "Software Development", "ACCO": "Accounting",
    "BAFI": "Banking & Finance", "MARK": "Marketing",
    "MECO": "Media & Communications", "INDEDO": "Info Design & Documentation",
    "HUMARESO": "Human Resources", "LEGA": "Legal", "ITOPHE": "IT Ops & Helpdesk",
    "MATH": "Mathematics", "CUSTSERV": "Customer Service", "ARCH": "Architecture",
    "PROJMANA": "Project Management", "SCREDE": "Scientific R&D",
    "MEDIINFO": "Medical Information", "MANA": "Management", "AREN": "Arts & Entertainment",
}
# Skill-matched pairs: similar pay / education level, different AI exposure.
# These are the cleanest identification of an *AI* effect (as opposed to a
# generic "tech over-hiring hangover" or rate-cycle effect).
MATCHED_PAIRS = [
    ("FOPRSE", "CUSTSERV", "Entry-level: food service vs. customer service"),
    ("MEDITECH", "MEDIINFO", "Healthcare support: med. technician vs. med. records/coding"),
    ("CONS", "ARCH", "Built environment: construction trades vs. architecture/drafting"),
    ("LOGISUPP", "INDEDO", "Support roles: logistics support vs. documentation"),
    ("NURS", "HUMARESO", "Mid-skill: nursing vs. HR"),
]

# ---------------------------------------------------------------------------
# 2. BLS Current Employment Statistics (FRED, monthly, thousands, SA)
# ---------------------------------------------------------------------------
CES_FRONTLINE = {
    "USCONS": "Construction", "CES2023800001": "Specialty trade contractors",
    "CES6562000001": "Health care & social assistance", "CES6562100001": "Ambulatory health care",
    "CES6562200001": "Hospitals", "CES4300000001": "Transportation & warehousing",
    "USLAH": "Leisure & hospitality", "MANEMP": "Manufacturing",
}
CES_KNOWLEDGE = {
    "USINFO": "Information", "CES6054150001": "Computer systems design",
    "CES6054000001": "Professional, scientific & technical",
    "CES5552000001": "Finance & insurance", "CES6055000001": "Management of companies",
    "CES6056130001": "Employment services",
}
# Average hourly earnings, all employees ($/hr)
AHE_FRONTLINE = {"CES2000000003": "Construction", "CES6500000003": "Education & health",
                 "CES7000000003": "Leisure & hospitality", "CES4300000003": "Transport & warehousing",
                 "CES3000000003": "Manufacturing"}
AHE_KNOWLEDGE = {"CES5000000003": "Information", "CES5500000003": "Financial activities",
                 "CES6000000003": "Professional & business services"}
OTHER_FRED = {
    "CGBD2024": "Unemployment rate, bachelor's degree, age 20-24",
    "LNS14000036": "Unemployment rate, age 20-24",
    "LNS14027662": "Unemployment rate, bachelor's+ age 25+",
    "LNS14027660": "Unemployment rate, HS grad no college, 25+",
    "UNRATE": "Unemployment rate",
    "JTS2300JOL": "Job openings: construction",
    "JTS6200JOL": "Job openings: health care & social assistance",
    "JTS540099JOL": "Job openings: professional & business services",
    "PAYEMS": "Total nonfarm payrolls",
    "IHLIDXUS": "Indeed job postings, all",
}

# ---------------------------------------------------------------------------
# 3. Equity baskets — chosen by BUSINESS MODEL, not by past returns.
#    Losers are deliberately kept in (TBI, AMN, CHGG on the other side...).
# ---------------------------------------------------------------------------
FRONTLINE_LONG = {
    # train the frontline workforce
    "UTI": "Universal Technical Institute (trade school)",
    "LINC": "Lincoln Educational (trade school)",
    "ATGE": "Adtalem (nursing & medical schools)",
    # clothe and equip the frontline workforce (revenue ~ frontline headcount)
    "CTAS": "Cintas (uniform rental)", "UNF": "UniFirst (uniform rental)",
    "SNA": "Snap-on (mechanic tools)", "BOOT": "Boot Barn (workwear)",
    "GWW": "Grainger (MRO supply)", "FAST": "Fastenal (MRO supply)",
    "MSM": "MSC Industrial (MRO supply)", "WSO": "Watsco (HVAC distribution to contractors)",
    # sell frontline labour hours
    "FIX": "Comfort Systems (mechanical/electrical trades)", "EME": "EMCOR (trades contractor)",
    "ROL": "Rollins (pest control technicians)",
    "TBI": "TrueBlue (blue-collar staffing)", "AMN": "AMN Healthcare (nurse staffing)",
}
KNOWLEDGE_SHORT = {
    # sell white-collar labour hours that AI substitutes
    "ACN": "Accenture", "CTSH": "Cognizant", "EPAM": "EPAM Systems", "GLOB": "Globant",
    "INFY": "Infosys ADR", "WIT": "Wipro ADR", "IT": "Gartner (research seats)",
    "RHI": "Robert Half (white-collar staffing)", "KFRC": "Kforce (tech/finance staffing)",
    "ASGN": "ASGN (IT staffing)",
    # customer-service BPO (most directly AI-substitutable labour)
    "CNXC": "Concentrix", "TTEC": "TTEC", "TASK": "TaskUs",
    # white-collar upskilling / freelance knowledge marketplaces
    "CHGG": "Chegg", "COUR": "Coursera", "UPWK": "Upwork", "FVRR": "Fiverr",
}
BENCH = "SPY"

# ---------------------------------------------------------------------------
# 4. Frontier-AI capability events (public release / announcement dates).
#    Day 0 = first US trading day on/after the date.  Approximate where the
#    rollout was staggered; see README.
# ---------------------------------------------------------------------------
AI_EVENTS = {
    "2022-11-30": "ChatGPT launch",
    "2023-03-14": "GPT-4",
    "2023-12-06": "Gemini 1.0",
    "2024-03-04": "Claude 3",
    "2024-05-13": "GPT-4o",
    "2024-06-20": "Claude 3.5 Sonnet",
    "2024-09-12": "OpenAI o1-preview",
    "2024-10-22": "Claude computer use",
    "2024-12-20": "OpenAI o3 announced",
    "2025-01-27": "DeepSeek-R1 market shock",
    "2025-02-24": "Claude 3.7 Sonnet + Claude Code",
    "2025-04-16": "OpenAI o3 / o4-mini",
    "2025-05-22": "Claude 4",
    "2025-08-07": "GPT-5",
    "2025-09-29": "Claude Sonnet 4.5",
    "2025-11-18": "Gemini 3",
    "2025-11-24": "Claude Opus 4.5",
    "2026-01-12": "Claude Cowork",
    "2026-02-03": "Cowork plugins / 'SaaSpocalypse' selloff",
    "2026-02-05": "Claude Opus 4.6 / GPT-5.3-Codex",
}
