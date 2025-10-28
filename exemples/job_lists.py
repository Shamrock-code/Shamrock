from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(
    0, 10, "Global Job Boards for HPC, Numerical Methods & Applied Mathematics", ln=True, align="C"
)
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)

sections = {
    "1. HPC-Focused Job Boards": [
        (
            "hpc.social Jobs Board",
            "https://hpc.social/jobs",
            "Community-driven HPC roles (developers, cluster admins, RSEs).",
        ),
        (
            "HPC-AI Society Job Board",
            "https://hpc-ai-society.org/job-board",
            "HPC, AI, simulation engineers, global academic/industry listings.",
        ),
        (
            "National Lab Portals (DOE, NERSC, ORNL, LLNL, ANL)",
            "https://science.osti.gov",
            "Postdoc and staff positions in HPC, numerical simulation, and applied computing.",
        ),
        (
            "PRACE / EuroHPC job listings",
            "https://prace-ri.eu",
            "European HPC centers, engineering and researcher roles.",
        ),
    ],
    "2. Applied Mathematics & Numerical Methods": [
        (
            "SIAM Career Center",
            "https://jobs.siam.org",
            "Applied mathematics, numerical analysis, computational science roles (academic + industry).",
        ),
        (
            "MathJobs.org (AMS)",
            "https://www.mathjobs.org",
            "Academic and research positions in mathematics and applied numerical analysis.",
        ),
        (
            "EU-MATHS-IN Jobs Portal",
            "https://www.eu-maths-in.eu/jobs",
            "European network for industrial and applied math opportunities.",
        ),
        (
            "EuroScienceJobs – Maths & Computing",
            "https://www.eurosciencejobs.com/jobs/maths_and_computing",
            "European research and industrial applied math positions.",
        ),
    ],
    "3. Computational Science / Research Software Engineering": [
        (
            "Research Software Alliance (ReSA)",
            "https://researchsoft.org",
            "Global network for RSEs and computational scientists; check member institutions' job boards.",
        ),
        (
            "NERSC Careers",
            "https://www.nersc.gov/about/careers/",
            "Computational science, HPC software, and performance engineering roles.",
        ),
        (
            "EPCC (University of Edinburgh) HPC Jobs",
            "https://www.epcc.ed.ac.uk/about/careers",
            "UK-based HPC research and RSE positions.",
        ),
        (
            "Compute Canada / Alliance Canada",
            "https://alliancecan.ca/en",
            "Canadian HPC and research software positions.",
        ),
    ],
    "4. Computational Astrophysics & Physics": [
        (
            "AAS Job Register",
            "https://jobregister.aas.org",
            "Astrophysics roles including computational and simulation-focused positions.",
        ),
        (
            "AcademicJobsOnline",
            "https://academicjobsonline.org",
            "Academic and postdoctoral roles in computational astrophysics and physics worldwide.",
        ),
        (
            "Euraxess",
            "https://euraxess.ec.europa.eu",
            "European research jobs including computational physics and astrophysics.",
        ),
        (
            "IOP Physics World Jobs",
            "https://www.physicsworldjobs.com",
            "Physics and computational science positions worldwide.",
        ),
    ],
    "5. Industry & Private Sector (Simulation / HPC)": [
        (
            "Hewlett Packard Enterprise (HPC & AI)",
            "https://careers.hpe.com",
            "HPC systems engineering, AI/ML simulation, and performance optimization.",
        ),
        (
            "NVIDIA Careers",
            "https://www.nvidia.com/en-us/about-nvidia/careers/",
            "GPU computing, numerical software, and AI research roles.",
        ),
        (
            "Dassault Systèmes Careers",
            "https://careers.3ds.com",
            "Simulation software, computational modeling, and applied numerical R&D.",
        ),
        (
            "Siemens Simcenter / ANSYS Careers",
            "https://www.ansys.com/en-gb/careers",
            "Engineering simulation, HPC solver development, and applied mathematics positions.",
        ),
    ],
}

for section, entries in sections.items():
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, section, ln=True)
    pdf.set_font("Helvetica", "", 12)
    for name, url, desc in entries:
        pdf.multi_cell(0, 8, f"- {name}\n  {url}\n  {desc}\n")
    pdf.ln(5)

pdf.output("Global_HPC_AppliedMath_JobBoards.pdf")
