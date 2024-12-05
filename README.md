# Parse student progress export from Webwork and do some analyses

Simeon Wong  
Written for MAT188 at the University of Toronto


## Workflow
1. Install Python, UV (package manager), and MikTex (LaTeX)
    `winget install astral-sh.uv python.python.3.12 miktex.miktex`
1. Install required packages
    `uv sync`
1. Download student progress data into `../Data`
    1. Download HTML-only Webwork student progress report webpages
    1. Gradescope exports
1. Setup global data files in `../Data`
    1. list of required learning standards
    1. student roster information from UTAGT
1. Download learning standards Latex project to `./Learning_Standards`
1. Run reports
    1. `lsa_v4.py` produces a CSV file indicating which learning standards were achieved by every student in the course
    1. `make_ls_report_v2.py` produces a PDF containing detailed learning reports for each individual student for upload to Gradescope
    
