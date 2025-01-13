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
    
## Algorithm and architecture
Learning standards are computed in 3 steps.

### 1. Compiling scores
* In long table format
* Rows = students x questions x attempts
* Raw or semi-digested data comes from grading platforms
    * WeBWorK for problem sets and gateway
    * Consolidated spreadsheet from Gradescope for Tutorials
    * Spreadsheets from Gradescope for Midterms
* Question items are identified by a **score_key**
    * For WeBWorK, Gateway, and exams, the format is `[assessment]-[question number]`. eg. `ww1-2` denotes WeBWorK problem set 1, question 2
    * For Tutorials, the format is `[assessment]-[grading group]-[question number]`. eg. `tut1-2-4` denotes tutorial 1, SBG grading group 2, question 4 of that group
    * The above descriptions are conventions, and except for tutorials, are not interpreted or parsed by the script, but are used directly in the next step to match against learning standards
    * For tutorials only, because not all tutorial questions are graded, the SGB grading group indicates whether that question was graded.
        * Students are assigned a grading group based on their tutorial session
* The output of this step is saved as `debug_raw_scores.csv`
    * Key columns: `login_name` (utorid), `score_key`, `correct`, `is_graded`
    * This table should contain all scoring data for all students for all problems that form part of the learning standards

### 2. Evaluating learning standards 
* Using the score table from the prev. step, and the `standards_lookup_table.xlsx` spreadsheet
    * Rows with learning standards, using the LS key from the Learning Standards LaTeX project
    * Cols are modalities (eg. exam, tutorial, gateway, webwork)
    * The cells indicate which, and how many correct questions are required to demonstrate this standard
        * Questions are identifed by their `score_key`s
        * Format is `[num req] | [score_key1], [score_key2], ...`. 
        * eg. `2|gw1-1,gw1-2,gw1-3`
            * Means to demonstrate this standard, two or more of the questions indicated by gw1-1, gw1-2, gw1-3 are marked as correct
* Learning standards are computed iteratively for each student:
    * For each standard x modality, the requirements are parsed, then evaluated against that students' correctly answered questions in the score table.
* The output of this step is a wide table saved to `standards_achieved.csv`
    * One row per student
    * Columns indicate the standard x modality

### 3. 