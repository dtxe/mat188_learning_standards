from typing import Optional

import os
import os.path
import shutil
import pandas as pd
import datetime
import argparse
import yaml
from types import SimpleNamespace

from tqdm import tqdm


def build_tex(config: SimpleNamespace, row: pd.Series):

    # row.dropna(inplace=True)
    subset_cols = row.drop('student', level=0).drop('summary', level=0)

    with open("ls_report_page.tex", "r") as f:
        template = f.read()

    # make replacements
    template = template.replace(
        "REPLfullnameREPL",
        f"{row[('student', 'first_name')]} {row[('student', 'last_name')]}")
    template = template.replace("REPLutoridREPL",
                                str(row[('student', 'student_id')]))

    # build summary table
    summary_data = subset_cols.groupby(level='modality').agg(['count', 'sum'])

    summary_tex = []
    for modality, (total, achieved) in summary_data.iterrows():
        summary_tex.append(
            f'{modality} & {achieved:.0f} & {total:.0f} \\\\ \\midrule')
    template = template.replace("REPLsummarytableREPL", '\n'.join(summary_tex))

    # build detailed table
    def achieved_to_text(val):
        if val == 0:
            # achieved = 'No'
            achieved = r'\text{\sffamily X}'
        elif val == 1:
            # achieved = 'Yes'
            achieved = r'\checkmark'
        else:
            # achieved = r'\textit{Not tested}'
            achieved = '\t'
        return achieved

    pivot = subset_cols.reset_index()
    pivot = pivot.pivot(columns='modality',
                        values=pivot.columns[2],
                        index='standard')
    pivot = pivot.map(achieved_to_text)
    pivot.index = pivot.index.map(lambda x: f'\\PulledLS{{{x}}}')
    # rows = pivot.to_latex()
    rows = pandas_to_latex(pivot)

    template = template.replace("REPLdetailedtableREPL", rows)

    with open(config.output_dir + "/ls_reports/tex/combined.tex", "a") as f:
        f.write(template)


def pandas_to_latex(df: pd.DataFrame):
    rows = [pandas_to_latex_hdr(df)]
    for i, row in df.iterrows():
        rows.append(i + ' & ' + ' & '.join([str(x)
                                            for x in row]) + r' \\ \midrule')

    rows.append(r'''\bottomrule
        \end{tabularx}''')

    return '\n'.join(rows)


def pandas_to_latex_hdr(df: pd.DataFrame):
    cols = 'l|' * len(df.columns)
    col_names = ' & '.join([f'\\textbf{{{x}}}' for x in df.columns])

    return r'''
    \begin{tabularx}{\textwidth}{||X| ''' + cols + r'''|}
         \toprule
         \textbf{Learning Standard}           & ''' + col_names + r'''   \\ \midrule \midrule
    \endhead
    '''


def run(config: SimpleNamespace):
    student_progress = pd.read_csv(config.output_dir +
                                   "/standards_achieved.csv",
                                   header=[0, 1],
                                   index_col=0)

    roster = pd.read_csv(config.input_paths.utatg_roster,
                         index_col=3)['Student Number']
    student_progress[('student', 'student_id')] = [
        roster[x] for x in student_progress.index
    ]

    if not os.path.exists(config.output_dir + '/ls_reports/tex'):
        os.makedirs(config.output_dir + '/ls_reports/tex')

    if not os.path.exists(config.output_dir + '/ls_reports/pdf'):
        os.makedirs(config.output_dir + '/ls_reports/pdf')

    # insert an empty first row into the dataframe
    templaterow = pd.DataFrame(columns=student_progress.columns,
                               index=['_template'])
    templaterow.iloc[0, :-3] = 0
    templaterow[[('student', 'first_name'), ('student', 'last_name'),
                 ('student', 'student_id')]] = ' '
    student_progress = pd.concat([templaterow, student_progress])

    # write header
    if os.path.exists(config.output_dir + "/ls_reports/tex/combined.tex"):
        os.remove(config.output_dir + "/ls_reports/tex/combined.tex")

    shutil.copy("ls_report_header.tex",
                config.output_dir + "/ls_reports/tex/combined.tex")

    # for testing, only build first 20 students
    if config.debug:
        student_progress = student_progress.iloc[::len(student_progress) // 20]

    for ri, row in tqdm(student_progress.iterrows(),
                        desc='Building reports',
                        total=len(student_progress)):
        build_tex(config, row)

    # end document
    with open(config.output_dir + "/ls_reports/tex/combined.tex", "a") as f:
        f.write("\n\\end{document}")

    # compile
    t1 = datetime.datetime.now()
    os.system(
        f'pdflatex -output-directory "{config.output_dir}/ls_reports/pdf" "{config.output_dir}/ls_reports/tex/combined.tex"'
    )

    # remove everything that doesn't end with pdf
    os.remove(config.output_dir + '/ls_reports/pdf/combined.aux')
    os.remove(config.output_dir + '/ls_reports/pdf/combined.log')
    os.remove(config.output_dir + '/ls_reports/pdf/combined.out')

    print(f'Built PDF in {(datetime.datetime.now() - t1).total_seconds()} s.')


def to_namespace(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_namespace(v)
    return SimpleNamespace(**d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build learning standard reports.')
    parser.add_argument('config',
                        help='Path to the configuration file.',
                        type=str)
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode, only generate 20 reports for testing.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = to_namespace(config | args.__dict__)

    run(config)
