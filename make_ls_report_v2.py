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


def build_tex(row: pd.Series, filename: Optional[str] = None):
    filename = filename or row.name

    # row.dropna(inplace=True)
    subset_cols = row.drop('student', level=0).drop('fraction_achieved',
                                                    level=0)

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
    rows = []
    for modality, key in subset_cols.index:
        if row[(modality, key)] == 0:
            achieved = 'No'
        elif row[(modality, key)] == 1:
            achieved = 'Yes'
        else:
            achieved = r'\textit{Not tested}'

        tblrow = f'\\PulledLS{{{key}}} & {achieved} & {modality} \\\\ \\midrule'
        rows.append(tblrow)

    template = template.replace("REPLdetailedtableREPL", '\n'.join(rows))

    with open(config.output_dir + "/ls_reports/tex/combined.tex", "a") as f:
        f.write(template)


def run(config: SimpleNamespace):
    student_progress = pd.read_csv(config.output_dir + "/standards_achieved.csv",
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
        filename = ri

        build_tex(row, filename=filename)

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
