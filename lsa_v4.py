# # Learning Standards Analysis
#
# Simeon Wong
# MAT188 2023F at the University of Toronto

# %%
import pandas as pd
import numpy as np
import glob
from itertools import compress
import wwparse_v2
from tqdm import tqdm
import re
import argparse
import multiprocessing as mp
import psutil
import functools
from types import SimpleNamespace
import os, os.path
import yaml

nthreads = psutil.cpu_count(logical=False) - 1

class Configs(SimpleNamespace):

    def __init__(self, dictionary:dict, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, Configs(value))
            else:
                self.__setattr__(key, value)

    def __getattribute__(self, value):
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None  

def extract_tutorial_number(x: str):
    rel = re.search(r'TUT(\d{4})', x)

    if rel is None:
        return pd.NA
    else:
        return int(rel.group(1))


def grade_by_ls(student: tuple, lsref):
    this_student, this_student_scores = student

    this_standards_achieved = pd.Series(index=pd.MultiIndex.from_frame(
        lsref[['modality', 'standard']].drop_duplicates()),
                                        name=this_student)

    for this_idx, this_standard in lsref.iterrows():
        question_keys = [x.strip() for x in this_standard['reqs'].split(',')]

        if '|' in question_keys[0]:
            n_correct_required = int(question_keys[0].split('|')[0])
            question_keys[0] = question_keys[0].split('|')[1]
            ratio_required = n_correct_required / len(question_keys)
        else:
            ratio_required = 1

        # get correctness for each question
        question_isgraded = [
            ~np.any(this_student_scores.loc[x, 'is_graded'] == False)
            if x in this_student_scores.index else True for x in question_keys
        ]
        question_iscorrect = [
            np.any(this_student_scores.loc[x, 'correct'])
            if x in this_student_scores.index else False for x in question_keys
        ]

        # check if the required number of questions are correct
        if np.sum(question_isgraded) == 0:
            ls_cor = np.nan
        else:
            corsum = np.nansum(question_iscorrect)
            ls_cor = int(
                (corsum >= 1)
                and (corsum / np.sum(question_isgraded) >= ratio_required))

        # if all requirements are met, set this standard to true
        this_standards_achieved[tuple(this_standard[['modality',
                                                     'standard']])] = ls_cor

    return this_standards_achieved


def load_data(config: SimpleNamespace):
    #######################################################################
    ## Data loading

    # import table of learning standards
    lsref = pd.read_excel(config.input_paths.standards_lookup_table,
                          sheet_name='grading',
                          engine='openpyxl')

    lsref = lsref.set_index('standard').stack().reset_index()
    lsref.columns = ['standard', 'modality', 'reqs']

    # ignore exams
    if not config.compute_exams:
        lsref = lsref[lsref['modality'] != 'exam']

    # load roster
    roster = pd.read_csv(config.input_paths.utatg_roster)
    gradebook = pd.read_csv(config.input_paths.quercus_gradebook)
    gradebook = gradebook[~gradebook['SIS User ID'].isna()]
    gradebook = gradebook[['SIS User ID',
                           'Section']].set_index(['SIS User ID'])
    gradebook['tut'] = gradebook['Section'].apply(extract_tutorial_number)
    # gradebook['tut_day'] = gradebook['tut'].apply(
    #     lambda x: tut_day_tbl.loc[x, 'day'])

    roster = roster[roster['UTORid'].isin(
        gradebook.index)]  # drop students who dropped the course
    roster.rename(columns={'Student Number': 'SID'}, inplace=True)
    roster['SID'] = pd.to_numeric(roster['SID'], errors='coerce')

    if config.debug:
        roster = pd.concat(
            (roster[:20], roster[-20:]
             ))  # DEBUGGING: only keep first and last 20 students for speed

    scores = []

    ##### WEBWORK #####
    for filename in glob.glob(config.input_paths.webwork_glob):
        print(f'Loading {filename}...')
        this_score = wwparse_v2.parse_csv(filename)

        # a question is correct if all parts are correct
        this_score['correct'] = this_score['score'] == 1

        scores.append(this_score)

    ##### TUTORIALS #####
    # for filename in glob.glob(config.input_paths.tutorial_glob):
    #     print(f'Loading {filename}...')
    #     this_score = pd.read_csv(filename)
    #     this_score['SID'] = pd.to_numeric(this_score['SID'], errors='coerce')

    #     # merge with roster
    #     this_score = this_score.merge(roster[['Email', 'UTORid']],
    #                                   how='left',
    #                                   on='Email')
    #     email_merged = this_score[~this_score['UTORid'].isna()]
    #     email_unmerged = this_score[this_score['UTORid'].isna()]
    #     email_unmerged = email_unmerged.drop(
    #         columns=['Email', 'UTORid']).merge(roster[['SID', 'UTORid']],
    #                                            how='left',
    #                                            on='SID')
    #     this_score = pd.concat([email_merged, email_unmerged])
    #     this_score = this_score.rename(columns={'UTORid': 'login_name'})

    #     # parse score key
    #     ls_cols = [
    #         x for x in this_score.columns
    #         if re.match(r'^tut\d{1,2}\-\d\-\d{1,2}$', x)
    #     ]
    #     this_score = this_score[['login_name', 'SBG'] + ls_cols]

    #     this_score = this_score.melt(id_vars=['login_name', 'SBG'],
    #                                  var_name='score_key',
    #                                  value_name='correct')
    #     this_score['score_key_sbg'] = this_score['score_key'].apply(
    #         lambda x: re.match(r'^tut\d{1,2}\-(\d)\-\d{1,2}$', x).group(1))

    #     # identify graded questions
    #     # method: from tutorial CSV files
    #     # this_score['is_graded'] = this_score['SBG'].astype(
    #     #     int) == this_score['score_key_sbg'].astype(int)

    #     this_score = this_score[[
    #         'login_name', 'score_key', 'correct', 'is_graded'
    #     ]]

    #     scores.append(this_score)

    tdc = pd.read_csv(config.input_paths.tutorial_consolidated)
    tdc = tdc.fillna(False)
    tdc_score_keys = [
        x for x in tdc.columns
        if re.match(r'^tut\d{1,2}[\.\-]\d[\.\-]\d{1,2}$', x)
    ]
    tdc_scores = tdc.melt(id_vars=['UTORid'],
                          value_vars=tdc_score_keys,
                          var_name='score_key',
                          value_name='correct')
    tdc_scores[['tut',
                'tut_gradegroup']] = tdc_scores['score_key'].str.extract(
                    r'tut(\d{1,2})[\.\-](\d)[\.\-]\d{1,2}')
    tdc_scores['tut'] = tdc_scores['tut'].astype(int)
    tdc_scores['tut_gradegroup'] = tdc_scores['tut_gradegroup'].astype(int)

    # normalize score_key
    tdc_scores['score_key'] = tdc_scores['score_key'].str.replace('.', '-')

    tdc_graded_cols = [x for x in tdc.columns if re.match(r'SBG\d{1,2}', x)]
    tdc_graded = tdc.melt(id_vars=['UTORid'],
                          value_vars=tdc_graded_cols,
                          var_name='tut',
                          value_name='tut_gradegroup')
    tdc_graded['tut'] = tdc_graded['tut'].str.extract(r'SBG(\d{1,2})').astype(int)

    tdc_scores['is_graded'] = tdc_scores.apply(lambda x: (
        (x['UTORid'] == tdc_graded['UTORid']) &
        (x['tut'] == tdc_graded['tut']) &
        (x['tut_gradegroup'] == tdc_graded['tut_gradegroup'])).sum() > 0,
                                               axis=1)

    tdc_scores = tdc_scores.rename(
        columns={'UTORid': 'login_name'})
    tdc_scores = tdc_scores[[
        'login_name', 'score_key', 'correct', 'is_graded'
    ]]
    scores.append(tdc_scores)

    # load midterm data
    if config.compute_exams:
        if config.input_paths.midterm_glob:
            for filename in glob.glob(config.input_paths.midterm_glob):
                print(f'Loading {filename}...')
                this_score = pd.read_csv(filename)

                # remove sum rows
                this_score = this_score[~this_score['SID'].isna()]

                # merge with roster
                this_score = this_score.merge(roster[['Email', 'UTORid']],
                                            how='left',
                                            on='Email')
                this_score = this_score.rename(columns={'UTORid': 'login_name'})

                # parse score key
                ls_cols = [x for x in this_score.columns if '|' in x]
                this_score = this_score[['login_name'] + ls_cols]
                this_score.columns = ['login_name'
                                    ] + [x.split('|')[1] for x in ls_cols]

                # stack into long format
                this_score = this_score.melt(id_vars='login_name',
                                            var_name='score_key',
                                            value_name='correct')

                # check if correct has type string, convert to int
                this_score['correct'] = this_score['correct'].map({
                    'TRUE': 1,
                    'FALSE': 0,
                    True: 1,
                    False: 0
                })

                scores.append(this_score)


    if config.input_paths.midterm_consolidated:
        print('Loading ' + config.input_paths.midterm_consolidated)
        midterm_data = pd.read_excel(config.input_paths.midterm_consolidated)
        
        # merge with roster
        midterm_data = midterm_data.merge(roster[['Email', 'UTORid']],
                                        how='left',
                                        on='Email')
        midterm_data = midterm_data.rename(columns={'UTORid': 'login_name'})

        # parse score key
        ls_cols = [x for x in midterm_data.columns if re.match(r'^ex\d\-\d\-\d{1,2}', x)]
        midterm_data = midterm_data[['login_name'] + ls_cols]
        midterm_data.columns = ['login_name'
                                ] + ls_cols

        # stack into long format
        midterm_data = midterm_data.melt(id_vars='login_name',
                                        var_name='score_key',
                                        value_name='correct')
        midterm_data = midterm_data.dropna(subset=['login_name'])

        # check if correct has type string, convert to int
        midterm_data['correct'] = midterm_data['correct'].map({
            'TRUE': 1,
            'FALSE': 0,
            True: 1,
            False: 0
        })

        scores.append(midterm_data)


    # concatenate all scores
    scores = pd.concat(scores, ignore_index=True)

    # load manually scored items
    manual_scores = pd.read_excel(config.input_paths.manual_scores)

    scores_key = list(scores[['login_name',
                              'score_key']].itertuples(index=False, name=None))
    manual_scores_key = list(manual_scores[['login_name', 'score_key'
                                            ]].itertuples(index=False,
                                                          name=None))
    in_manual_scores = [x in manual_scores_key for x in scores_key]

    # drop values from scores where login_name and score_key match from manual_scores
    scores = scores.iloc[~np.array(in_manual_scores)]
    scores = pd.concat([scores, manual_scores], ignore_index=True)

    # remove score rows without an associated utorid
    scores = scores[scores['login_name'] != ''].dropna(subset=['login_name'])

    # save for debugging
    scores.to_csv(config.output_dir + '/debug_raw_scores.csv')

    return scores, roster, lsref


def run(config: SimpleNamespace):
    scores, roster, lsref = load_data(config)

    #######################################################################
    # Which learning standards has each student achieved?
    # remove requirements that don't have associated questions in our score db
    uniq_scorekey = scores['score_key'].unique()

    # Save unique score keys for debugging
    pd.Series(uniq_scorekey).to_csv(config.output_dir +
                                    '/debug_uniq_scorekey.csv',
                                    index=False)

    for this_idx, this_standard in lsref.iterrows():
        if (this_standard['reqs'] == '') or pd.isna(this_standard['reqs']):
            continue

        # parse standards
        if '|' in this_standard['reqs']:
            n_req, reqstr = this_standard['reqs'].split('|')[0:2]
            n_req = int(n_req)
            reqs = reqstr.split(',')
        else:
            reqs = this_standard['reqs'].split(',')
            n_req = len(reqs)

        in_db = [tr.strip() in uniq_scorekey for tr in reqs]

        # only keep standards that have questions in the db
        reqs = list(compress(reqs, in_db))

        n_req = min(n_req, len(reqs))

        # reconstruct reqs string
        if n_req > 0:
            lsref.loc[this_idx, 'reqs'] = str(n_req) + '|' + (','.join(reqs))
        else:
            lsref.loc[this_idx, 'reqs'] = pd.NA

    # remove empty standards with no associated items
    lsref = lsref[~lsref['reqs'].isna()]

    # initialize output table
    # standards_achieved = pd.DataFrame(index=scores['login_name'].unique(),
    #                                   columns=pd.MultiIndex.from_frame(lsref[[
    #                                       'modality', 'standard'
    #                                   ]].drop_duplicates()))

    # with multiple threads, call the grading function for each student
    # with mp.Pool(nthreads) as p:
    #     standards_achieved = pd.concat(tqdm(
    #         p.imap_unordered(
    #             functools.partial(grade_by_ls, lsref=lsref),
    #             zip(roster['UTORid'].unique(), [
    #                 scores[scores['login_name'] == this_student].set_index(
    #                     'score_key')
    #                 for this_student in roster['UTORid'].unique()
    #             ]),
    #             chunksize=10,
    #         ),
    #         total=len(roster['UTORid'].unique()),
    #         desc='Evaluating learning standards by student'),
    #                                    axis=1).T
    standards_achieved = []
    for this_student in tqdm(roster['UTORid'].unique(),
                             desc='Evaluating learning standards by student'):
        this_student_scores = scores[scores['login_name'] ==
                                     this_student].set_index('score_key')
        standards_achieved.append(
            grade_by_ls((this_student, this_student_scores), lsref))

    standards_achieved = pd.concat(standards_achieved, axis=1).T

    # compute fraction standards achieved across each modality
    modalities = lsref['modality'].unique()
    for this_modality in modalities:
        this_modality_standards = standards_achieved.loc[:, this_modality]
        standards_achieved.loc[:,
                               ('fraction_achieved',
                                this_modality)] = this_modality_standards.mean(
                                    axis=1, skipna=True)

    # Join student names for easy lookup
    roster.set_index('UTORid', inplace=True)
    standards_achieved = standards_achieved[standards_achieved.index.isin(
        roster.index)]
    standards_achieved[('student',
                        'first_name')] = roster.loc[standards_achieved.index,
                                                    'First Name']
    standards_achieved[('student',
                        'last_name')] = roster.loc[standards_achieved.index,
                                                   'Last Name']
    standards_achieved[('student', 'SID')] = roster.loc[standards_achieved.index,
                                                        'SID']
    standards_achieved.sort_index(axis=0, inplace=True)

    standards_achieved.to_csv(config.output_dir + '/standards_achieved.csv')




#%% Main
#######################################################################
# Parse arguments
if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--compute-exams', action='store_true')
    parser.add_argument('--generate-reports', action='store_true')
    args = parser.parse_args(
    ) if 'ipykernel' not in sys.modules else parser.parse_args(
        ['--debug', 'config2024.yml'])

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = Configs(config | args.__dict__)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    run(config)

    if args.generate_reports:
        import make_ls_report_v2
        make_ls_report_v2.run(config)
