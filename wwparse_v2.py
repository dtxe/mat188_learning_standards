import os.path
import pandas as pd


def parse_problem_num(x):
    x = x.strip()

    # if x is a numeric string
    if x.isdigit():
        return x
    else:
        return 'ID'


def parse_csv(filename: str):
    df = pd.read_csv(filename, header=[6, 2])

    df.columns = pd.MultiIndex.from_tuples([
        (parse_problem_num(col[1]), col[0].strip()) for col in df.columns
    ])

    set_name = os.path.splitext(os.path.split(filename)[1])[0].split('_')[-1]

    # drop the remainder of the header rows
    df = df.iloc[df[df[
        ('ID',
         'STUDENT ID')].str.startswith('STUDENT ID').fillna(False)].index[0] +
                 1:]

    problem_cols = [col for col in df.columns if col[1] == 'STATUS']

    # Create a long dataframe by melting the wide dataframe
    long_df = pd.melt(df,
                      id_vars=[('ID', 'login ID')],
                      value_vars=problem_cols,
                      var_name='problem_num',
                      value_name='value')
    long_df.columns = ['login_name', 'problem_num', 'score']
    long_df['score'] = long_df['score'].str.strip().astype(float)

    # construct score_key
    long_df['score_key'] = long_df['problem_num'].astype(int).apply(
        lambda x: f'{set_name}-{x:d}')

    # normalize login name
    long_df['login_name'] = long_df['login_name'].str.strip()

    long_df = long_df.sort_values(by=['login_name', 'score_key']).reset_index(
        drop=True)

    return long_df
