import pandas as pd

def parse_csv(filename: str):
    df = pd.read_csv('./data./sWeBWorK_1ful.csv', header=[6, 2])

    df.columns = pd.MultiIndex.from_tuples([(col[0].strip(), col[1].strip())
                                            for col in df.columns])

    # drop the remainder of the header rows
    df = df.iloc[df[df['STUDENT ID'].iloc[:, 0].str.startswith('STUDENT ID')].
                index[0] + 1:]

    parsed_df = parse_df(df)

    return parsed_df


def parse_df(df: pd.DataFrame):
    # Fix the column reference issue by selecting the correct multi-index column for 'login ID'
    df.columns = df.columns.map(lambda x: x[0].strip() + " " + x[1].strip() if x[1] else x[0].strip())

    # Now try to melt the dataframe again
    # Filter out the necessary columns
    login_col = 'login ID'  # 'login ID' field
    problem_cols = [col for col in df.columns if 'STATUS' in col or '#incorr' in col]  # Problem related columns

    # Create a long dataframe by melting the wide dataframe
    long_df = pd.melt(df,
                      id_vars=[login_col],
                      value_vars=problem_cols,
                      var_name='problem_detail',
                      value_name='value')


    # Now, split 'problem_detail' into 'metric' and 'problem_num'
    long_df['metric'] = long_df['problem_detail'].apply(
        lambda x: 'total_pct' if 'STATUS' in x else 'n_incor')
    long_df['problem_num'] = long_df['problem_detail'].apply(
        lambda x: ''.join(filter(str.isdigit, x)))

    # Drop the original 'problem_detail' column
    long_df = long_df.drop(columns=['problem_detail'])

    # Pivot the dataframe to have total_pct and n_incor as columns
    parsed_df = long_df.pivot_table(index=['login ID', 'problem_num'],
                                    columns='metric',
                                    values='value',
                                    aggfunc='first').reset_index()

    # Rename columns to fit the required format
    parsed_df.columns = ['login_name', 'problem_num', 'n_incor', 'total_pct']

    parsed_df = parsed_df.sort_values(by=['login_name', 'problem_num']).reset_index(drop=True)

    return parsed_df

