import pandas as pd
import numpy as np
from pandasql import sqldf
from datetime import datetime
from dateutil.relativedelta import relativedelta


def transform_enroll():
    """
    Transforms patient-month level enrollment data to patient 
       enrollment spans

    Args:
        none

    Returns:
        df: dataframe with patient ID, enrollment start, and enrollment end
    """
    # prep data for transformation
    df_in = pd.read_csv('patient_id_month_year.csv')
    df_in = df_in.dropna(axis=1,how='all').dropna(axis=0,how='all')
    assert [c.lower() for c in df_in.columns] == ['patient_id', 'month_year']
    assert len(df_in) == len(df_in.drop_duplicates())
    df_in['month_start'] = df_in['month_year'].apply(
        lambda x: datetime.strptime(x, '%m/%d/%y').date())
    df_in.sort_values(by=['patient_id', 'month_start'], inplace=True)

    # flag whether the month is the start or end of an enrollment span 
    df_in['start_flag'] = enroll_flag(df_in, 'start')
    df_in['end_flag'] = enroll_flag(df_in, 'end')

    # drop rows in the middle of enrollment spans and define lagged months
    df_start_end = df_in.loc[(df_in['end_flag']) | (df_in['start_flag'])]
    df_start_end['lag_month'] = shift_month_by_id(df_start_end, 'lag')

    # only keep rows at the end of enrollment spans
    df_end = df_start_end.loc[df_start_end['end_flag'] == 1]
    # define the end of the enrollment span (last day of the month)
    df_end['enrollment_end_date'] = df_end['month_start'].apply(
        lambda x: x + relativedelta(months=1) - relativedelta(days=1)
        )
    # define beginning of enrollment span with first day of month or lead month
    df_end['enrollment_start_date'] = df_end.apply(
        lambda x: x['lag_month'] if not x['start_flag'] else x['month_start'], 
        axis=1
        )

    print(f'Row count for enrollment spans: {len(df_end)}')
    return df_end[
        ['patient_id', 'enrollment_start_date', 'enrollment_end_date']
        ]


def enroll_flag(df, flag):
    """
    Flags an enrollment month if it is the beginning or end of an enrollment
       span

    Args:
        df (dataframe): dataframe with columns 'patient_id' and 'month_start'
        flag (str): type of flag; either 'start' or 'end' (not case sensitive)

    Returns:
        series: series of booleans
    """
    assert flag.lower() in ['start','end']
    flag_dict = {'start': [-1, 'lag'], 'end': [1, 'lead']}
    # calculate the next and previous months of enrollment with no gap
    df_flag = df.copy()
    df_flag['no_gap_month'] = df_flag['month_start'].apply(
        lambda x: x+relativedelta(months=flag_dict[flag][0])
        )
    # get the actual next and previous enrollment months for a patient ID
    df_flag['shift_month'] = shift_month_by_id(df, flag_dict[flag][1])
    return df_flag.apply(
        lambda x: False if x['no_gap_month'] == x['shift_month'] else True, 
        axis=1
        )


def shift_month_by_id(df, type):
    """
    Gets the lag or lead month start by patient ID

    Args:
        df (dataframe): dataframe with columns 'patient_id' and 'month_start'
        type (str): type of shift; either 'lag' or 'lead' (not case sensitive)

    Returns:
        series: series of shifted month start values
    """
    assert type.lower() in ['lag','lead']
    shift_val = np.where(type.lower() == 'lag', 1, -1)
    return df.groupby(['patient_id'])['month_start'].shift(int(shift_val))


def agg_visits(enroll_df):
    """
    Joins visit count data with enrollment data and aggregates the outpatient 
       visit counts by enrollment span

    Args:
        df (dataframe): dataframe with columns 'patient_id' and 'month_start'

    Returns:
        df: dataframe with patient ID, enrollment start, enrollment end, 
            outpatient visit count, and outpatient visit day count
    """
    # prep data for aggregation
    df_in = pd.read_csv('outpatient_visits_file.csv')
    df_in = df_in.dropna(axis=1,how='all').dropna(axis=0,how='all')
    assert [c.lower() for c in df_in.columns] == [
        'patient_id', 'date', 'outpatient_visit_count'
        ]
    df_in['visit_date'] = df_in['date'].apply(
        lambda x: datetime.strptime(x, '%m/%d/%y').date()
        )

    # join enrollment and visit data by ID and dates within enrollment spans, 
    # aggregating by enrollment spans
    sql_merge = '''
        select a.*
              ,sum(coalesce(b.outpatient_visit_count,0)) 
                    as ct_outpatient_visits
              ,coalesce(count(distinct b.visit_date), 0) 
                    as ct_days_with_outpatient_visits
        from enroll_df as a
            left join df_in as b
                on a.patient_id = b.patient_id
                and b.visit_date between 
                    a.enrollment_start_date and a.enrollment_end_date
        group by a.patient_id, a.enrollment_start_date, a.enrollment_end_date
    '''

    df_merge = sqldf(sql_merge, locals())
    unique_vals_ct = len(df_merge.ct_days_with_outpatient_visits.unique())
    print(f'Number of distinct counts of OP visits during enrollment spans: {unique_vals_ct}')
    return df_merge


def main():
    enroll_df = transform_enroll()
    enroll_df.to_csv('patient_enrollment_span.csv', index=False)
    agg_df = agg_visits(enroll_df)
    agg_df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
