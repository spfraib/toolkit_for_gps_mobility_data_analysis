import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit, lag, countDistinct, to_timestamp
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, TimestampType, DoubleType
import pandas as pd
from datetime import timedelta


def time_at_work(user_tmp, date_trunc, cluster_label):
    """minimum fraction of 'duration' per day that you don't spend in your home_location for each cluster stop -> [10%,20%,30%]
       if user_tmp[user_tmp.location_type !='H'].empty:
       return np.NaN
    Args:
        x (_type_): _description_
        user_tmp (DataFrame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration' and 'cluster_label'

    Returns:
        Float:  time_at_work / time_not_at_home
    """
    time_not_at_home = user_tmp[user_tmp.date_trunc ==
                                date_trunc]['duration'].sum()
    time_at_work = user_tmp[(user_tmp['cluster_label'] == cluster_label) & (
            user_tmp.date_trunc == date_trunc)]['duration'].sum()
    return time_at_work / time_not_at_home


def days_at_work_dynamic(x, user_tmp, work_period_window):
    """_summary_

    Args:
        x (_type_): _description_
        user_tmp (_type_): _description_
        work_period_window (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 2. minimum fraction of observed days of activity -> [20%,30%,40%]
    tmpdf = user_tmp[(user_tmp.date_trunc >= x.date_trunc - timedelta(
        days=work_period_window)) & (user_tmp.date_trunc <= x.date_trunc)]
    active_days = tmpdf['date_trunc'].nunique()

    tmpdf = tmpdf[(user_tmp.cluster_label == x.cluster_label)]
    work_days = tmpdf['date_trunc'].nunique()

    return work_days / active_days


def days_at_home_dynamic(x, user_tmp, home_period_window):
    # 2. minimum fraction of observed days of activity -> [20%,30%,40%]
    tmpdf = user_tmp[(user_tmp.date_trunc >= x.date_trunc - timedelta(
        days=home_period_window)) & (user_tmp.date_trunc <= x.date_trunc)]
    active_days = tmpdf['date_trunc'].nunique()

    tmpdf = tmpdf[(user_tmp.cluster_label == x.cluster_label)]
    home_days = tmpdf['date_trunc'].nunique()

    return home_days / active_days



schema_df = StructType([
    StructField('user_id', StringType(), False),
    StructField('t_start', LongType(), False),
    StructField('t_end', LongType(), False),
    StructField('duration', LongType(), False),
    StructField('lat', DoubleType(), False),
    StructField('lon', DoubleType(), False),
    StructField('total_duration_stop_location', LongType(), False),
    StructField('total_pings_stop', LongType(), False),
    StructField('cluster_label', LongType(), False),
    StructField('median_accuracy', DoubleType(), False),
    StructField('location_type', StringType(), True),
    StructField('home_label', LongType(), True),
    StructField('work_label', LongType(), True),
    StructField('geom_id', StringType(), False),
    StructField('date', TimestampType(), True),
    StructField('t_start_hour', IntegerType(), True),
    StructField('t_end_hour', IntegerType(), True),
    StructField("date_trunc", TimestampType(), True)
])


def get_durations(durations, start_hour_day, end_hour_day):
    durations = (durations
                 .withColumn('hour', F.hour('date'))
                 .withColumn('day_night', F.when((col('hour') >= start_hour_day) & (col('hour') < end_hour_day), 'day')
                             .when((col('hour') < start_hour_day) | (col('hour') >= end_hour_day), 'night')
                             .otherwise(None))  # stops that cross day/night boundaries
                 .select('user_id', 'date_trunc', 'day_night', 'location_type', 'duration'))
    total_durations = durations.groupby('date_trunc', 'user_id').agg(
        F.sum('duration').alias('total_duration'))
    durations = durations.groupby('date_trunc', 'day_night', 'user_id').pivot(
        'location_type').agg(F.sum('duration').alias('duration'))
    durations = durations.join(total_durations, on=['date_trunc', 'user_id'])
    durations = durations.withColumn('absolute_H', col('H'))
    durations = durations.withColumn('absolute_W', col('W'))
    durations = durations.withColumn('absolute_O', col('O'))
    durations = durations.withColumn('H', col('H') / col('total_duration'))
    durations = durations.withColumn('W', col('W') / col('total_duration'))
    durations = durations.withColumn(
        'O', col('O') / col('total_duration')).drop('total_duration')
    return durations


def home_rolling_on_date(x, home_period_window, min_periods_over_window):
    # the output dataframe will have as index the last date of the window and consider the previous "c.home_period_window" days to compute the window. Notice that this will be biased for the first c.home_period_window
    x = x.sort_values('date_trunc')
    x = x.set_index('date_trunc')
    tmp = x[['duration', 'total_pings_stop']].rolling(f'{home_period_window}D', min_periods=int(
        min_periods_over_window * home_period_window)).sum()
    tmp['days_count'] = x['duration'].rolling(
        f'{home_period_window}D').count()

    return tmp


def initialize_user_df(user_df):
    user_df['location_type'] = 'O'
    user_df['home_label'] = -1
    user_df['work_label'] = -1
    return user_df

def get_home_tmp(user_df, start_hour_day, end_hour_day):
    return user_df[(user_df['t_start_hour'] >= end_hour_day) | (user_df['t_end_hour'] <= start_hour_day)].copy()

def compute_cumulative_duration(home_tmp, home_period_window, min_periods_over_window, min_pings_home_cluster_label):
    home_tmp = home_tmp[['cluster_label', 'date_trunc', 'duration', 'total_pings_stop']].groupby(['cluster_label', 'date_trunc']).sum().reset_index().sort_values('date_trunc')
    home_tmp = home_tmp.merge(home_tmp[['date_trunc', 'cluster_label', 'duration', 'total_pings_stop']].groupby(['cluster_label']).apply(home_rolling_on_date, home_period_window, min_periods_over_window).reset_index(), on=['date_trunc', 'cluster_label'], suffixes=('', '_cum'))
    home_tmp = home_tmp[home_tmp.total_pings_stop_cum > min_pings_home_cluster_label].drop('total_pings_stop_cum', axis=1)
    return home_tmp.dropna(subset=['duration_cum'])

def add_home_label(user_df, home_tmp):
    date_cluster = home_tmp.drop_duplicates(['cluster_label', 'date_trunc'])[['date_trunc', 'cluster_label']].copy()
    date_cluster = date_cluster.drop_duplicates(['date_trunc'])
    home_label = list(zip(date_cluster.cluster_label, date_cluster.date_trunc))
    idx = pd.MultiIndex.from_frame(user_df[['cluster_label', 'date_trunc']])
    user_df.loc[idx.isin(home_label), 'home_label'] = user_df.loc[idx.isin(home_label), 'cluster_label']
    return user_df

def interpolate_missing_dates(user_df, home_tmp):
    first_day = home_tmp['date_trunc'].min()
    last_day = home_tmp['date_trunc'].max()
    base_dates = pd.date_range(start=first_day, end=last_day)
    date_cluster = home_tmp.drop_duplicates(['cluster_label', 'date_trunc'])[['date_trunc', 'cluster_label']].copy()
    date_cluster = date_cluster.drop_duplicates(['date_trunc'])
    date_cluster = date_cluster.sort_values(by='date_trunc').set_index('date_trunc')
    date_cluster = date_cluster.reindex(base_dates)
    date_cluster = date_cluster.interpolate(method='nearest').ffill().bfill() if pd.notna(date_cluster['cluster_label']).sum() > 1 else date_cluster.ffill().bfill()
    date_cluster.index.name = 'date_trunc'
    date_cluster = date_cluster.reset_index()
    home_label = list(zip(date_cluster.cluster_label, date_cluster.date_trunc))
    idx = pd.MultiIndex.from_frame(user_df[['cluster_label', 'date_trunc']])
    user_df.loc[idx.isin(home_label), 'location_type'] = 'H'
    return user_df

def remove_unused_cols(df):
    return df[['uid', 't_start', 't_end', 'duration', 'latitude', 'longitude',
               'total_duration_stop_location', 'total_pings_stop',
               'cluster_label', 'median_accuracy', 'location_type',
               'home_label', 'work_label', 'geom_id', 'date', 't_start_hour',
               't_end_hour', 'weekday', 'date_trunc']]

def get_labels_home(user_df, start_hour_day, end_hour_day, min_pings_home_cluster_label, work_activity_average, home_period_window, min_periods_over_window):
    """Label the stops of a user as home or not

    Args:
        user_df (Data Frame): PySpark DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration' and 'cluster_label'
        start_hour_day (int): Starting hour of the activity day
        end_hour_day (int): Ending hour of the activity day
        min_pings_home_cluster_label (int): Minimum number of pings in a cluster to be considered as home
        work_activity_average (float): Average fraction of time spent at work to be considered as work
        home_period_window (int): Number of days to consider for the rolling window
        min_periods_over_window (int): Minimum number of days to consider for the rolling window

    Returns:
        Data Frame: PySpark DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration' and 'cluster_label'. It contains the following columns:
            - location_type: 'H' if home, 'W' if work, 'O' if other
            - home_label: cluster_label of the home location
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - weekday: day of the week of the stop
            - t_start_hour: starting hour of the stop
            - t_end_hour: ending hour of the stop
            - date: date of the stop
            - geom_id: geom_id of the stop
            - uid: user_id of the stop
            - t_start: starting timestamp of the stop
            - t_end: ending timestamp of the stop
            - duration: duration of the stop
            - latitude: latitude of the stop
            - longitude: longitude of the stop
            - total_duration_stop_location: total duration of the stop
            - total_pings_stop: total number of pings of the stop
            - cluster_label: cluster_label of the stop
            - median_accuracy: median accuracy of the stop
    """
    def pandas_labels_home(key,data):
        user_df = data
        user_df = initialize_user_df(user_df)
        home_tmp = get_home_tmp(user_df, start_hour_day, end_hour_day)
        if home_tmp.empty:
            return user_df
        home_tmp = compute_cumulative_duration(home_tmp, home_period_window, min_periods_over_window, min_pings_home_cluster_label)
        if home_tmp.empty:
            return user_df
        user_df = add_home_label(user_df, home_tmp)
        user_df = interpolate_missing_dates(user_df, home_tmp)
        return remove_unused_cols(user_df)#user_df) #if home_tmp.cluster_label.unique().size != 0 else user_df.drop(['location_type', 'home_label'], axis=1)

    schema_df = StructType([
    StructField('uid', StringType(), False),#
    StructField('t_start', LongType(), False),#
    StructField('t_end', LongType(), False),#
    StructField('duration', LongType(), False),#
    StructField('latitude', DoubleType(), False),#
    StructField('longitude', DoubleType(), False),#
    StructField('total_duration_stop_location', LongType(), False),#
    StructField('total_pings_stop', LongType(), False),#
    StructField('cluster_label', LongType(), False),#
    StructField('median_accuracy', DoubleType(), False),#
    StructField('location_type', StringType(), True),
    StructField('home_label', LongType(), True),
    StructField('work_label', LongType(), True),
    StructField('geom_id', StringType(), False),#
    StructField('date', TimestampType(), True),#
    StructField('t_start_hour', IntegerType(), True),#
    StructField('t_end_hour', IntegerType(), True),#
    StructField('weekday', IntegerType(), True),
    StructField("date_trunc", TimestampType(), True)#
    ])
    # print(user_df.columns)

    return user_df.groupBy("uid").applyInPandas(pandas_labels_home, schema = schema_df)

def work_rolling_on_date(x, work_period_window, min_periods_over_window_work):
    """If on average over "period" centered in "date" a candidate satisfy the conditions then for "date" is selected as WORK location

    Args:
        x (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        work_period_window (int): Number of days to consider for the rolling window
        min_periods_over_window_work (int): Minimum number of days to consider for the rolling window

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - duration_average: average duration of the stop
            - date_trunc: date of the stop
            - cluster_label: cluster_label of the stop
    """
    x = x.sort_values('date_trunc')
    return x.set_index('date_trunc')[['duration']].rolling(f'{work_period_window}D', min_periods=int(
        min_periods_over_window_work * work_period_window)).mean()


def get_work_tmp(user_df, start_hour_day, end_hour_day, home_list=None):
    """Get the work locations of a user

    Args:
        user_df (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        start_hour_day (int): Starting hour of the activity day
        end_hour_day (int): Ending hour of the activity day
        home_list (list of ints, optional): List of cluster_labels of the home locations. Defaults to None.

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - duration_average: average duration of the stop
            - date_trunc: date of the stop
            - cluster_label: cluster_label of the stop
    """
    if home_list is not None:
        work_tmp = user_df[~(user_df['cluster_label'].isin(home_list))].copy()
    else:
        work_tmp = user_df.copy()
    work_tmp = work_tmp[((work_tmp['t_start_hour'] >= start_hour_day + 4) & (work_tmp['t_end_hour'] <= end_hour_day - 6)) & (~work_tmp['weekday'].isin([1, 7]))]
    return work_tmp


def compute_average_duration(work_tmp, work_activity_average, work_period_window, min_periods_over_window_work):
    """Compute the average duration of the work locations

    Args:
        work_tmp (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        work_activity_average (float): Minimum average fraction of time spent at work to be considered as work
        work_period_window (int): Number of days to consider for the rolling window
        min_periods_over_window_work (int): Minimum number of days to consider for the rolling window

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - duration_average: average duration of the stop
            - date_trunc: date of the stop
            - cluster_label: cluster_label of the stop
    """
    work_tmp = work_tmp[['cluster_label', 'date_trunc', 'duration']].groupby(['cluster_label', 'date_trunc']).sum().reset_index()
    work_tmp = work_tmp.merge(work_tmp[['date_trunc', 'cluster_label', 'duration']].groupby(['cluster_label']).apply(work_rolling_on_date, work_period_window, min_periods_over_window_work).reset_index(), on=['date_trunc', 'cluster_label'], suffixes=('', '_average'))
    work_tmp = work_tmp[(work_tmp.duration_average >= work_activity_average)]
    return work_tmp.dropna(subset=['duration_average'])

def add_work_label(user_df, work_tmp):
    """Add the work label to the user_df

    Args:
        user_df (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        work_tmp (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - cluster_label: cluster_label of the stop
            - location_type: 'H' if home, 'W' if work, 'O' if other
            - home_label: cluster_label of the home location
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - weekday: day of the week of the stop
            - t_start_hour: starting hour of the stop
            - t_end_hour: ending hour of the stop
            - date: date of the stop
            - geom_id: geom_id of the stop
            - uid: user_id of the stop
            - t_start: starting timestamp of the stop
            - t_end: ending timestamp of the stop
            - duration: duration of the stop
            - latitude: latitude of the stop
            - longitude: longitude of the stop
            - total_duration_stop_location: total duration of the stop
            - total_pings_stop: total number of pings of the stop
            - median_accuracy: median accuracy of the stop
    """
    work_label = list(zip(work_tmp.cluster_label, work_tmp.date_trunc))
    idx = pd.MultiIndex.from_frame(user_df[['cluster_label', 'date_trunc']])
    user_df.loc[idx.isin(work_label), 'work_label'] = user_df.loc[idx.isin(work_label), 'cluster_label']
    return user_df

def interpolate_missing_dates_work(user_df, work_tmp):
    """Interpolate the missing dates of the work locations

    Args:
        user_df (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        work_tmp (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - location_type: 'H' if home, 'W' if work, 'O' if other
            - home_label: cluster_label of the home location
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - weekday: day of the week of the stop
            - t_start_hour: starting hour of the stop
            - t_end_hour: ending hour of the stop
            - date: date of the stop
            - geom_id: geom_id of the stop
            - uid: user_id of the stop
            - t_start: starting timestamp of the stop
            - t_end: ending timestamp of the stop
            - duration: duration of the stop
            - latitude: latitude of the stop
            - longitude: longitude of the stop
            - total_duration_stop_location: total duration of the stop
            - total_pings_stop: total number of pings of the stop
            - cluster_label: cluster_label of the stop
            - median_accuracy: median accuracy of the stop
    """
    first_day = work_tmp['date_trunc'].min()
    last_day = work_tmp['date_trunc'].max()
    base_dates = pd.date_range(start=first_day, end=last_day)
    date_cluster = work_tmp.drop_duplicates(['cluster_label', 'date_trunc'])[['date_trunc', 'cluster_label']].copy()
    date_cluster = date_cluster.drop_duplicates(['date_trunc'])
    date_cluster = date_cluster.sort_values(by='date_trunc').set_index('date_trunc')
    date_cluster = date_cluster.reindex(base_dates)
    date_cluster = date_cluster.interpolate(method='nearest').ffill().bfill() if pd.notna(date_cluster['cluster_label']).sum() > 1 else date_cluster.ffill().bfill()
    date_cluster.index.name = 'date_trunc'
    date_cluster = date_cluster.reset_index()
    work_label = list(zip(date_cluster.cluster_label, date_cluster.date_trunc))
    idx = pd.MultiIndex.from_frame(user_df[['cluster_label', 'date_trunc']])
    user_df.loc[idx.isin(work_label), 'location_type'] = 'W'
    return user_df


def get_labels_work(user_df, start_hour_day, end_hour_day, min_pings_home_cluster_label, work_activity_average, work_period_window, min_periods_over_window_work):
    """Get the work locations of a user

    Args:
        user_df (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        start_hour_day (int): Starting hour of the activity day
        end_hour_day (int): Ending hour of the activity day
        min_pings_home_cluster_label (int): Minimum number of pings in a cluster to be considered as home
        work_activity_average (float): Minimum average fraction of time spent at work to be considered as work
        work_period_window (int): Minimum number of days to consider for the rolling window
        min_periods_over_window_work (float): Minimum fraction of days to consider for the rolling window

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - location_type: 'H' if home, 'W' if work, 'O' if other
            - home_label: cluster_label of the home location
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - weekday: day of the week of the stop
            - t_start_hour: starting hour of the stop
            - t_end_hour: ending hour of the stop
            - date: date of the stop
            - geom_id: geom_id of the stop
            - uid: user_id of the stop
            - t_start: starting timestamp of the stop
            - t_end: ending timestamp of the stop
            - duration: duration of the stop
            - latitude: latitude of the stop
            - longitude: longitude of the stop
            - total_duration_stop_location: total duration of the stop
            - total_pings_stop: total number of pings of the stop
            - cluster_label: cluster_label of the stop
            - median_accuracy: median accuracy of the stop
    """
    def pandas_labels_work(key, data):
        user_df = data
        # user_df['location_type'] = 'O'
        #  user_df['work_label'] = -1
        home_list = user_df.loc[user_df["location_type"] == "H"]#.unique()
        if home_list.empty:
            home_list = None
        else:
            home_list = home_list["cluster_label"].unique()
        work_tmp = get_work_tmp(user_df, start_hour_day, end_hour_day, home_list)
        # work_tmp = get_work_tmp(user_df, start_hour_day, end_hour_day)
        if work_tmp.empty:
            return user_df
        work_tmp = compute_average_duration(work_tmp, work_activity_average, work_period_window, min_periods_over_window_work)
        if work_tmp.empty:
            return user_df
        user_df = add_work_label(user_df, work_tmp)
        user_df = interpolate_missing_dates_work(user_df, work_tmp)
        return remove_unused_cols(user_df)

    schema_df = StructType([
    StructField('uid', StringType(), False),
    StructField('t_start', LongType(), False),
    StructField('t_end', LongType(), False),
    StructField('duration', LongType(), False),
    StructField('latitude', DoubleType(), False),
    StructField('longitude', DoubleType(), False),
    StructField('total_duration_stop_location', LongType(), False),
    StructField('total_pings_stop', LongType(), False),
    StructField('cluster_label', LongType(), False),
    StructField('median_accuracy', DoubleType(), False),
    StructField('location_type', StringType(), True),
    StructField('home_label', LongType(), True),
    StructField('work_label', LongType(), True),
    StructField('geom_id', StringType(), False),
    StructField('date', TimestampType(), True),
    StructField('t_start_hour', IntegerType(), True),
    StructField('t_end_hour', IntegerType(), True),
    StructField('weekday', IntegerType(), True),
    StructField("date_trunc", TimestampType(), True)
    ])

    return user_df.groupBy("uid").applyInPandas(pandas_labels_work, schema=schema_df)


def get_labels(user_df, start_hour_day, end_hour_day, min_pings_home_cluster_label, work_activity_average, home_period_window, min_periods_over_window, work_period_window, min_periods_over_window_work):
    """Compute the labels of a user

    Args:
        user_df (Data Frame): DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'
        start_hour_day (int): Starting hour of the activity day
        end_hour_day (int): Ending hour of the activity day
        min_pings_home_cluster_label (int): Minimum number of pings in a cluster to be considered as home
        work_activity_average (float): Minimum average fraction of time spent at work to be considered as work
        work_period_window (int): Minimum number of days to consider for the rolling window
        min_periods_over_window_work (float): Minimum fraction of days to consider for the rolling window

    Returns:
        Data Frame: DataFrame containing all the stops of a user. It is filtered by cluster_label and date_trunc. Needs the columns 'duration', 'cluster_label'. It contains the following columns:
            - location_type: 'H' if home, 'W' if work, 'O' if other
            - home_label: cluster_label of the home location
            - work_label: cluster_label of the work location
            - date_trunc: date of the stop
            - weekday: day of the week of the stop
            - t_start_hour: starting hour of the stop
            - t_end_hour: ending hour of the stop
            - date: date of the stop
            - geom_id: geom_id of the stop
            - uid: user_id of the stop
            - t_start: starting timestamp of the stop
            - t_end: ending timestamp of the stop
            - duration: duration of the stop
            - latitude: latitude of the stop
            - longitude: longitude of the stop
            - total_duration_stop_location: total duration of the stop
            - total_pings_stop: total number of pings of the stop
            - cluster_label: cluster_label of the stop
            - median_accuracy: median accuracy of the stop
    """
    user_df = get_labels_home(user_df, start_hour_day, end_hour_day, min_pings_home_cluster_label, work_activity_average, home_period_window, min_periods_over_window)
    user_df = get_labels_work(user_df, start_hour_day, end_hour_day, min_pings_home_cluster_label, work_activity_average, work_period_window, min_periods_over_window_work)
    return user_df