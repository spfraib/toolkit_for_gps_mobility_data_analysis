from collections import Counter
from cpputils import get_stationary_events
from datetime import timezone
import numpy as np
from pyspark.sql.functions import lag, col, countDistinct, to_timestamp, lit, from_unixtime, pandas_udf, PandasUDFType
from sklearn.cluster import DBSCAN
import pandas as pd

from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, TimestampType, DoubleType, FloatType


def get_most_frequent_label(a):
    """Get the most frequent label from a cluster

    Args:
        a (grouped df): cluster df

    Returns:
        int: label with the most frequent location label
    """
    if a.size > 0:
        cnt = Counter(a)
        return cnt.most_common(1)[0][0]
    return None


def compute_intervals(centroids, labels, timestamps, accuracy):
    """If the label is -1 it means that the point doesn't belong to any cluster. Otherwise there should be at least 2 points for a stop locations and they should assert (len(centroids) == len(community_labels))

    Args:
        centroids (list): List with coordinate tuple
        labels (int): Stop label
        timestamps (date): Timestamp of ping
        accuracy (float): Accuracy associated to the GPS point

    Returns:
        list: list with consecutive locations of a users
    """
    i = 0
    seen = 0
    trajectory = []
    while i < len(labels):
        if labels[i] == -1:
            i += 1
        else:
            start_index = i
            while (i + 1 < len(labels)) and (labels[i] == labels[i + 1]):
                i += 1
            trajectory.append((timestamps[start_index], timestamps[i], *centroids[seen],
                               np.median(accuracy[start_index: i]), i - start_index + 1))
            seen += 1
            i += 1

    return trajectory


def assert_timestamps_ordered(data, timestamp_col='datetime'):
    """Assert that timestamps in the data frame are in ascending order.

    Args:
        data (data frame): Data frame with geolocated pings with timestamp.
        timestamp_col (str): Name of the column containing the timestamps.
    """
    one_second = pd.Timedelta(seconds=1)
    assert np.all(np.diff(data[timestamp_col]) >= one_second), f"{timestamp_col} must be in ascending order"


def assert_latitude_range(data, lat_col='lat'):
    """Assert that the latitude values in the data frame are within the valid range.

    Args:
        data (data frame): Data frame with geolocated pings with timestamp.
        lat_col (str): Name of the column containing the latitude values.
    """
    assert (np.min(data[lat_col]) > -90 and np.max(data[lat_col]) <
            90), f"{lat_col} must have values between -90 and 90 degrees"


def assert_longitude_range(data, lon_col='lon'):
    """Assert that the longitude values in the data frame are within the valid range.

    Args:
        data (data frame): Data frame with geolocated pings with timestamp.
        lon_col (str): Name of the column containing the longitude values.
    """
    assert (np.min(data[lon_col]) > -180 and np.max(data[lon_col]) <
            180), f"{lon_col} must have values between -180 and 180 degrees"

def data_assertions(data):
  assert np.all(data[:-1, 2] <= data[1:, 2]), "Timestamps must be ordered"
  assert (np.min(data[:, 0]) > -90 and np.max(data[:, 0]) < 90),         "lat (column 0) must have values between -90 and 90"
  assert (np.min(data[:, 1]) > -180 and np.max(data[:, 1]) < 180),    "lon (column 1) must have values between -180 and 180"

#  def run_infostop(data, r1=50, min_staying_time=300, min_size=2, max_time_between=3600, distance_metric='haversine'):
#      """Apply Infostop algorithm to a set of pings.
#
#      Args:
#          data (data frame): Data frame with geolocated pings with timestamp.
#          r1 (float): Radius of maximum distance between pings.
#          min_staying_time (float): Minimum time of consecutive pings inside a radius to be considered a stop.
#          min_size (int): Number of pings to consider a stop candidate.
#          max_time_between (float): Maximum time between two consecutive pings.
#          distance_metric (str): Metric to measure distance.
#
#      Returns:
#          data frame: Data frame with pings and labeled stops including centroids of stops.
#      """
#      data_assertions(data)
#
#      centroids, stat_labels = get_stationary_events(
#          data[:, :3], r1, min_size, min_staying_time, max_time_between, distance_metric)
#      return centroids, stat_labels #compute_intervals(centroids, stat_labels, data[:, 2], data[:, 3], data)

def run_infostop(data, r1=50, min_staying_time=300, min_size=2, max_time_between=3600, distance_metric='haversine'):
    """Apply Infostop algorithm to a set of pings.

    Args:
        data (data frame): Data frame with geolocated pings with timestamp.
        r1 (float): Radius of maximum distance between pings.
        min_staying_time (float): Minimum time of consecutive pings inside a radius to be considered a stop.
        min_size (int): Number of pings to consider a stop candidate.
        max_time_between (float): Maximum time between two consecutive pings.
        distance_metric (str): Metric to measure distance.

    Returns:
        data frame: Data frame with pings and labeled stops including centroids of stops.
    """
    data_assertions(data)

    centroids, stat_labels = get_stationary_events(
        data[:, :3], r1, min_size, min_staying_time, max_time_between, distance_metric)
    return compute_intervals(centroids, stat_labels, data[:, 2], data[:, 3])


def to_unix_int(date):
    """converts str date to UNIX Time

    Args:
        date (date): ymd

    Returns:
        int: UNIX date
    """
    return int(date.replace(tzinfo=timezone.utc).timestamp())

def get_stop_location(df, group_col, ordered_col):
    #  import pandas as pd
    def pandas_stop_location(key, data):
        identifier =  data['uid'].values[0]
        res = run_infostop(data[["latitude", "longitude", 'epoch_time', "horizontal_accuracy"]].values, r1=50, min_staying_time=300, min_size=2, max_time_between=3600, distance_metric='haversine')

        df = pd.DataFrame(res, columns=["t_start",  "t_end", "latitude", "longitude", "median_accuracy", "total_pings_stop"])
        df = df[df['median_accuracy'] < 200]
        df['uid'] = identifier
        if not df.empty:
              db = DBSCAN(eps=3.1392246115209545e-05, min_samples=1, metric='haversine', algorithm='ball_tree').fit(np.radians(df[['latitude', 'longitude']].values)) # notice that we don't have noise here, since any point that we consider is a stop location and hence has been already pre filtered by run_infostop (min_samples = 1 => no label =-1)
              df['cluster_label'] = db.labels_
        else:
          df['cluster_label'] = None
        return df
        #  pass

    schema = StructType([
        StructField("t_start", FloatType(), False),
        StructField("t_end", FloatType(), False),
        StructField("latitude", FloatType(), False),
        StructField("longitude", FloatType(), False),
        StructField("median_accuracy", FloatType(), False),
        StructField("total_pings_stop", IntegerType(), False),
        StructField("uid", StringType(), False),
        StructField("cluster_label", IntegerType(), False)
    ])

    return df.orderBy(ordered_col).groupBy(group_col).applyInPandas(pandas_stop_location, schema=schema)

def _to_unix_int(dt):
    return int(dt.timestamp())

def create_date_list():
    @pandas_udf('array<struct<t_start:long,t_end:long>>', PandasUDFType.SCALAR)
    def pandas_make_list(start: pd.Series, end: pd.Series) -> pd.Series:
        #  def _to_unix_int(dt):
        #      return int(dt.value // 10**9)

        result = []
        for s, e in zip(start, end):
            s = pd.to_datetime(s, unit='s')
            e = pd.to_datetime(e, unit='s')

            parts = pd.date_range(s, e, freq='d', normalize=True).tolist()
            if parts:
                # Update the first timestamp
                parts[0] = s
                # Append the end timestamp if it is not midnight
                if parts[-1] != e:
                    parts.append(e)
                unix_parts = [int(x.value // 10**9) for x in parts]
                res = [(unix_parts[i], unix_parts[i+1]) for i in range(len(unix_parts) - 1)]
                result.append(res)
            else:
                result.append([(_to_unix_int(s), _to_unix_int(e))])

        return pd.Series(result)

    return pandas_make_list

def get_stop_cluster(current, sl, group_col, db_scan_radius=3.1392246115209545e-05):
    # group_col = 'uid'
    def pandas_stop_cluster(key, data):
        if not data.empty:
            db = DBSCAN(eps=db_scan_radius, min_samples=1, metric='haversine',
                        algorithm='ball_tree').fit(np.radians(data[['latitude',
                                                                    'longitude']].values))
            data['cluster_label'] = db.labels_
        else:
            data['cluster_label'] = None
        return data

    schema_cluster_df = StructType([
        StructField('user_id', StringType(), False),
        StructField('lat', DoubleType(), False),
        StructField('lon', DoubleType(), False),
        StructField('cluster_label', LongType(), True),
        StructField('median_accuracy', DoubleType(), True),
        StructField('total_pings_stop', LongType(), True),
        StructField('total_duration_stop_location', LongType(), True),
        StructField('t_start', LongType(), False),
        StructField('t_end', LongType(), False),
        StructField('duration', LongType(), False),
    ])

    df = current.union(sl)

    return df.groupBy(group_col).applyInPandas(pandas_stop_cluster, schema=schema_cluster_df)

