from datetime import datetime, timezone


def get_date_time_from_float(float_time: float):
    """
    Convert a float time to a datetime object
    """
    dt_object = datetime.fromtimestamp(float_time)
    dt_str = datetime.strftime(dt_object, "%Y-%m-%d %H:%M:%S.%f")
    return dt_str


def convert_utc_to_local_time(utc_time: str):
    """
    Convert a UTC time to a local time
    """
    utc_time = datetime.strptime(utc_time, "%Y%m%d_%H%M%S_%f")
    local_time = utc_time.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return local_time