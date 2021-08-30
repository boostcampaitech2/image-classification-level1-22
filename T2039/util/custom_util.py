from pytz import timezone
from datetime import datetime as dt

def get_now_str():
    return dt.now().astimezone(timezone("Asia/Seoul")).strftime('%Y%m%d%H%M%S')