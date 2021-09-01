import json
import sys
import random
import requests

class SlackNoti:
    def __init__(self, url):
        self.url = url
        self.title = (f"New Incoming Message :zap:")
        self.slack_data = {
            "username": "NotificationBot",
            "icon_emoji": ":satellite:",
            #"channel" : "#somerandomcahnnel",
            "attachments": [
                {
                    "color": "#9733EE",
                    "fields": [
                        {
                            "title": self.title,
                            "value": '',
                            "short": "false",
                        }
                    ]
                }
            ]
        }

    def send_message(self, msg):
        self.slack_data['attachments'][0]['fields'][0]['value'] = msg

        byte_length = str(sys.getsizeof(self.slack_data))
        headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
        response = requests.post(self.url, data=json.dumps(self.slack_data), headers=headers)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)        
