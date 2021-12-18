subscribers = dict()


def subscribe(event_type: str, function):
    if event_type in subscribers:
        subscribers[event_type].append(function)
    else:
        subscribers[event_type] = []


def post_event(event_type: str, data):
    if event_type in subscribers:
        for function in subscribers[event_type]:
            function(data)
