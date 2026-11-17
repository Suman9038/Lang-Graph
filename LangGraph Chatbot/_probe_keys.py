import requests

for key in ("JL20KU7U29PYDDR", "8JL20KU7U29PYDDR"):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=MSFT&apikey={key}"
    try:
        data = requests.get(url, timeout=20).json()
        if "Global Quote" in data and data["Global Quote"]:
            q = data["Global Quote"]
            print(f"{key}: LIVE  {q.get('01. symbol')} price={q.get('05. price')} day={q.get('07. latest trading day')}")
        else:
            print(f"{key}: NOT LIVE -> {data}")
    except Exception as exc:
        print(f"{key}: request failed: {exc}")
