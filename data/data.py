import json

def getData():
    try:
        with open('./data/data.json', 'r') as data:
            data = data.read()
        files = json.loads(data)
        return files
    except:
        return 0