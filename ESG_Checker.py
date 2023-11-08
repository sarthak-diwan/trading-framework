import json
class ESG_Checker:
    def __init__(self, file_path="/home/sarthak/btp/data.json") -> None:
        with open(file_path) as f:
            data = json.load(f)
        pass