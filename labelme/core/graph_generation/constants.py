

DEBUG = False

#https://www.pngwing.com/en/free-png-tnfnv
color_map = {"RI": [255, 0, 0], "OTHER": [255, 0, 0],
             "LAD": [255, 255, 0],
             "LCX": [102, 255, 102],
             "LMA": [0, 102, 255],
             "D1": [255, 0, 255], "D2": [255, 0, 255], "D3": [255, 0, 255],
             "SEP1": [102, 0, 255], "SEP2": [102, 0, 255], "SEP3": [102, 0, 255], "SEP": [102, 0, 255],
             "OM1": [102, 255, 255], "OM2": [102, 255, 255], "OM3": [102, 255, 255], "OM4": [102, 255, 255]}

semantic_mapping = {"OTHER": [255, 0, 0],
                    "LAD": [255, 255, 0],
                    "LCX": [102, 255, 102],
                    "LMA": [0, 102, 255],
                    "D": [255, 0, 255],
                    "SEP": [102, 0, 255],
                    "OM": [102, 255, 255]}


search_sequence = ["SEP", "D", "OM", "LAD", "LCX", "LMA"]
artery_keep_dict = ["SEP1", "D1", "OM1", "LAD", "LCX", "LMA"]
artery_keep_dict_tw = ["D1", "D2","D3", "OM1", "OM2", "OM3", "OM4", "LAD", "LCX", "LMA"]

VESSEL_OTHER_COLOR = [255, 0, 0]
VESSEL_OTHER_COLOR_KEY = "OTHER"