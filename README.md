python 3.6  ##except 3.6.0
pip 21.2.2 
pip install . 
pip install -r requirements.txt

AttributeError: 'str' object has no attribute 'decode'
Solution: Remove .decode('utf8') at two places.
