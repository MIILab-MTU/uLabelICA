First use setup.py
    python 3.6  ##except 3.6.0
    pip 21.2.2 
    pip install . 
    pip install -r requirements.txt

File "miniconda3/envs/labelme-custom/lib/python3.6/site-packages/keras/engine/saving.py", line 1004, in load_weights_from_hdf5_group 
original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'

Solution: Remove .decode('utf8') at two places.
