import subprocess

try:
    script_path = '/home/weihuazhou/Desktop/automated-labelme/labelme/core/graph_generation/data_graph_generation.py'
    subprocess.run(['python',script_path])
except Exception as e:
    print(e)
