import re

with open('benchmark_100_queries.py', 'r') as f:
    content = f.read()

new_content = "import os\nos.environ['no_proxy'] = '127.0.0.1,localhost'\nos.environ['NO_PROXY'] = '127.0.0.1,localhost'\n" + content

with open('benchmark_100_queries.py', 'w') as f:
    f.write(new_content)
