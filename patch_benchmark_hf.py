with open('benchmark_100_queries.py', 'r') as f:
    content = f.read()

new_content = "import os\nos.environ['SENTENCE_TRANSFORMERS_HOME'] = '/Users/zesan/.cache/huggingface'\n" + content

with open('benchmark_100_queries.py', 'w') as f:
    f.write(new_content)
