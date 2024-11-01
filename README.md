# Python Documentation Generator

A tool that automatically analyzes Python codebases and generates comprehensive documentation using AI.

## Dependencies

```txt
# requirements.txt
python-dotenv>=1.0.0
openai>=1.0.0
aiohttp>=3.8.0
tqdm>=4.65.0
pyyaml>=6.0.1
black>=23.0.0
pydantic>=2.0.0
```

## Project Structure
```
documentation-generator/
├── .env
├── .gitignore
├── config.yaml
├── requirements.txt
├── main.py
├── cache.py
├── config.py
├── documentation.py
├── extract.py
├── utils.py
└── workflow.py
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd documentation-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here

# Optional Azure Configuration
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

5. Create initial configuration:
```bash
python main.py --create-config
```

This will create a `config.yaml` file that you can customize:
```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4-0125-preview
  max_tokens: 6000
  temperature: 0.2

cache:
  enabled: true
  directory: .cache
  max_size_mb: 100
  ttl_hours: 24

exclude_dirs:
  - .git
  - .github
  - __pycache__
  - venv
  - .cache

concurrency_limit: 5
```

## Usage

### Basic Usage

1. Analyze a local directory:
```bash
python main.py /path/to/your/python/project output.md
```

2. Analyze a GitHub repository:
```bash
python main.py https://github.com/username/repo output.md
```

### Advanced Options

- Specify concurrency limit:
```bash
python main.py /path/to/project output.md --concurrency 10
```

- Use Azure OpenAI:
```bash
python main.py /path/to/project output.md --service azure
```

- Use custom config file:
```bash
python main.py /path/to/project output.md --config my-config.yaml
```

### Cache Management

The tool automatically caches analysis results in the `.cache` directory. To clear the cache:

1. Delete the `.cache` directory
2. Disable caching in config.yaml:
```yaml
cache:
  enabled: false
```

## Output Format

The tool generates a markdown file with:
1. Module summaries
2. Function/method documentation
3. Complexity analysis
4. Updated source code with docstrings

Example output structure:
```markdown
# Code Documentation Analysis

## path/to/file.py

### Summary
Brief description of the module...

### Functions and Methods
| Name | Type | Complexity | Description |
|------|------|------------|-------------|
| function_name | Function | 3 🟢 | Description... |

### Source Code
```python
# Source code with updated docstrings
```
```

## Environment Variables

Required:
- `OPENAI_API_KEY`: Your OpenAI API key

Optional (for Azure):
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure endpoint URL
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Azure deployment name

## Common Issues and Solutions

1. Rate Limiting
   - Decrease concurrency limit in config.yaml
   - The tool automatically implements exponential backoff

2. Memory Usage
   - Adjust cache size in config.yaml
   - Process smaller batches of files

3. Parsing Errors
   - The tool automatically attempts to format problematic files using black
   - Check file encoding (use UTF-8)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License