# Assignment 3.2: Multi-Agent LLM System

## Overview
This assignment implements a multi-agent system where different LLM agents collaborate to handle complex tasks. Each agent specializes in a specific subtask:

- **Planning Agent**: Breaks down complex queries into subtasks
- **Summarization Agent**: Summarizes long texts and documents
- **QA Agent**: Answers specific questions and provides detailed responses
- **Coordinator Agent**: Orchestrates communication between agents

## System Architecture
The multi-agent system uses message passing for communication between agents, with a shared memory system for storing intermediate results and context.

## Features
- Modular agent design with specialized capabilities
- Message passing communication protocol
- Shared memory for context storage
- Conversation history tracking
- Task delegation and result aggregation

## Directory Structure
```
assignment-3.2-multi-agent-system/
├── README.md
├── requirements.txt
├── multi_agent_system.ipynb          # Main notebook with examples
├── main.py                           # Command-line interface
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Base agent class
│   ├── planning_agent.py             # Task planning agent
│   ├── summarization_agent.py        # Text summarization agent
│   ├── qa_agent.py                   # Question answering agent
│   └── coordinator_agent.py          # System coordinator
├── utils/
│   ├── __init__.py
│   ├── message_system.py             # Message passing system
│   ├── shared_memory.py              # Shared memory implementation
│   └── llm_interface.py              # LLM API interface
└── data/
    └── sample_documents.json         # Sample data for testing
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
Open `multi_agent_system.ipynb` to see interactive examples and demonstrations.

### Command Line
```bash
python main.py "What are the key insights from the attached documents and how should we proceed?"
```

## Example Scenarios
1. **Document Analysis**: Upload multiple documents, get summaries and key insights
2. **Research Planning**: Break down complex research questions into actionable steps
3. **Multi-step QA**: Answer complex questions requiring multiple reasoning steps

## Configuration
Set your OpenAI API key in environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Architecture Details
- Agents communicate through structured messages
- Shared memory maintains conversation context
- Coordinator routes messages and aggregates results
- Each agent has specialized prompts and capabilities
