"""
LLM interface for agents
"""
import os
import openai
from typing import List, Dict, Any, Optional
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMInterface:
    """Interface for interacting with language models"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key found. Using mock responses.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            openai.api_key = self.api_key
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_tokens: int = 1000, 
                         temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        if self.mock_mode:
            return self._generate_mock_response(messages)
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_mock_response(messages)
    
    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response when API is not available"""
        last_message = messages[-1]["content"].lower()
        
        # Simple rule-based mock responses
        if "plan" in last_message or "steps" in last_message:
            return """Based on the request, here's a suggested plan:
1. Analyze the input requirements
2. Break down the task into manageable components
3. Identify necessary resources and dependencies
4. Execute each step systematically
5. Review and validate results"""
        
        elif "summarize" in last_message or "summary" in last_message:
            return """Here's a summary of the key points:
- Main topic addresses important aspects of the subject
- Several key findings were identified
- The analysis reveals significant patterns and trends
- Recommendations include strategic approaches for improvement
- Further investigation may be beneficial in specific areas"""
        
        elif "question" in last_message or "?" in last_message:
            return """Based on my analysis, here are the key points to consider:
- The question touches on important aspects that require careful consideration
- Multiple factors contribute to the overall understanding
- Evidence suggests that several approaches could be effective
- It's important to consider both immediate and long-term implications
- Additional context would help provide more specific recommendations"""
        
        else:
            return """I understand your request. Based on the information provided:
- I've analyzed the key components of your query
- The situation requires a multi-faceted approach
- Several factors should be considered in the solution
- I recommend proceeding with a systematic methodology
- Please let me know if you need more specific details on any aspect"""
    
    def create_system_prompt(self, agent_type: str, agent_name: str) -> str:
        """Create system prompt for different agent types"""
        base_prompt = f"You are {agent_name}, a specialized AI agent in a multi-agent system. "
        
        if agent_type == "planning":
            return base_prompt + """Your role is to break down complex tasks into manageable steps and create actionable plans. 
Focus on:
- Task decomposition
- Resource identification
- Timeline estimation
- Risk assessment
- Clear, sequential steps"""
        
        elif agent_type == "summarization":
            return base_prompt + """Your role is to create clear, concise summaries of documents and information.
Focus on:
- Key points extraction
- Main themes identification
- Important details preservation
- Logical structure
- Actionable insights"""
        
        elif agent_type == "qa":
            return base_prompt + """Your role is to provide detailed, accurate answers to questions.
Focus on:
- Comprehensive analysis
- Evidence-based responses
- Clear explanations
- Multiple perspectives
- Practical recommendations"""
        
        elif agent_type == "coordinator":
            return base_prompt + """Your role is to coordinate between different agents and manage the overall workflow.
Focus on:
- Task delegation
- Result synthesis
- Communication facilitation
- Quality assurance
- Final output coordination"""
        
        else:
            return base_prompt + "Your role is to assist with various tasks as needed."
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost based on token usage"""
        # Approximate pricing for GPT-3.5-turbo (as of 2024)
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
