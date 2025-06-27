"""
Base agent class for the multi-agent system
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils.message_system import MessageBus, Message
from utils.shared_memory import SharedMemory
from utils.llm_interface import LLMInterface


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, name: str, agent_type: str, message_bus: MessageBus, 
                 shared_memory: SharedMemory, llm_interface: LLMInterface):
        self.name = name
        self.agent_type = agent_type
        self.message_bus = message_bus
        self.shared_memory = shared_memory
        self.llm_interface = llm_interface
        self.conversation_history = []
        self.status = "idle"
        
        # Register with message bus
        self.message_bus.register_agent(self.name, self)
        
        # Create system prompt
        self.system_prompt = self.llm_interface.create_system_prompt(
            self.agent_type, self.name
        )
    
    def process_messages(self):
        """Process all pending messages for this agent"""
        messages = self.message_bus.deliver_messages(self.name)
        for message in messages:
            self._handle_message(message)
    
    def _handle_message(self, message: Message):
        """Handle incoming message"""
        print(f"[{self.name}] Received message: {message.message_type} from {message.sender}")
        
        # Store message in conversation history
        self.conversation_history.append({
            "type": "received",
            "from": message.sender,
            "content": message.content,
            "timestamp": message.timestamp
        })
        
        # Process based on message type
        if message.message_type == "task_request":
            self._handle_task_request(message)
        elif message.message_type == "information_request":
            self._handle_information_request(message)
        elif message.message_type == "result_sharing":
            self._handle_result_sharing(message)
        else:
            self._handle_custom_message(message)
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any], priority: int = 1):
        """Send message to another agent"""
        message_id = self.message_bus.send_to_agent(
            self.name, receiver, message_type, content, priority
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "type": "sent",
            "to": receiver,
            "content": content,
            "timestamp": message_id,
            "message_id": message_id
        })
        
        return message_id
    
    def broadcast_message(self, message_type: str, content: Dict[str, Any], priority: int = 1):
        """Broadcast message to all other agents"""
        self.message_bus.broadcast_message(self.name, message_type, content, priority)
    
    def store_in_memory(self, key: str, value: Any, tags: List[str] = None):
        """Store information in shared memory"""
        return self.shared_memory.store(key, value, self.name, tags)
    
    def retrieve_from_memory(self, key: str) -> Any:
        """Retrieve information from shared memory"""
        return self.shared_memory.retrieve(key, self.name)
    
    def search_memory_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Search shared memory by tags"""
        return self.shared_memory.search_by_tags(tags, self.name)
    
    def generate_llm_response(self, user_input: str, context: str = "") -> str:
        """Generate response using LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": user_input})
        
        self.status = "processing"
        response = self.llm_interface.generate_response(messages)
        self.status = "idle"
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "type": self.agent_type,
            "status": self.status,
            "messages_processed": len(self.conversation_history),
            "memory_entries": len(self.shared_memory.get_all_keys(self.name))
        }
    
    @abstractmethod
    def _handle_task_request(self, message: Message):
        """Handle task request - must be implemented by subclasses"""
        pass
    
    def _handle_information_request(self, message: Message):
        """Handle information request - can be overridden"""
        content = message.content
        if "query" in content:
            response = self.generate_llm_response(content["query"])
            self.send_message(
                message.sender,
                "information_response",
                {"query": content["query"], "response": response}
            )
    
    def _handle_result_sharing(self, message: Message):
        """Handle result sharing - can be overridden"""
        content = message.content
        if "result" in content:
            # Store result in shared memory
            key = f"result_{message.sender}_{content.get('task_id', 'unknown')}"
            self.store_in_memory(key, content["result"], ["result", message.sender])
    
    def _handle_custom_message(self, message: Message):
        """Handle custom message types - can be overridden"""
        print(f"[{self.name}] Received unknown message type: {message.message_type}")
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return f"[{self.name}] No conversation history"
        
        summary = f"[{self.name}] Conversation Summary (last 5 interactions):\n"
        for entry in self.conversation_history[-5:]:
            if entry["type"] == "sent":
                summary += f"  → To {entry['to']}: {str(entry['content'])[:50]}...\n"
            else:
                summary += f"  ← From {entry['from']}: {str(entry['content'])[:50]}...\n"
        
        return summary
