"""
Message system for agent communication
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Message:
    """Message structure for agent communication"""
    id: str
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    priority: int = 1
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class MessageQueue:
    """Message queue for handling agent communication"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.processed_messages: List[Message] = []
    
    def send_message(self, sender: str, receiver: str, message_type: str, 
                    content: Dict[str, Any], priority: int = 1) -> str:
        """Send a message to the queue"""
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            priority=priority
        )
        
        # Insert message based on priority (higher priority first)
        inserted = False
        for i, msg in enumerate(self.messages):
            if message.priority > msg.priority:
                self.messages.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            self.messages.append(message)
        
        return message_id
    
    def get_messages_for_agent(self, agent_name: str) -> List[Message]:
        """Get all pending messages for a specific agent"""
        agent_messages = [msg for msg in self.messages if msg.receiver == agent_name]
        # Remove from pending queue
        self.messages = [msg for msg in self.messages if msg.receiver != agent_name]
        # Add to processed
        self.processed_messages.extend(agent_messages)
        return agent_messages
    
    def get_message_history(self, agent_name: Optional[str] = None) -> List[Message]:
        """Get message history, optionally filtered by agent"""
        if agent_name:
            return [msg for msg in self.processed_messages 
                   if msg.sender == agent_name or msg.receiver == agent_name]
        return self.processed_messages
    
    def clear_history(self):
        """Clear all message history"""
        self.processed_messages.clear()


class MessageBus:
    """Central message bus for coordinating agent communication"""
    
    def __init__(self):
        self.queue = MessageQueue()
        self.agents: Dict[str, Any] = {}
        self.conversation_history: List[Dict] = []
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent with the message bus"""
        self.agents[agent_name] = agent_instance
        print(f"Agent '{agent_name}' registered with message bus")
    
    def broadcast_message(self, sender: str, message_type: str, 
                         content: Dict[str, Any], priority: int = 1):
        """Broadcast message to all agents except sender"""
        for agent_name in self.agents:
            if agent_name != sender:
                self.queue.send_message(sender, agent_name, message_type, content, priority)
    
    def send_to_agent(self, sender: str, receiver: str, message_type: str,
                     content: Dict[str, Any], priority: int = 1) -> str:
        """Send message to specific agent"""
        return self.queue.send_message(sender, receiver, message_type, content, priority)
    
    def deliver_messages(self, agent_name: str) -> List[Message]:
        """Deliver pending messages to an agent"""
        return self.queue.get_messages_for_agent(agent_name)
    
    def add_to_conversation(self, role: str, content: str, agent: str = None):
        """Add entry to conversation history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "agent": agent
        }
        self.conversation_history.append(entry)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = "Conversation Summary:\n"
        for entry in self.conversation_history[-10:]:  # Last 10 entries
            agent_info = f" [{entry['agent']}]" if entry['agent'] else ""
            summary += f"{entry['role']}{agent_info}: {entry['content'][:100]}...\n"
        
        return summary
