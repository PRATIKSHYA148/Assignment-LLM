"""
Shared memory system for multi-agent communication
"""
import json
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Memory entry structure"""
    key: str
    value: Any
    created_by: str
    created_at: str
    last_modified: str
    access_count: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class SharedMemory:
    """Thread-safe shared memory for agent communication"""
    
    def __init__(self):
        self._memory: Dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._access_log: List[Dict] = []
    
    def store(self, key: str, value: Any, agent_name: str, tags: List[str] = None) -> bool:
        """Store a value in shared memory"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            if key in self._memory:
                # Update existing entry
                entry = self._memory[key]
                entry.value = value
                entry.last_modified = timestamp
                entry.access_count += 1
                if tags:
                    entry.tags.extend(tags)
                    entry.tags = list(set(entry.tags))  # Remove duplicates
            else:
                # Create new entry
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    created_by=agent_name,
                    created_at=timestamp,
                    last_modified=timestamp,
                    tags=tags or []
                )
                self._memory[key] = entry
            
            # Log access
            self._log_access("store", key, agent_name)
            return True
    
    def retrieve(self, key: str, agent_name: str) -> Optional[Any]:
        """Retrieve a value from shared memory"""
        with self._lock:
            if key in self._memory:
                entry = self._memory[key]
                entry.access_count += 1
                self._log_access("retrieve", key, agent_name)
                return entry.value
            return None
    
    def search_by_tags(self, tags: List[str], agent_name: str) -> Dict[str, Any]:
        """Search memory entries by tags"""
        with self._lock:
            results = {}
            for key, entry in self._memory.items():
                if any(tag in entry.tags for tag in tags):
                    results[key] = entry.value
                    entry.access_count += 1
            
            self._log_access("search", f"tags:{','.join(tags)}", agent_name)
            return results
    
    def get_all_keys(self, agent_name: str = None) -> List[str]:
        """Get all keys in memory, optionally filtered by creator"""
        with self._lock:
            if agent_name:
                return [key for key, entry in self._memory.items() 
                       if entry.created_by == agent_name]
            return list(self._memory.keys())
    
    def delete(self, key: str, agent_name: str) -> bool:
        """Delete an entry from memory"""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                self._log_access("delete", key, agent_name)
                return True
            return False
    
    def update_tags(self, key: str, tags: List[str], agent_name: str) -> bool:
        """Update tags for an entry"""
        with self._lock:
            if key in self._memory:
                self._memory[key].tags = tags
                self._memory[key].last_modified = datetime.now().isoformat()
                self._log_access("update_tags", key, agent_name)
                return True
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self._lock:
            total_entries = len(self._memory)
            total_accesses = sum(entry.access_count for entry in self._memory.values())
            
            # Most accessed entries
            most_accessed = sorted(
                [(key, entry.access_count) for key, entry in self._memory.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Entries by creator
            creators = {}
            for entry in self._memory.values():
                creators[entry.created_by] = creators.get(entry.created_by, 0) + 1
            
            return {
                "total_entries": total_entries,
                "total_accesses": total_accesses,
                "most_accessed": most_accessed,
                "entries_by_creator": creators,
                "recent_access_log": self._access_log[-10:]
            }
    
    def _log_access(self, operation: str, key: str, agent_name: str):
        """Log memory access"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "key": key,
            "agent": agent_name
        }
        self._access_log.append(log_entry)
        
        # Keep only last 1000 log entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
    
    def export_memory(self) -> Dict[str, Any]:
        """Export memory contents for persistence"""
        with self._lock:
            export_data = {}
            for key, entry in self._memory.items():
                export_data[key] = {
                    "value": entry.value,
                    "created_by": entry.created_by,
                    "created_at": entry.created_at,
                    "last_modified": entry.last_modified,
                    "access_count": entry.access_count,
                    "tags": entry.tags
                }
            return export_data
    
    def import_memory(self, data: Dict[str, Any], importing_agent: str):
        """Import memory contents from exported data"""
        with self._lock:
            for key, entry_data in data.items():
                entry = MemoryEntry(
                    key=key,
                    value=entry_data["value"],
                    created_by=entry_data["created_by"],
                    created_at=entry_data["created_at"],
                    last_modified=entry_data["last_modified"],
                    access_count=entry_data["access_count"],
                    tags=entry_data["tags"]
                )
                self._memory[key] = entry
            
            self._log_access("import", f"{len(data)} entries", importing_agent)
    
    def clear(self, agent_name: str = None):
        """Clear memory, optionally only entries created by specific agent"""
        with self._lock:
            if agent_name:
                # Remove only entries created by specific agent
                keys_to_remove = [key for key, entry in self._memory.items() 
                                if entry.created_by == agent_name]
                for key in keys_to_remove:
                    del self._memory[key]
            else:
                # Clear all memory
                self._memory.clear()
            
            self._log_access("clear", f"agent:{agent_name}" if agent_name else "all", 
                           agent_name or "system")
