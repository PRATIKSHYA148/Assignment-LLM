"""
Summarization agent for text summarization and content extraction
"""
from typing import Dict, List, Any
from agents.base_agent import BaseAgent
from utils.message_system import Message


class SummarizationAgent(BaseAgent):
    """Agent specialized in text summarization and content extraction"""
    
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="SummarizationAgent",
            agent_type="summarization",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )
        
        # Summarization-specific attributes
        self.summary_cache = {}
        self.summarization_styles = {
            "brief": "Provide a concise 2-3 sentence summary highlighting only the most critical points.",
            "detailed": "Create a comprehensive summary covering all major points, key details, and implications.",
            "bullet": "Present the summary as clear bullet points organized by topic or importance.",
            "executive": "Write an executive summary suitable for decision-makers, focusing on insights and recommendations."
        }
    
    def _handle_task_request(self, message: Message):
        """Handle summarization requests"""
        content = message.content
        text_to_summarize = content.get("text", "")
        documents = content.get("documents", [])
        style = content.get("style", "detailed")
        focus_areas = content.get("focus_areas", [])
        
        if text_to_summarize:
            summary = self.summarize_text(text_to_summarize, style, focus_areas)
        elif documents:
            summary = self.summarize_documents(documents, style, focus_areas)
        else:
            summary = {"error": "No text or documents provided for summarization"}
        
        # Store summary in shared memory
        summary_id = f"summary_{message.id}"
        self.store_in_memory(summary_id, summary, ["summary", style])
        self.summary_cache[summary_id] = summary
        
        # Send summary back to requester
        self.send_message(
            message.sender,
            "summary_response",
            {
                "summary_id": summary_id,
                "summary": summary,
                "style": style
            }
        )
        
        # Notify coordinator about completed summary
        self.send_message(
            "CoordinatorAgent",
            "summary_completed",
            {
                "summary_id": summary_id,
                "requester": message.sender,
                "word_count": len(summary.get("content", "").split()) if isinstance(summary.get("content"), str) else 0
            }
        )
    
    def summarize_text(self, text: str, style: str = "detailed", focus_areas: List[str] = None) -> Dict[str, Any]:
        """Summarize a single text"""
        
        # Create summarization prompt
        style_instruction = self.summarization_styles.get(style, self.summarization_styles["detailed"])
        focus_instruction = ""
        
        if focus_areas:
            focus_instruction = f"\nPay special attention to these areas: {', '.join(focus_areas)}"
        
        prompt = f"""
        Please summarize the following text:
        
        {text}
        
        Style: {style_instruction}{focus_instruction}
        
        Also provide:
        1. Key themes or topics
        2. Important entities (people, organizations, places)
        3. Main insights or conclusions
        4. Any actionable items or recommendations
        """
        
        # Generate summary using LLM
        response = self.generate_llm_response(prompt)
        
        # Parse and structure the response
        summary = self._parse_summary_response(response, text, style)
        
        return summary
    
    def summarize_documents(self, documents: List[Dict[str, Any]], style: str = "detailed", 
                          focus_areas: List[str] = None) -> Dict[str, Any]:
        """Summarize multiple documents"""
        
        document_summaries = []
        all_themes = set()
        all_entities = set()
        
        # Summarize each document individually
        for doc in documents:
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            
            if content:
                doc_summary = self.summarize_text(content, "brief", focus_areas)
                doc_summary["title"] = title
                doc_summary["document_id"] = doc.get("id", "unknown")
                document_summaries.append(doc_summary)
                
                # Collect themes and entities
                all_themes.update(doc_summary.get("themes", []))
                all_entities.update(doc_summary.get("entities", []))
        
        # Create overall summary
        combined_text = "\n\n".join([doc.get("content", "") for doc in documents])
        overall_summary = self.summarize_text(combined_text, style, focus_areas)
        
        # Combine results
        summary = {
            "type": "multi_document",
            "style": style,
            "overall_summary": overall_summary,
            "document_summaries": document_summaries,
            "document_count": len(documents),
            "common_themes": list(all_themes),
            "all_entities": list(all_entities),
            "cross_document_insights": self._find_cross_document_insights(document_summaries)
        }
        
        return summary
    
    def _parse_summary_response(self, response: str, original_text: str, style: str) -> Dict[str, Any]:
        """Parse LLM summary response into structured format"""
        
        lines = response.split('\n')
        
        summary = {
            "type": "single_text",
            "style": style,
            "content": "",
            "themes": [],
            "entities": [],
            "insights": [],
            "actionable_items": [],
            "word_count_original": len(original_text.split()),
            "word_count_summary": 0
        }
        
        current_section = "content"
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "key themes" in line.lower() or "themes" in line.lower():
                current_section = "themes"
                summary["content"] = "\n".join(content_lines).strip()
                summary["word_count_summary"] = len(summary["content"].split())
            elif "entities" in line.lower():
                current_section = "entities"
            elif "insights" in line.lower() or "conclusions" in line.lower():
                current_section = "insights"
            elif "actionable" in line.lower() or "recommendations" in line.lower():
                current_section = "actionable_items"
            elif current_section == "content":
                content_lines.append(line)
            elif current_section in ["themes", "entities", "insights", "actionable_items"]:
                if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                    summary[current_section].append(line[1:].strip() if line[0] in '-*•' else line)
        
        # If content wasn't set (no sections found), use all lines as content
        if not summary["content"]:
            summary["content"] = response.strip()
            summary["word_count_summary"] = len(summary["content"].split())
        
        return summary
    
    def _find_cross_document_insights(self, document_summaries: List[Dict[str, Any]]) -> List[str]:
        """Find insights that span across multiple documents"""
        insights = []
        
        # Find common themes
        theme_counts = {}
        for doc_summary in document_summaries:
            for theme in doc_summary.get("themes", []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        common_themes = [theme for theme, count in theme_counts.items() if count > 1]
        if common_themes:
            insights.append(f"Common themes across documents: {', '.join(common_themes)}")
        
        # Find entity relationships
        entity_counts = {}
        for doc_summary in document_summaries:
            for entity in doc_summary.get("entities", []):
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        recurring_entities = [entity for entity, count in entity_counts.items() if count > 1]
        if recurring_entities:
            insights.append(f"Entities mentioned across multiple documents: {', '.join(recurring_entities)}")
        
        return insights
    
    def extract_key_quotes(self, text: str, num_quotes: int = 3) -> List[str]:
        """Extract key quotes or important statements from text"""
        
        prompt = f"""
        Extract the {num_quotes} most important quotes or statements from the following text.
        Choose quotes that best represent the main ideas or provide key insights:
        
        {text}
        
        Return only the quotes, one per line.
        """
        
        response = self.generate_llm_response(prompt)
        quotes = [line.strip().strip('"\'') for line in response.split('\n') if line.strip()]
        
        return quotes[:num_quotes]
    
    def create_abstract(self, text: str, max_words: int = 150) -> str:
        """Create an academic-style abstract"""
        
        prompt = f"""
        Create a concise academic abstract for the following text, limited to {max_words} words.
        Include: purpose, methodology (if applicable), key findings, and conclusions.
        
        {text}
        """
        
        response = self.generate_llm_response(prompt)
        
        # Ensure word limit
        words = response.split()
        if len(words) > max_words:
            response = ' '.join(words[:max_words]) + "..."
        
        return response
    
    def compare_documents(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two documents and identify similarities and differences"""
        
        prompt = f"""
        Compare these two documents and identify:
        1. Similarities in content, themes, or conclusions
        2. Key differences in approach or findings
        3. Complementary information
        4. Conflicting viewpoints (if any)
        
        Document 1: {doc1.get('title', 'Untitled')}
        {doc1.get('content', '')}
        
        Document 2: {doc2.get('title', 'Untitled')}
        {doc2.get('content', '')}
        """
        
        response = self.generate_llm_response(prompt)
        
        comparison = {
            "document1": doc1.get('title', 'Document 1'),
            "document2": doc2.get('title', 'Document 2'),
            "analysis": response,
            "similarity_score": self._estimate_similarity(doc1.get('content', ''), doc2.get('content', ''))
        }
        
        return comparison
    
    def _estimate_similarity(self, text1: str, text2: str) -> float:
        """Estimate similarity between two texts (simple word overlap)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
