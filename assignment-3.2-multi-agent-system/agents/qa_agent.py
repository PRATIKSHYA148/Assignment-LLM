"""
Question Answering agent for detailed responses and analysis
"""
from typing import Dict, List, Any
from agents.base_agent import BaseAgent
from utils.message_system import Message


class QAAgent(BaseAgent):
    """Agent specialized in question answering and detailed analysis"""
    
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="QAAgent",
            agent_type="qa",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )
        
        # QA-specific attributes
        self.qa_cache = {}
        self.knowledge_base = {}
        self.question_types = {
            "factual": "Provide specific, fact-based answers with evidence.",
            "analytical": "Analyze the question deeply, considering multiple perspectives and implications.",
            "comparative": "Compare different options, approaches, or viewpoints systematically.",
            "explanatory": "Explain concepts, processes, or relationships in clear, understandable terms.",
            "evaluative": "Evaluate and assess based on criteria, providing reasoned judgments."
        }
    
    def _handle_task_request(self, message: Message):
        """Handle question answering requests"""
        content = message.content
        question = content.get("question", "")
        context = content.get("context", "")
        question_type = content.get("type", "analytical")
        sources = content.get("sources", [])
        
        if not question:
            answer = {"error": "No question provided"}
        else:
            answer = self.answer_question(question, context, question_type, sources)
        
        # Store answer in shared memory
        qa_id = f"qa_{message.id}"
        self.store_in_memory(qa_id, answer, ["qa", question_type])
        self.qa_cache[qa_id] = answer
        
        # Send answer back to requester
        self.send_message(
            message.sender,
            "qa_response",
            {
                "qa_id": qa_id,
                "question": question,
                "answer": answer,
                "type": question_type
            }
        )
        
        # Notify coordinator about completed QA
        self.send_message(
            "CoordinatorAgent",
            "qa_completed",
            {
                "qa_id": qa_id,
                "requester": message.sender,
                "confidence": answer.get("confidence", 0.5)
            }
        )
    
    def answer_question(self, question: str, context: str = "", question_type: str = "analytical", 
                       sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question with detailed analysis"""
        
        # Detect question type if not specified
        if question_type == "auto":
            question_type = self._detect_question_type(question)
        
        # Prepare context from sources
        source_context = self._prepare_source_context(sources or [])
        full_context = f"{context}\n\n{source_context}".strip()
        
        # Create QA prompt
        type_instruction = self.question_types.get(question_type, self.question_types["analytical"])
        
        prompt = f"""
        Question: {question}
        
        Context: {full_context}
        
        Instructions: {type_instruction}
        
        Please provide:
        1. A direct answer to the question
        2. Supporting evidence or reasoning
        3. Alternative perspectives (if applicable)
        4. Confidence level in your answer
        5. Any limitations or assumptions
        6. Follow-up questions that might be relevant
        """
        
        # Generate answer using LLM
        response = self.generate_llm_response(prompt, full_context)
        
        # Parse and structure the response
        answer = self._parse_qa_response(response, question, question_type, sources)
        
        return answer
    
    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question based on its content"""
        question_lower = question.lower()
        
        # Question word analysis
        if any(word in question_lower for word in ["what is", "who is", "when", "where"]):
            return "factual"
        elif any(word in question_lower for word in ["why", "how does", "analyze", "explain"]):
            return "analytical"
        elif any(word in question_lower for word in ["compare", "contrast", "difference", "versus"]):
            return "comparative"
        elif any(word in question_lower for word in ["explain", "describe", "how to"]):
            return "explanatory"
        elif any(word in question_lower for word in ["evaluate", "assess", "judge", "rate"]):
            return "evaluative"
        else:
            return "analytical"  # default
    
    def _prepare_source_context(self, sources: List[Dict[str, Any]]) -> str:
        """Prepare context from provided sources"""
        if not sources:
            return ""
        
        context_parts = []
        for i, source in enumerate(sources):
            title = source.get("title", f"Source {i+1}")
            content = source.get("content", "")
            author = source.get("author", "")
            
            source_text = f"Source: {title}"
            if author:
                source_text += f" (by {author})"
            source_text += f"\n{content}\n"
            
            context_parts.append(source_text)
        
        return "\n---\n".join(context_parts)
    
    def _parse_qa_response(self, response: str, question: str, question_type: str, 
                          sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM QA response into structured format"""
        
        lines = response.split('\n')
        
        answer = {
            "question": question,
            "type": question_type,
            "direct_answer": "",
            "reasoning": "",
            "evidence": [],
            "alternative_perspectives": [],
            "confidence": 0.7,  # default
            "limitations": [],
            "follow_up_questions": [],
            "sources_used": len(sources) if sources else 0
        }
        
        current_section = "direct_answer"
        answer_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "direct answer" in line.lower() or line.startswith("1."):
                current_section = "direct_answer"
                if "direct answer" not in line.lower():
                    answer_lines.append(line)
            elif "evidence" in line.lower() or "supporting" in line.lower() or line.startswith("2."):
                current_section = "evidence"
                if not answer["direct_answer"]:
                    answer["direct_answer"] = "\n".join(answer_lines).strip()
            elif "reasoning" in line.lower():
                current_section = "reasoning"
            elif "alternative" in line.lower() or "perspectives" in line.lower() or line.startswith("3."):
                current_section = "alternative_perspectives"
            elif "confidence" in line.lower() or line.startswith("4."):
                current_section = "confidence"
            elif "limitations" in line.lower() or "assumptions" in line.lower() or line.startswith("5."):
                current_section = "limitations"
            elif "follow" in line.lower() or "questions" in line.lower() or line.startswith("6."):
                current_section = "follow_up_questions"
            elif current_section == "direct_answer":
                answer_lines.append(line)
            elif current_section in ["evidence", "alternative_perspectives", "limitations", "follow_up_questions"]:
                if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                    clean_line = line[1:].strip() if line[0] in '-*•' else line
                    answer[current_section].append(clean_line)
            elif current_section == "reasoning":
                if not answer["reasoning"]:
                    answer["reasoning"] = line
                else:
                    answer["reasoning"] += " " + line
            elif current_section == "confidence":
                # Extract confidence score
                import re
                conf_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if conf_match:
                    conf_value = float(conf_match.group(1))
                    if conf_value <= 1.0:
                        answer["confidence"] = conf_value
                    elif conf_value <= 100:
                        answer["confidence"] = conf_value / 100
        
        # Set direct answer if not already set
        if not answer["direct_answer"]:
            answer["direct_answer"] = "\n".join(answer_lines).strip()
        
        # If no structured content found, use the whole response as direct answer
        if not answer["direct_answer"] and not any(answer[key] for key in ["evidence", "reasoning", "alternative_perspectives"]):
            answer["direct_answer"] = response.strip()
        
        return answer
    
    def ask_follow_up(self, original_qa_id: str, follow_up_question: str) -> Dict[str, Any]:
        """Ask a follow-up question based on previous QA"""
        if original_qa_id not in self.qa_cache:
            return {"error": "Original question not found"}
        
        original_qa = self.qa_cache[original_qa_id]
        original_question = original_qa.get("question", "")
        original_answer = original_qa.get("direct_answer", "")
        
        # Create context for follow-up
        context = f"Previous Question: {original_question}\nPrevious Answer: {original_answer}"
        
        # Answer follow-up question
        follow_up_answer = self.answer_question(follow_up_question, context)
        follow_up_answer["is_follow_up"] = True
        follow_up_answer["original_qa_id"] = original_qa_id
        
        return follow_up_answer
    
    def synthesize_multiple_answers(self, questions_and_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize information from multiple Q&A pairs"""
        
        if not questions_and_answers:
            return {"error": "No Q&A pairs provided"}
        
        # Prepare synthesis prompt
        qa_text = ""
        for i, qa in enumerate(questions_and_answers):
            qa_text += f"Q{i+1}: {qa.get('question', '')}\n"
            qa_text += f"A{i+1}: {qa.get('direct_answer', '')}\n\n"
        
        prompt = f"""
        Based on the following questions and answers, provide a synthesized analysis that:
        1. Identifies common themes and patterns
        2. Highlights any contradictions or conflicts
        3. Draws overall conclusions
        4. Suggests areas for further investigation
        
        {qa_text}
        """
        
        synthesis_response = self.generate_llm_response(prompt)
        
        synthesis = {
            "type": "synthesis",
            "num_qa_pairs": len(questions_and_answers),
            "synthesis": synthesis_response,
            "source_questions": [qa.get("question", "") for qa in questions_and_answers],
            "confidence": min([qa.get("confidence", 0.5) for qa in questions_and_answers])
        }
        
        return synthesis
    
    def fact_check_answer(self, answer: Dict[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fact-check an answer against provided sources"""
        
        direct_answer = answer.get("direct_answer", "")
        source_content = self._prepare_source_context(sources)
        
        prompt = f"""
        Fact-check the following answer against the provided sources:
        
        Answer to check: {direct_answer}
        
        Sources: {source_content}
        
        Provide:
        1. Accuracy assessment (High/Medium/Low)
        2. Specific facts that are supported by sources
        3. Claims that cannot be verified
        4. Any contradictions found
        """
        
        fact_check_response = self.generate_llm_response(prompt)
        
        fact_check = {
            "original_answer": direct_answer,
            "fact_check_analysis": fact_check_response,
            "sources_checked": len(sources),
            "timestamp": self.shared_memory._access_log[-1]["timestamp"] if self.shared_memory._access_log else ""
        }
        
        return fact_check
