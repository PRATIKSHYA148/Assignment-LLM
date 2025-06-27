"""
Planning agent for task decomposition and planning
"""
from typing import Dict, List, Any
from agents.base_agent import BaseAgent
from utils.message_system import Message


class PlanningAgent(BaseAgent):
    """Agent specialized in breaking down complex tasks into actionable plans"""
    
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="PlanningAgent",
            agent_type="planning",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )
        
        # Planning-specific attributes
        self.active_plans = {}
        self.plan_templates = self._load_plan_templates()
    
    def _load_plan_templates(self) -> Dict[str, List[str]]:
        """Load predefined planning templates"""
        return {
            "research": [
                "Define research objectives and scope",
                "Identify key sources and resources",
                "Conduct preliminary literature review",
                "Develop research methodology",
                "Collect and analyze data",
                "Synthesize findings and insights",
                "Prepare final report and recommendations"
            ],
            "analysis": [
                "Gather and organize relevant data",
                "Identify key metrics and KPIs",
                "Perform exploratory data analysis",
                "Apply analytical frameworks",
                "Validate findings and assumptions",
                "Draw conclusions and insights",
                "Present results and recommendations"
            ],
            "project": [
                "Define project scope and objectives",
                "Identify stakeholders and requirements",
                "Create work breakdown structure",
                "Estimate resources and timeline",
                "Develop risk mitigation strategies",
                "Execute project phases",
                "Monitor progress and adjust as needed"
            ]
        }
    
    def _handle_task_request(self, message: Message):
        """Handle task planning requests"""
        content = message.content
        task = content.get("task", "")
        context = content.get("context", "")
        requirements = content.get("requirements", [])
        
        # Generate plan using LLM
        plan = self.create_task_plan(task, context, requirements)
        
        # Store plan in shared memory
        plan_id = f"plan_{message.id}"
        self.store_in_memory(plan_id, plan, ["plan", "task_breakdown"])
        self.active_plans[plan_id] = plan
        
        # Send plan back to requester
        self.send_message(
            message.sender,
            "plan_response",
            {
                "plan_id": plan_id,
                "plan": plan,
                "task": task
            }
        )
        
        # Notify coordinator about new plan
        self.send_message(
            "CoordinatorAgent",
            "plan_created",
            {
                "plan_id": plan_id,
                "requester": message.sender,
                "task": task,
                "steps_count": len(plan.get("steps", []))
            }
        )
    
    def create_task_plan(self, task: str, context: str = "", requirements: List[str] = None) -> Dict[str, Any]:
        """Create a detailed plan for a given task"""
        
        # Detect plan type
        plan_type = self._detect_plan_type(task)
        
        # Create planning prompt
        prompt = f"""
        Create a detailed plan for the following task:
        Task: {task}
        Context: {context}
        Requirements: {', '.join(requirements or [])}
        
        Please provide:
        1. A clear objective statement
        2. Step-by-step breakdown of the task
        3. Estimated effort for each step
        4. Dependencies between steps
        5. Required resources
        6. Potential risks and mitigation strategies
        7. Success criteria
        
        Format as a structured plan with clear, actionable steps.
        """
        
        # Generate plan using LLM
        response = self.generate_llm_response(prompt, context)
        
        # Parse and structure the response
        plan = self._parse_plan_response(response, plan_type)
        
        return plan
    
    def _detect_plan_type(self, task: str) -> str:
        """Detect the type of plan needed based on task description"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["research", "study", "investigate", "analyze"]):
            return "research"
        elif any(word in task_lower for word in ["analyze", "examination", "review", "assessment"]):
            return "analysis"
        else:
            return "project"
    
    def _parse_plan_response(self, response: str, plan_type: str) -> Dict[str, Any]:
        """Parse LLM response into structured plan"""
        
        # Basic parsing - in production, this would be more sophisticated
        lines = response.split('\n')
        
        plan = {
            "type": plan_type,
            "objective": "",
            "steps": [],
            "resources": [],
            "risks": [],
            "success_criteria": [],
            "estimated_duration": "TBD"
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if "objective" in line.lower():
                current_section = "objective"
            elif "step" in line.lower() or ("1." in line or "2." in line):
                current_section = "steps"
            elif "resource" in line.lower():
                current_section = "resources"
            elif "risk" in line.lower():
                current_section = "risks"
            elif "success" in line.lower() or "criteria" in line.lower():
                current_section = "success_criteria"
            elif current_section:
                # Add content to current section
                if current_section == "objective" and not plan["objective"]:
                    plan["objective"] = line
                elif current_section == "steps":
                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*')):
                        plan["steps"].append({
                            "description": line,
                            "estimated_effort": "TBD",
                            "dependencies": [],
                            "status": "pending"
                        })
                elif current_section in ["resources", "risks", "success_criteria"]:
                    if line.startswith(('-', '*', '•')):
                        plan[current_section].append(line[1:].strip())
        
        # If no steps were parsed, create default steps based on template
        if not plan["steps"] and plan_type in self.plan_templates:
            for i, step_desc in enumerate(self.plan_templates[plan_type]):
                plan["steps"].append({
                    "description": f"{i+1}. {step_desc}",
                    "estimated_effort": "TBD",
                    "dependencies": [],
                    "status": "pending"
                })
        
        return plan
    
    def update_plan_status(self, plan_id: str, step_index: int, status: str):
        """Update status of a specific plan step"""
        if plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
            if 0 <= step_index < len(plan["steps"]):
                plan["steps"][step_index]["status"] = status
                
                # Update in shared memory
                self.store_in_memory(plan_id, plan, ["plan", "task_breakdown", "updated"])
    
    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get progress summary for a plan"""
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}
        
        plan = self.active_plans[plan_id]
        total_steps = len(plan["steps"])
        completed_steps = sum(1 for step in plan["steps"] if step["status"] == "completed")
        in_progress_steps = sum(1 for step in plan["steps"] if step["status"] == "in_progress")
        
        return {
            "plan_id": plan_id,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "in_progress_steps": in_progress_steps,
            "progress_percentage": (completed_steps / total_steps) * 100 if total_steps > 0 else 0,
            "current_step": next((i for i, step in enumerate(plan["steps"]) 
                                if step["status"] == "in_progress"), None)
        }
    
    def suggest_next_steps(self, plan_id: str) -> List[str]:
        """Suggest next actionable steps for a plan"""
        if plan_id not in self.active_plans:
            return []
        
        plan = self.active_plans[plan_id]
        next_steps = []
        
        for step in plan["steps"]:
            if step["status"] == "pending":
                # Check if dependencies are met
                dependencies_met = all(
                    any(s["status"] == "completed" for s in plan["steps"] 
                        if dep in s["description"])
                    for dep in step["dependencies"]
                ) if step["dependencies"] else True
                
                if dependencies_met:
                    next_steps.append(step["description"])
        
        return next_steps[:3]  # Return top 3 next steps
