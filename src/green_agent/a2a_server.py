"""
A2A Protocol-Compliant Server for ArchXBench Green Agent

Implements the AgentBeats A2A (Agent-to-Agent) protocol specification:
- Assessment request handling
- Task updates (streaming logs)
- Artifact generation (results JSON)
- Agent card exposure
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Optional, Dict, Any, List, AsyncIterator
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent.agent import create_green_agent, ArchXBenchGreenAgent


# ========== A2A Protocol Models ==========

class A2AParticipant(BaseModel):
    """A2A participant mapping (role -> endpoint URL)"""
    pass  # Dynamic keys like {"agent": "http://purple:9009"}


class A2AAssessmentRequest(BaseModel):
    """
    A2A Assessment Request Format (AgentBeats standard)
    
    Sent by the assessment runner to start an evaluation.
    """
    participants: Dict[str, str] = Field(
        ..., 
        description="Mapping of role names to A2A endpoint URLs"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Assessment-specific configuration (e.g., domain, num_tasks, levels)"
    )


class A2AAgentCard(BaseModel):
    """A2A Agent Card (Agent Identity and Capabilities)"""
    name: str
    version: str
    description: str
    url: str
    skills: List[str]
    capabilities: Dict[str, Any]


class A2ATaskUpdate(BaseModel):
    """A2A Task Update (Streaming log message)"""
    task_id: str
    timestamp: str
    level: str  # info, warning, error, success
    message: str
    metadata: Optional[Dict[str, Any]] = None


class A2AArtifact(BaseModel):
    """A2A Artifact (Result payload)"""
    task_id: str
    name: str
    mime_type: str
    content: Any  # JSON object or base64 string


# ========== A2A Server ==========

class ArchXBenchA2AServer:
    """
    AgentBeats-compatible A2A server for ArchXBench green agent.
    
    Implements:
    - POST /assessment - Handle assessment requests
    - GET /card - Expose agent card
    - Health check
    """
    
    def __init__(
        self, 
        host: str = "0.0.0.0",
        port: int = 9009,
        card_url: Optional[str] = None,
        benchmark_root: Optional[str] = None,
        use_dynamic_loader: bool = True
    ):
        self.host = host
        self.port = port
        base_url = f"http://{host}:{port}"
        self.card_url = card_url or f"{base_url}/.well-known/agent-card.json"
        
        # Initialize green agent
        self.agent = create_green_agent(
            benchmark_root=benchmark_root,
            use_dynamic_loader=use_dynamic_loader
        )
        
        # Active task tracking (for streaming updates)
        self.active_tasks: Dict[str, List[A2ATaskUpdate]] = {}
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with A2A endpoints"""
        
        app = FastAPI(
            title="ArchXBench Green Agent (A2A)",
            description="A2A-compliant evaluation server for RTL Synthesis Benchmark",
            version="1.0.0"
        )
        
        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # ========== A2A Protocol Endpoints ==========
        
        @app.get("/card", response_model=A2AAgentCard)
        @app.get("/.well-known/agent-card.json", response_model=A2AAgentCard)
        async def get_agent_card():
            """
            A2A Agent Card - Required by AgentBeats
            
            Returns agent identity, capabilities, and skills.
            """
            # Derive level list from agent metrics (no direct 'levels' attribute on agent)
            metrics = self.agent.get_metrics()
            level_names = list(metrics.get("levels", {}).keys())
            return A2AAgentCard(
                name="ArchXBench Green Agent",
                version="1.0.0",
                description=(
                    "Comprehensive RTL synthesis benchmark evaluator. "
                    "Tests autonomous agents on Verilog design tasks across 7 difficulty levels. "
                    "Features dynamic benchmark loading, compilation, simulation, synthesis, "
                    "and PPA metrics (Power, Performance, Area)."
                ),
                url=self.card_url,
                skills=[
                    "verilog-evaluation",
                    "rtl-synthesis",
                    "hardware-verification",
                    "testbench-simulation",
                    "ppa-analysis",
                    "architectural-validation"
                ],
                capabilities={
                    "benchmark": "ArchXBench",
                    "total_tasks": len(self.agent.tasks),
                    "levels": level_names,
                    "difficulty_range": "Level-0 (Basic) to Level-6 (Complex Systems)",
                    "evaluation_tools": {
                        "compiler": "Icarus Verilog",
                        "simulator": "vvp (Verilog VVP)",
                        "synthesizer": "Yosys"
                    },
                    "features": {
                        "dynamic_loading": True,
                        "feedback_generation": True,
                        "architectural_compliance": True,
                        "ppa_metrics": True,
                        "llm_validation": "optional"
                    },
                    "domains": [
                        "combinational-logic",
                        "sequential-logic",
                        "arithmetic-units",
                        "pipelined-designs",
                        "floating-point",
                        "dsp-kernels",
                        "accelerators"
                    ]
                }
            )
        
        @app.post("/assessment")
        async def handle_assessment(request: A2AAssessmentRequest):
            """
            A2A Assessment Handler - Required by AgentBeats
            
            Receives assessment request from platform and orchestrates evaluation.
            Returns streaming task updates followed by final artifact.
            
            Request format:
            {
                "participants": {"agent": "http://purple-agent:9009"},
                "config": {
                    "levels": ["level-0"],
                    "num_tasks": 3,
                    "timeout_per_task": 300
                }
            }
            
            Response: SSE stream of task updates + final artifact
            """
            # Generate unique task ID for this assessment
            task_id = f"archxbench-{uuid4().hex[:8]}"
            self.active_tasks[task_id] = []
            
            # Extract configuration
            config = request.config
            levels = config.get("levels", ["level-0"])
            num_tasks = config.get("num_tasks", None)
            timeout = config.get("timeout_per_task", 300)
            
            # Get participant endpoints
            participants = request.participants
            
            # Stream assessment execution
            return StreamingResponse(
                self._run_assessment_stream(
                    task_id=task_id,
                    participants=participants,
                    levels=levels,
                    num_tasks=num_tasks,
                    timeout=timeout
                ),
                media_type="text/event-stream"
            )
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "agent": "ArchXBench Green Agent",
                "protocol": "A2A v1.0",
                "total_tasks": len(self.agent.tasks),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # ========== Task Discovery Endpoints (for compatibility) ==========
        
        @app.get("/tasks")
        async def list_tasks(
            level: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
        ):
            """
            List available benchmark tasks.
            
            Compatible with REST API for testing/development.
            """
            tasks = self.agent.get_task_list(level=level, limit=limit, offset=offset)
            return tasks
        
        @app.get("/tasks/{task_id:path}")
        async def get_task(task_id: str, include_testbench: bool = True):
            """Get detailed information for a specific task."""
            try:
                return self.agent.get_task(task_id, include_testbench=include_testbench)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @app.get("/levels")
        async def list_levels():
            """List all benchmark levels with descriptions"""
            metrics = self.agent.get_metrics()
            return {
                "levels": metrics["levels"],
                "tasks_per_level": metrics["tasks_per_level"]
            }
        
        @app.get("/")
        async def root():
            """Root endpoint - redirect to card"""
            return {
                "agent": "ArchXBench Green Agent",
                "protocol": "A2A",
                "endpoints": {
                    "card": "/card",
                    "assessment": "/assessment",
                    "health": "/health",
                    "tasks": "/tasks",
                    "levels": "/levels"
                }
            }
        
        return app
    
    async def _run_assessment_stream(
        self,
        task_id: str,
        participants: Dict[str, str],
        levels: List[str],
        num_tasks: Optional[int],
        timeout: int
    ) -> AsyncIterator[str]:
        """
        Run assessment and stream updates in A2A format.
        
        Yields SSE-formatted task updates followed by final artifact.
        """
        import httpx
        
        # Send initial update
        yield self._format_task_update(
            task_id=task_id,
            level="info",
            message=f"Starting ArchXBench assessment: {len(levels)} level(s), {num_tasks or 'all'} tasks",
            metadata={"levels": levels, "participants": list(participants.keys())}
        )
        
        # Get tasks to evaluate
        all_tasks = []
        for level in levels:
            level_tasks = self.agent.get_task_list(level=level)
            all_tasks.extend(level_tasks)
        
        if num_tasks:
            all_tasks = all_tasks[:num_tasks]
        
        yield self._format_task_update(
            task_id=task_id,
            level="info",
            message=f"Loaded {len(all_tasks)} tasks for evaluation",
            metadata={"task_count": len(all_tasks)}
        )
        
        # Get purple agent endpoint
        agent_endpoint = participants.get("agent")
        if not agent_endpoint:
            yield self._format_task_update(
                task_id=task_id,
                level="error",
                message="No purple agent endpoint provided in participants",
                metadata={"participants": participants}
            )
            yield self._format_artifact(
                task_id=task_id,
                name="error",
                content={"error": "Missing agent participant"}
            )
            return
        
        # Run evaluations
        results = []
        passed_count = 0
        failed_count = 0
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            for idx, task_info in enumerate(all_tasks, 1):
                task_name = task_info["task_id"]
                
                yield self._format_task_update(
                    task_id=task_id,
                    level="info",
                    message=f"[{idx}/{len(all_tasks)}] Requesting solution for: {task_name}",
                    metadata={"current_task": task_name, "progress": f"{idx}/{len(all_tasks)}"}
                )
                
                # Get task details
                try:
                    task_details = self.agent.get_task(task_name, include_testbench=False)
                except Exception as e:
                    yield self._format_task_update(
                        task_id=task_id,
                        level="error",
                        message=f"Failed to load task {task_name}: {str(e)}",
                        metadata={"task_id": task_name, "error": str(e)}
                    )
                    continue
                
                # Request solution from purple agent
                try:
                    # Call purple agent's A2A endpoint (assuming it has /solve or similar)
                    # For now, we'll call our own evaluation endpoint as fallback
                    # In real A2A, purple agent would have a /task endpoint
                    
                    # Placeholder: In full A2A, we'd POST task to purple agent
                    # and it would return verilog_code
                    # For this demo, we skip purple agent call (they run separately)
                    
                    yield self._format_task_update(
                        task_id=task_id,
                        level="warning",
                        message=f"Skipping A2A call to purple agent (run assessments via AgentBeats platform)",
                        metadata={"task_id": task_name}
                    )
                    
                    # Placeholder result
                    results.append({
                        "task_id": task_name,
                        "status": "skipped",
                        "reason": "Direct A2A assessment not implemented (use AgentBeats runner)"
                    })
                    
                except Exception as e:
                    yield self._format_task_update(
                        task_id=task_id,
                        level="error",
                        message=f"Failed to evaluate {task_name}: {str(e)}",
                        metadata={"task_id": task_name, "error": str(e)}
                    )
                    failed_count += 1
                    results.append({
                        "task_id": task_name,
                        "passed": 0,
                        "failed": 1,
                        "total": 1,
                        "success": False,
                        "error": str(e)
                    })
        
        # Send completion update
        yield self._format_task_update(
            task_id=task_id,
            level="success",
            message=f"Assessment complete: {passed_count}/{len(all_tasks)} tasks passed",
            metadata={
                "passed": passed_count,
                "failed": failed_count,
                "total": len(all_tasks)
            }
        )
        
        # Send final artifact
        artifact_content = {
            "participants": participants,
            "results": results,
            "summary": {
                "total_tasks": len(all_tasks),
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": passed_count / len(all_tasks) if all_tasks else 0.0,
                "levels": levels
            }
        }
        
        yield self._format_artifact(
            task_id=task_id,
            name="results",
            content=artifact_content
        )
    
    def _format_task_update(
        self,
        task_id: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format task update as SSE message"""
        update = {
            "type": "task_update",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "metadata": metadata or {}
        }
        return f"data: {json.dumps(update)}\n\n"
    
    def _format_artifact(
        self,
        task_id: str,
        name: str,
        content: Any
    ) -> str:
        """Format artifact as SSE message"""
        artifact = {
            "type": "artifact",
            "task_id": task_id,
            "name": name,
            "mime_type": "application/json",
            "content": content
        }
        return f"data: {json.dumps(artifact)}\n\n"
    
    def run(self):
        """Start the A2A server"""
        print(f"🟢 ArchXBench Green Agent (A2A Protocol)")
        print(f"   Server: http://{self.host}:{self.port}")
        print(f"   Card: {self.card_url}")
        print(f"   Tasks: {len(self.agent.tasks)}")
        print(f"   Protocol: Agent-to-Agent (A2A) v1.0")
        print(f"   AgentBeats: Compatible")
        print()
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# ========== Main Entry Point ==========

def main():
    """Run the A2A server with AgentBeats-compatible arguments"""
    parser = argparse.ArgumentParser(
        description="ArchXBench Green Agent - A2A Protocol Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "9009")),
        help="Port to listen on (default: 9009)"
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=os.environ.get("CARD_URL"),
        help="Public URL for agent card (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Get benchmark configuration
    benchmark_root = os.environ.get("ARCHXBENCH_ROOT")
    use_dynamic = os.environ.get("USE_DYNAMIC_LOADER", "true").lower() == "true"
    
    # Create and run server
    server = ArchXBenchA2AServer(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        benchmark_root=benchmark_root,
        use_dynamic_loader=use_dynamic
    )
    
    server.run()


if __name__ == "__main__":
    main()
