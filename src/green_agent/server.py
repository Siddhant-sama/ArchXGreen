"""
A2A Protocol Server for ArchXBench Green Agent

Exposes RESTful endpoints following the A2A (Agent-to-Agent) protocol
for purple agents to interact with the benchmark.
"""

import os
import sys
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent.agent import create_green_agent, ArchXBenchGreenAgent


# ========== Pydantic Models for API ==========

class TaskListItem(BaseModel):
    task_id: str
    level: str
    problem_name: str
    difficulty_score: int
    level_description: str


class TaskDetail(BaseModel):
    task_id: str
    level: str
    problem_name: str
    difficulty_score: int
    problem_description: str
    design_specs: str
    testbench: Optional[str] = None


class SubmissionRequest(BaseModel):
    task_id: str = Field(..., description="Task identifier (e.g., 'level-0/mux2to1')")
    verilog_code: str = Field(..., description="The Verilog RTL code to evaluate")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    validate_architecture: bool = Field(True, description="Enable architectural compliance validation")
    use_llm_validation: bool = Field(False, description="Enable LLM-based validation (requires API key)")
    generate_feedback: bool = Field(False, description="Generate detailed feedback for improvement")


class EvaluationResponse(BaseModel):
    task_id: str
    passed: int
    failed: int
    total: int
    success: bool
    pass_rate: float
    compilation_success: bool
    simulation_success: bool
    synthesis_success: bool
    error_message: Optional[str] = None
    execution_time_ms: float
    feedback: Optional[dict] = None
    ppa_metrics: Optional[dict] = None
    architectural_compliance: Optional[dict] = None


class SessionCreateRequest(BaseModel):
    agent_id: str = Field(..., description="Identifier for the purple agent")


class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    started_at: str


class HealthResponse(BaseModel):
    status: str
    benchmark: str
    version: str
    total_tasks: int
    timestamp: str


# ========== Create FastAPI App ==========

def create_app(
    benchmark_root: Optional[str] = None,
    use_dynamic_loader: bool = True
) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        benchmark_root: Path to local benchmark directory (if not using dynamic loader)
        use_dynamic_loader: If True, fetch benchmarks from GitHub dynamically
    """
    
    app = FastAPI(
        title="ArchXBench Green Agent",
        description="""
        A2A Protocol Server for ArchXBench - RTL Synthesis Benchmark
        
        This green agent provides:
        - Task discovery and retrieval
        - Verilog code evaluation against testbenches
        - Session management for tracking agent progress
        - Benchmark metrics and scoring
        - Dynamic benchmark loading from GitHub
        - Detailed feedback for iterative improvement
        
        **Protocol:** Agent-to-Agent (A2A) for AgentBeats platform
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize green agent with dynamic loading support
    agent = create_green_agent(
        benchmark_root=benchmark_root,
        use_dynamic_loader=use_dynamic_loader
    )
    
    # Store agent in app state
    app.state.agent = agent
    
    # ========== Health & Info Endpoints ==========
    
    @app.get("/", response_model=HealthResponse, tags=["Health"])
    async def root():
        """Health check and basic info"""
        return HealthResponse(
            status="healthy",
            benchmark="ArchXBench",
            version="1.0.0",
            total_tasks=len(agent.tasks),
            timestamp=datetime.utcnow().isoformat()
        )
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            benchmark="ArchXBench",
            version="1.0.0",
            total_tasks=len(agent.tasks),
            timestamp=datetime.utcnow().isoformat()
        )
    
    @app.get("/metrics", tags=["Metrics"])
    async def get_metrics():
        """
        Get benchmark metadata and scoring information.
        
        Returns information about:
        - Total tasks and tasks per level
        - Scoring methodology
        - Evaluation tool details
        """
        return agent.get_metrics()
    
    # ========== Task Discovery Endpoints ==========
    
    @app.get("/tasks", response_model=List[TaskListItem], tags=["Tasks"])
    async def list_tasks(
        level: Optional[str] = Query(None, description="Filter by level (e.g., 'level-0')"),
        limit: Optional[int] = Query(None, description="Maximum number of tasks"),
        offset: int = Query(0, description="Pagination offset")
    ):
        """
        List available benchmark tasks.
        
        Purple agents use this endpoint to discover tasks they can attempt.
        """
        tasks = agent.get_task_list(level=level, limit=limit, offset=offset)
        return tasks
    
    @app.get("/tasks/{task_id:path}", response_model=TaskDetail, tags=["Tasks"])
    async def get_task(
        task_id: str,
        include_testbench: bool = Query(True, description="Include testbench in response")
    ):
        """
        Get detailed information for a specific task.
        
        Returns:
        - Problem description
        - Design specifications  
        - Testbench (optional, for reference)
        """
        try:
            return agent.get_task(task_id, include_testbench=include_testbench)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.get("/levels", tags=["Tasks"])
    async def list_levels():
        """List all benchmark levels with descriptions"""
        metrics = agent.get_metrics()
        return {
            "levels": metrics["levels"],
            "tasks_per_level": metrics["tasks_per_level"]
        }
    
    # ========== Evaluation Endpoints ==========
    
    @app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
    async def evaluate_submission(request: SubmissionRequest):
        """
        Evaluate a Verilog submission against the task's testbench.
        
        This is the primary endpoint for purple agents to submit solutions.
        The green agent will:
        1. Compile the Verilog code with the testbench
        2. Run simulation
        3. Validate architectural compliance (rule-based + optional LLM)
        4. Parse results and return pass/fail counts
        
        **Architectural Validation:**
        - `validate_architecture=True`: Enable rule-based validation (default, free)
        - `use_llm_validation=True`: Enable LLM validation (requires API key, ~$0.01-0.03 per task)
        
        **Note:** LLM validation requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
        If not set, LLM validation is skipped (no error, falls back to rule-based only).
        """
        result = agent.evaluate_submission(
            task_id=request.task_id,
            verilog_code=request.verilog_code,
            session_id=request.session_id,
            validate_architecture=request.validate_architecture,
            use_llm_validation=request.use_llm_validation,
            generate_feedback=request.generate_feedback
        )
        
        return EvaluationResponse(
            task_id=result.task_id,
            passed=result.passed,
            failed=result.failed,
            total=result.total,
            success=result.success,
            pass_rate=result.pass_rate,
            compilation_success=result.compilation_success,
            simulation_success=result.simulation_success,
            synthesis_success=result.synthesis_success,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            feedback=result.feedback.to_dict() if result.feedback else None,
            ppa_metrics=result.ppa_metrics.to_dict() if result.ppa_metrics else None,
            architectural_compliance=result.architectural_compliance.to_dict() if result.architectural_compliance else None
        )
    
    @app.post("/evaluate/upload", response_model=EvaluationResponse, tags=["Evaluation"])
    async def evaluate_upload(
        task_id: str = Form(..., description="Task identifier (e.g., 'level-0/mux2to1')"),
        verilog_file: UploadFile = File(..., description="Verilog source file (.v or .sv)"),
        session_id: Optional[str] = Form(None, description="Optional session ID for tracking"),
        validate_architecture: bool = Form(True, description="Enable architectural compliance validation"),
        use_llm_validation: bool = Form(False, description="Enable LLM-based validation (requires API key)"),
        generate_feedback: bool = Form(False, description="Generate detailed feedback for improvement")
    ):
        """
        Evaluate a Verilog submission via file upload.
        
        **This endpoint is designed for manual testing via the /docs interface.**
        Agents should use the `/evaluate` endpoint with JSON payload instead.
        
        Upload your Verilog file directly to test multiline code without JSON escaping issues.
        The file content will be read and evaluated against the task's testbench.
        
        **Usage in /docs:**
        1. Click "Try it out"
        2. Select task_id from dropdown or type it
        3. Click "Choose File" and upload your .v file
        4. Set validation options
        5. Click "Execute"
        
        **Architectural Validation:**
        - `validate_architecture=True`: Enable rule-based validation (default, free)
        - `use_llm_validation=True`: Enable LLM validation (requires API key, ~$0.01-0.03 per task)
        """
        # Read file content
        try:
            verilog_code = await verilog_file.read()
            verilog_code = verilog_code.decode('utf-8')
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to read file: {str(e)}"
            )
        
        # Evaluate using the same logic as /evaluate
        result = agent.evaluate_submission(
            task_id=task_id,
            verilog_code=verilog_code,
            session_id=session_id,
            validate_architecture=validate_architecture,
            use_llm_validation=use_llm_validation,
            generate_feedback=generate_feedback
        )
        
        return EvaluationResponse(
            task_id=result.task_id,
            passed=result.passed,
            failed=result.failed,
            total=result.total,
            success=result.success,
            pass_rate=result.pass_rate,
            compilation_success=result.compilation_success,
            simulation_success=result.simulation_success,
            synthesis_success=result.synthesis_success,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            feedback=result.feedback.to_dict() if result.feedback else None,
            ppa_metrics=result.ppa_metrics.to_dict() if result.ppa_metrics else None,
            architectural_compliance=result.architectural_compliance.to_dict() if result.architectural_compliance else None
        )
    
    @app.post("/evaluate/batch", tags=["Evaluation"])
    async def evaluate_batch(
        submissions: List[SubmissionRequest] = Body(..., description="List of submissions")
    ):
        """
        Evaluate multiple submissions in batch.
        
        Useful for running full benchmark evaluations.
        Supports architectural validation and LLM validation per submission.
        """
        results = []
        for sub in submissions:
            result = agent.evaluate_submission(
                task_id=sub.task_id,
                verilog_code=sub.verilog_code,
                session_id=sub.session_id,
                validate_architecture=sub.validate_architecture,
                use_llm_validation=sub.use_llm_validation,
                generate_feedback=sub.generate_feedback
            )
            results.append(result.to_dict())
        
        # Compute aggregate score
        from green_agent.models import EvaluationResult
        eval_results = [
            EvaluationResult(**{k: v for k, v in r.items() if k != 'pass_rate'})
            for r in results
        ]
        score = agent.compute_weighted_score(eval_results)
        
        return {
            "results": results,
            "aggregate": score
        }
    
    # ========== Session Management Endpoints ==========
    
    @app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
    async def create_session(request: SessionCreateRequest):
        """
        Create a new evaluation session.
        
        Sessions track a purple agent's progress through the benchmark.
        This supports the A2A reproducibility requirement.
        """
        session_id = agent.create_session(request.agent_id)
        session = agent.sessions[session_id]
        
        return SessionResponse(
            session_id=session_id,
            agent_id=session.agent_id,
            started_at=session.started_at
        )
    
    @app.get("/sessions/{session_id}", tags=["Sessions"])
    async def get_session(session_id: str):
        """Get session results and progress"""
        try:
            return agent.get_session_results(session_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.post("/sessions/{session_id}/reset", tags=["Sessions"])
    async def reset_session(session_id: str):
        """
        Reset a session to clean state.
        
        This is required for reproducibility - ensures each evaluation
        run starts from the same state.
        """
        if agent.reset_session(session_id):
            return {"status": "reset", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Session not found")
    
    # ========== Dynamic Benchmark Management ==========
    
    @app.post("/benchmarks/update", tags=["Management"])
    async def update_benchmarks(force: bool = Query(False, description="Force update even if cache is valid")):
        """
        Update benchmarks from remote repository.
        
        This endpoint allows refreshing the benchmark tasks dynamically from the
        public ArchXBench GitHub repository. Useful for getting the latest tasks
        without restarting the server.
        
        **Note:** Only available when using dynamic loader mode.
        """
        result = agent.update_benchmarks(force=force)
        return result
    
    @app.get("/benchmarks/cache-info", tags=["Management"])
    async def get_cache_info():
        """
        Get information about cached benchmarks.
        
        Returns cache status, age, expiry, and source repository details.
        """
        return agent.get_cache_info()
    
    return app


# ========== Module-level app instance for uvicorn ==========

# Create default app instance for direct uvicorn usage
benchmark_root = os.environ.get("ARCHXBENCH_ROOT")
use_dynamic = os.environ.get("USE_DYNAMIC_LOADER", "true").lower() == "true"
app = create_app(benchmark_root=benchmark_root, use_dynamic_loader=use_dynamic)


# ========== Main Entry Point ==========

def main():
    """Run the green agent server"""
    import uvicorn
    
    benchmark_root = os.environ.get("ARCHXBENCH_ROOT")
    use_dynamic = os.environ.get("USE_DYNAMIC_LOADER", "true").lower() == "true"
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    # Validate benchmark root if using static mode
    if not use_dynamic and benchmark_root and not os.path.isdir(benchmark_root):
        print(f"Error: Benchmark root '{benchmark_root}' is not a valid directory")
        sys.exit(1)
    
    print(f"Starting ArchXBench Green Agent...")
    if use_dynamic:
        print(f"  Mode: Dynamic (fetching from GitHub)")
        print(f"  Repository: https://github.com/sureshpurini/ArchXBench")
    else:
        print(f"  Mode: Static")
        print(f"  Benchmark root: {os.path.abspath(benchmark_root) if benchmark_root else 'N/A'}")
    print(f"  Server: http://{host}:{port}")
    print(f"  API Docs: http://{host}:{port}/docs")
    
    try:
        app = create_app(benchmark_root=benchmark_root, use_dynamic_loader=use_dynamic)
        print("✓ Application created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Starting uvicorn server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
