"""FastAPI backend server for SkyRL Sandbox."""

import json
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from skyrl_sandbox.codegen import export_environment, preview_env_logic
from skyrl_sandbox.dataset_utils import DatasetManager
from skyrl_sandbox.models import (
    DatasetExportRequest,
    DatasetExportResponse,
    DatasetSpec,
    EnvironmentSpec,
    ExportRequest,
    ExportResponse,
    PreviewRequest,
    PreviewResponse,
    TrainingLaunchRequest,
    TrainingLogsResponse,
    TrainingRunInfo,
)
from skyrl_sandbox.training import TrainingManager


def get_specs_dir() -> Path:
    """Get the specs directory."""
    return Path(__file__).parent / "specs"


def get_static_dir() -> Path:
    """Get the static files directory."""
    return Path(__file__).parent / "static"


def get_datasets_dir() -> Path:
    """Directory for dataset specs."""
    return Path(__file__).parent / "datasets"


def get_repo_root() -> Path:
    """Repository root for launching training jobs."""
    return Path(__file__).resolve().parents[2]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SkyRL Sandbox",
        description="Web interface for building SkyRL-Gym environments",
        version="0.1.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Ensure specs directory exists
    specs_dir = get_specs_dir()
    specs_dir.mkdir(parents=True, exist_ok=True)

    datasets_dir = get_datasets_dir()
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(datasets_dir)

    try:
        training_manager: Optional[TrainingManager] = TrainingManager(get_repo_root())
    except RuntimeError as exc:
        training_manager = None
        print(f"[sandbox] Training manager disabled: {exc}")
    
    @app.get("/")
    async def root():
        """Serve the main HTML page."""
        static_dir = get_static_dir()
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "SkyRL Sandbox API is running. Frontend not yet built."}
    
    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "service": "skyrl-sandbox"}
    
    @app.get("/api/specs", response_model=List[str])
    async def list_specs():
        """List all saved environment specs."""
        specs_dir = get_specs_dir()
        yaml_files = list(specs_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]
    
    @app.get("/api/specs/{name}", response_model=EnvironmentSpec)
    async def get_spec(name: str):
        """Get a specific environment spec."""
        specs_dir = get_specs_dir()
        spec_file = specs_dir / f"{name}.yaml"
        
        if not spec_file.exists():
            raise HTTPException(status_code=404, detail=f"Spec '{name}' not found")
        
        try:
            with open(spec_file, "r") as f:
                data = yaml.safe_load(f)
            return EnvironmentSpec(**data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading spec: {str(e)}")
    
    @app.post("/api/specs", response_model=dict)
    async def save_spec(spec: EnvironmentSpec):
        """Save or update an environment spec."""
        specs_dir = get_specs_dir()
        spec_file = specs_dir / f"{spec.env_name}.yaml"
        
        try:
            with open(spec_file, "w") as f:
                yaml.dump(spec.model_dump(), f, default_flow_style=False, sort_keys=False)
            return {"success": True, "message": f"Spec '{spec.env_name}' saved successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving spec: {str(e)}")
    
    @app.delete("/api/specs/{name}")
    async def delete_spec(name: str):
        """Delete an environment spec."""
        specs_dir = get_specs_dir()
        spec_file = specs_dir / f"{name}.yaml"
        
        if not spec_file.exists():
            raise HTTPException(status_code=404, detail=f"Spec '{name}' not found")
        
        try:
            spec_file.unlink()
            return {"success": True, "message": f"Spec '{name}' deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting spec: {str(e)}")
    
    @app.post("/api/preview", response_model=PreviewResponse)
    async def preview(request: PreviewRequest):
        """Preview reward calculation with sample data."""
        try:
            result = preview_env_logic(
                spec=request.spec,
                model_output=request.model_output,
                ground_truth=request.ground_truth,
                current_turn=request.current_turn,
            )
            return PreviewResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in preview: {str(e)}")
    
    @app.post("/api/export", response_model=ExportResponse)
    async def export(request: ExportRequest):
        """Generate code files for an environment."""
        try:
            success, message, files_created = export_environment(request.spec)
            return ExportResponse(
                success=success,
                message=message,
                files_created=files_created
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in export: {str(e)}")

    # Dataset endpoints ------------------------------------------------------
    @app.get("/api/datasets", response_model=List[str])
    async def list_datasets():
        return dataset_manager.list_specs()

    @app.get("/api/datasets/{name}", response_model=DatasetSpec)
    async def get_dataset(name: str):
        try:
            return dataset_manager.load_spec(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    @app.post("/api/datasets", response_model=dict)
    async def save_dataset(spec: DatasetSpec):
        try:
            dataset_manager.save_spec(spec)
            return {"success": True, "message": f"Dataset '{spec.dataset_name}' saved"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/datasets/{name}", response_model=dict)
    async def delete_dataset(name: str):
        try:
            dataset_manager.delete_spec(name)
            return {"success": True, "message": f"Dataset '{name}' deleted"}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    @app.post("/api/datasets/export", response_model=DatasetExportResponse)
    async def export_dataset(request: DatasetExportRequest):
        try:
            return dataset_manager.export(request.spec, fmt=request.format)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/datasets/exports", response_model=List[dict])
    async def list_dataset_exports():
        return dataset_manager.list_exports()

    # Training endpoints -----------------------------------------------------
    def _require_training_manager() -> TrainingManager:
        if training_manager is None:
            raise HTTPException(
                status_code=400,
                detail="Training module unavailable. Ensure 'skyrl-train' exists.",
            )
        return training_manager

    @app.get("/api/training/runs", response_model=List[TrainingRunInfo])
    async def list_training_runs():
        manager = _require_training_manager()
        return manager.list_runs()

    @app.post("/api/training/start", response_model=TrainingRunInfo)
    async def start_training(request: TrainingLaunchRequest):
        manager = _require_training_manager()
        try:
            run = manager.start_run(request)
            return run.as_model()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/training/runs/{run_id}", response_model=TrainingRunInfo)
    async def get_training_run(run_id: str):
        manager = _require_training_manager()
        try:
            return manager.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

    @app.post("/api/training/runs/{run_id}/stop", response_model=TrainingRunInfo)
    async def stop_training_run(run_id: str):
        manager = _require_training_manager()
        try:
            return manager.stop_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

    @app.get("/api/training/runs/{run_id}/logs", response_model=TrainingLogsResponse)
    async def get_training_logs(run_id: str, tail: int = Query(200, ge=0, le=2000)):
        manager = _require_training_manager()
        try:
            return manager.get_logs(run_id, tail=tail)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    
    # Mount static files for specific paths only (not at root to avoid interfering with API routes)
    # Serve static assets from their respective directories
    static_dir = get_static_dir()
    if static_dir.exists():
        # Check if there's an _astro directory and mount it specifically
        astro_dir = static_dir / "_astro"
        if astro_dir.exists():
            app.mount("/_astro", StaticFiles(directory=astro_dir), name="astro-assets")
        
        # You can add other specific static directories here as needed
        # For example, if you have images, css, js folders:
        for subdir_name in ["images", "css", "js", "assets"]:
            subdir = static_dir / subdir_name
            if subdir.exists():
                app.mount(f"/{subdir_name}", StaticFiles(directory=subdir), name=f"{subdir_name}-files")
    
    return app


# Create the app instance
app = create_app()

