"""FastAPI backend server for SkyRL Sandbox."""

import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yaml

from skyrl_sandbox.models import (
    EnvironmentSpec,
    PreviewRequest,
    PreviewResponse,
    ExportRequest,
    ExportResponse,
)
from skyrl_sandbox.codegen import export_environment, preview_env_logic


def get_specs_dir() -> Path:
    """Get the specs directory."""
    return Path(__file__).parent / "specs"


def get_static_dir() -> Path:
    """Get the static files directory."""
    return Path(__file__).parent / "static"


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
    
    # Mount static files (after all routes)
    # Serve static assets from root (for Astro's _astro directory and other assets)
    static_dir = get_static_dir()
    if static_dir.exists():
        # Mount at root for assets (Astro builds assets referenced from root)
        # This will serve files like /_astro/... and other static assets
        app.mount("/", StaticFiles(directory=static_dir, html=False), name="static-assets")
    
    return app


# Create the app instance
app = create_app()

