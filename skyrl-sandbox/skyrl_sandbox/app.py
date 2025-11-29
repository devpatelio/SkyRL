"""CLI launcher for SkyRL Sandbox."""

import argparse
import sys
import webbrowser
from pathlib import Path


def main():
    """Main entry point for the SkyRL Sandbox application."""
    parser = argparse.ArgumentParser(
        description="SkyRL Sandbox - Build SkyRL-Gym environments visually"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Print startup message
    print("=" * 60)
    print("🚀 Starting SkyRL Sandbox")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    # Open browser automatically unless --no-browser is specified
    if not args.no_browser:
        url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
        print(f"\nOpening browser at {url}...")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Please open {url} manually.")
    
    # Start the server
    try:
        import uvicorn
        from skyrl_sandbox.server import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\nShutting down SkyRL Sandbox...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

