# SkyRL-Sandbox

A web interface for building custom environments and post-training through SkyRL-Gym and SkyRL-Train. 

## Key Features

- **Visual builder**: Define prompts, outputs, and reward logic through a simple web UI.
- **Codegen**: Auto-generate `env.py` and `register_env.py` for your custom environment for SkyRL-Gym / SkyRL-Train.
- **One-Click Training**: Launch training runs for your custom environment from the browser.

## Running App

You can run the sandbox using uvicorn:


```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-sandbox
uv run skyrl-sandbox
```

Then open your browser and see the results.

You can use the [Walkthrough Docs](https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html) to get a better sense of what to input into the web interface. 

Start off with the environments page and fill in the form. Once you preview the reward logic, you can 
click **Save environment** which will generate the yaml configuration followed by **Export code** to
create the `env.py` and `register_env.py` files under `skyrl-train/runs/_env_name`. You can then
go to the **Training** tab and select your environment and fill in the paths to the train/val data along with any custom config overrides. 