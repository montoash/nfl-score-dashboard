# NFL Score Prediction Dashboard

This project contains a FastAPI backend and a simple Tailwind‑powered HTML
dashboard for predicting NFL game scores. The backend wraps a machine
learning pipeline that downloads historical schedules via
[`nfl_data_py`](https://github.com/nflverse/nflverse-data)
and trains several regression models to forecast home and away scores. The
front‑end queries the API to display predicted spreads, totals and model
performance metrics.

## Project structure

| Path | Description |
|---|---|
| `nfl_score_prediction.py` | Core modelling code. Downloads data, engineers features, evaluates models and trains the final regressor. |
| `nfl_dashboard_api.py` | FastAPI application exposing prediction and metric endpoints and serving the dashboard HTML. |
| `nfl_dashboard.html` | Interactive front‑end that calls the API and renders predictions using Tailwind CSS. |
| `requirements.txt` | Python dependencies required to run the API and training pipeline. |
| `Procfile` | Process definition used by hosting platforms such as Heroku or Render to start the web server. |

## Running locally

1. **Install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the server**

   ```bash
   uvicorn nfl_dashboard_api:app --reload
   ```

3. **Open the dashboard**

   Navigate to [`http://localhost:8000`](http://localhost:8000) in your browser. The root path serves the dashboard and will display predictions for the most recent week by default. You can change the season and week using the dropdowns.

## Deployment guide

To make the dashboard accessible from any device, you can deploy this app to a free hosting service such as [Render](https://render.com) or [Heroku](https://www.heroku.com). Below are high‑level steps for Render:

1. **Create a GitHub repository**

   * Log in to your GitHub account and create a new repository (e.g. `nfl-score-dashboard`).
   * Upload all files from this `project` folder to the repository. You can use Git on your local machine or the GitHub web interface to add the files.

2. **Provision a web service on Render**

   * Sign in to [Render](https://dashboard.render.com) using your GitHub account.
   * Click **New &gt; Web Service** and select the repository you just created.
   * Configure the service:
     - **Environment**: `Python`
     - **Build command**: `pip install -r requirements.txt`
     - **Start command**: `uvicorn nfl_dashboard_api:app --host=0.0.0.0 --port=$PORT`
     - Leave other settings at their defaults or adjust as needed (free plan should suffice for light traffic).
   * Click **Create Web Service**. Render will install dependencies, start the FastAPI server and assign a public URL (e.g. `https://nfl-score-dashboard.onrender.com`).

3. **Access the dashboard**

   Once the service is live, visit the root URL provided by Render. You should see the NFL Score Predictions dashboard. Use the season and week selectors to generate fresh predictions on demand.

## Notes

* The first request to `/api/predict` triggers model training and evaluation, which can take several seconds. Subsequent requests in the same deployment reuse the cached model and respond faster.
* If `nfl_data_py` cannot download data due to network restrictions, the API falls back to serving a small set of sample predictions and metrics.