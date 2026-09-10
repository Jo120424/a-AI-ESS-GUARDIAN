#!/usr/bin/env python3
"""
Lightweight REST API & Static Server for Predictive ESS Research Platform.
Runs with ZERO external dependencies using Python standard library.
Provides data access to raw, processed, quality profiling reports,
and Step 3 experimental design configurations.
"""

import csv
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
from pathlib import Path

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

FRONTEND_DIR = BASE_DIR / "frontend"
METADATA_DIR = BASE_DIR / "data" / "metadata"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
QUALITY_DIR = BASE_DIR / "data" / "quality_reports"
CONFIG_FILE = BASE_DIR / "experiments" / "configs" / "experiment_config.yaml"
RESULTS_DIR = BASE_DIR / "results" / "step4"
FIGURES_DIR = RESULTS_DIR / "figures"
STEP5_DIR = BASE_DIR / "results" / "step5"
STEP5_FIGURES_DIR = STEP5_DIR / "figures"

PORT = 8000


class ESSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API Routes
        if path == "/api/status":
            self._send_json({
                "status": "online",
                "step": "Step 5 — Research Validation, Robustness, Ablation & Scientific Results",
                "phase": "Step 5 Validated & Audited",
                "tests_passing": 37,
                "dataset_loaded": PROCESSED_DIR.joinpath("dataset_processed.csv").exists(),
                "experiments_completed": RESULTS_DIR.joinpath("comparison.csv").exists(),
                "validation_completed": STEP5_DIR.joinpath("result_audit.csv").exists(),
                "primary_dataset": "NASA Capacitor Electrical Stress Degradation Dataset"
            })
            return

        elif path == "/api/dataset/primary":
            primary_file = METADATA_DIR / "dataset_metadata.json"
            if primary_file.exists():
                with open(primary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_error(404, "Primary dataset metadata not found")
            return

        elif path == "/api/dataset/candidates":
            candidates_file = METADATA_DIR / "candidate_datasets.json"
            if candidates_file.exists():
                with open(candidates_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_error(404, "Candidate datasets metadata not found")
            return

        elif path == "/api/dataset/profile":
            profile_file = QUALITY_DIR / "dataset_profile.json"
            if profile_file.exists():
                with open(profile_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_error(404, "Dataset profile not found")
            return

        elif path == "/api/dataset/processed":
            csv_file = PROCESSED_DIR / "dataset_processed.csv"
            if csv_file.exists():
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                self._send_json(records)
            else:
                self._send_error(404, "Processed dataset not found")
            return

        elif path == "/api/dataset/components":
            csv_file = QUALITY_DIR / "component_statistics.csv"
            if csv_file.exists():
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                self._send_json(records)
            else:
                self._send_error(404, "Component statistics not found")
            return

        elif path == "/api/experiment/ground_truth":
            t_screen = float(query.get("t_screen", [47.0])[0])
            from backend.experiments.ground_truth import generate_ground_truth_labels, load_processed_data
            try:
                df = load_processed_data()
                gt_df = generate_ground_truth_labels(df, t_screen=t_screen)
                raw_records = gt_df.to_dict(orient="records")
                records = [
                    {k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()}
                    for r in raw_records
                ]
                self._send_json(records)
            except Exception as e:
                self._send_error(500, f"Error generating ground truth: {e}")
            return

        elif path == "/api/experiment/config":
            if CONFIG_FILE.exists():
                # Lightweight YAML reader for key-value structure
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    raw_yaml = f.read()
                self._send_json({"raw_yaml": raw_yaml, "status": "configured"})
            else:
                self._send_error(404, "Experiment configuration not found")
            return

        elif path == "/api/experiment/results":
            comp_file = RESULTS_DIR / "comparison.csv"
            if comp_file.exists():
                with open(comp_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                self._send_json(records)
            else:
                self._send_error(404, "Experiment results not found. Run experiment first.")
            return

        elif path == "/api/experiment/component_results":
            comp_res_file = RESULTS_DIR / "component_level_results.csv"
            if comp_res_file.exists():
                with open(comp_res_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                self._send_json(records)
            else:
                self._send_error(404, "Component results not found. Run experiment first.")
            return

        elif path == "/api/experiment/metadata":
            meta_file = RESULTS_DIR / "experiment_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_error(404, "Experiment metadata not found")
            return

        elif path == "/api/experiment/features":
            feat_file = RESULTS_DIR / "feature_importance.csv"
            if feat_file.exists():
                with open(feat_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                self._send_json(records)
            else:
                self._send_error(404, "Feature importance results not found")
            return

        elif path == "/api/experiment/figure":
            fig_name = query.get("name", [""])[0]
            fig_path = FIGURES_DIR / fig_name
            if fig_name and fig_path.exists() and fig_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(fig_path.stat().st_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fig_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._send_error(404, f"Figure not found: {fig_name}")
            return

        elif path == "/api/experiment/run":
            from backend.experiments.run_experiment import run_full_experiment
            try:
                res = run_full_experiment()
                self._send_json({"status": "success", "summary": res["metadata"]})
            except Exception as e:
                self._send_error(500, f"Error running experiment: {e}")
            return

        elif path == "/api/validation/audit":
            csv_path = STEP5_DIR / "result_audit.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Audit results not found")
            return

        elif path == "/api/validation/component_errors":
            csv_path = STEP5_DIR / "component_error_analysis.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Component error analysis not found")
            return

        elif path == "/api/validation/sensitivity":
            csv_path = STEP5_DIR / "threshold_sensitivity.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Threshold sensitivity results not found")
            return

        elif path == "/api/validation/ablation":
            csv_path = STEP5_DIR / "ablation_results.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Ablation results not found")
            return

        elif path == "/api/validation/cost_analysis":
            csv_path = STEP5_DIR / "cost_sensitivity.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Cost analysis results not found")
            return

        elif path == "/api/validation/explainability":
            csv_path = STEP5_DIR / "explainability_summary.csv"
            if csv_path.exists():
                records = self._read_csv_dicts(csv_path)
                self._send_json({"records": records})
            else:
                self._send_error(404, "Explainability summary not found")
            return

        elif path == "/api/validation/figure":
            fig_name = query.get("name", [None])[0]
            fig_path = STEP5_FIGURES_DIR / f"{fig_name}.png"
            if fig_name and fig_path.exists() and fig_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(fig_path.stat().st_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fig_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._send_error(404, f"Step 5 figure not found: {fig_name}")
            return

        # Serve static frontend files
        return super().do_GET()

    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, message):
        self._send_json({"error": message, "status": status}, status=status)

    def _read_csv_dicts(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def run_server(port=PORT):
    handler = ESSRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"==================================================")
            print(f" Predictive ESS Research Platform Server")
            print(f" Serving frontend from: {FRONTEND_DIR}")
            print(f" URL: http://localhost:{port}")
            print(f" API Endpoints:")
            print(f"   - /api/status")
            print(f"   - /api/dataset/primary")
            print(f"   - /api/dataset/candidates")
            print(f"   - /api/dataset/profile")
            print(f"   - /api/dataset/processed")
            print(f"   - /api/dataset/components")
            print(f"   - /api/experiment/ground_truth")
            print(f"   - /api/experiment/config")
            print(f" Press Ctrl+C to terminate.")
            print(f"==================================================")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer gracefully shut down.")
    except Exception as e:
        print(f"Error starting server: {e}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
