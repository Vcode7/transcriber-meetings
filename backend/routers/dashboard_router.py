"""
dashboard_router.py — Standalone developer and backend monitoring console.
Serves HTML dashboard at GET /dashboard and provides diagnostics REST/WebSocket endpoints.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import asyncio
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy import text

from config import settings
from database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# HTML template string containing full CSS and JavaScript console
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiceSum - Backend Developer Console</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #080b11;
            --card-bg: rgba(13, 20, 35, 0.55);
            --border-color: rgba(255, 255, 255, 0.07);
            --ink: #f1f5f9;
            --pencil: #94a3b8;
            --accent: #6366f1; /* purple/indigo */
            --cyan: #06b6d4;
            --emerald: #10b981;
            --amber: #f59e0b;
            --rose: #f43f5e;
            --glow-color: rgba(99, 102, 241, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--ink);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 2rem;
            background: rgba(8, 11, 17, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--accent), var(--cyan));
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            color: #fff;
        }

        .logo-text h1 {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .logo-text span {
            font-size: 0.72rem;
            color: var(--pencil);
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .badge {
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.12);
            color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.2);
            animation: pulse-border 2s infinite;
        }

        .badge-live .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--emerald);
            box-shadow: 0 0 8px var(--emerald);
            animation: pulse-dot 1.5s infinite;
        }

        .badge-offline {
            background: rgba(244, 63, 94, 0.12);
            color: var(--rose);
            border: 1px solid rgba(244, 63, 94, 0.2);
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--ink);
            padding: 7px 14px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }

        .btn-accent {
            background: var(--accent);
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        }

        .btn-accent:hover {
            background: #4f46e5;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        }

        .btn-destructive {
            background: rgba(244, 63, 94, 0.15);
            color: var(--rose);
            border-color: rgba(244, 63, 94, 0.25);
        }

        .btn-destructive:hover {
            background: var(--rose);
            color: #fff;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1.25rem;
            padding: 1.5rem 2rem;
            flex: 1;
        }

        .card {
            grid-column: span 12;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.2rem;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.05);
        }

        @media (min-width: 768px) {
            .col-md-3 { grid-column: span 3; }
            .col-md-4 { grid-column: span 4; }
            .col-md-6 { grid-column: span 6; }
            .col-md-8 { grid-column: span 8; }
            .col-md-9 { grid-column: span 9; }
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--pencil);
        }

        .card-title {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--ink);
        }

        .card-title svg {
            color: var(--accent);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.25rem;
            color: #fff;
            display: flex;
            align-items: baseline;
            gap: 4px;
        }

        .metric-unit {
            font-size: 0.9rem;
            font-weight: 400;
            color: var(--pencil);
        }

        .metric-subtitle {
            font-size: 0.76rem;
            color: var(--pencil);
        }

        .sparkline-container {
            width: 100%;
            height: 55px;
            margin-top: 0.8rem;
            border-radius: 8px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        .sparkline-canvas {
            width: 100%;
            height: 100%;
        }

        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--cyan));
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            text-align: left;
        }

        th, td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            font-weight: 600;
            color: var(--pencil);
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.03em;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-resident {
            background: var(--emerald);
            box-shadow: 0 0 6px var(--emerald);
        }

        .status-offline {
            background: var(--pencil);
            opacity: 0.6;
        }

        /* Terminal Console */
        .console {
            background: #04060b;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 250px;
        }

        .console-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border-color);
        }

        .console-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .console-body {
            flex: 1;
            padding: 10px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            line-height: 1.4;
            overflow-y: auto;
            max-height: 300px;
            white-space: pre-wrap;
        }

        .log-line {
            margin-bottom: 3px;
            border-left: 2px solid transparent;
            padding-left: 6px;
        }

        .log-DEBUG { color: #38bdf8; border-color: #38bdf8; }
        .log-INFO { color: #f1f5f9; border-color: #e2e8f0; }
        .log-WARNING { color: #fbbf24; border-color: #fbbf24; }
        .log-ERROR { color: #f87171; border-color: #f87171; }

        .search-input {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: #fff;
            padding: 4px 10px;
            font-size: 0.78rem;
            outline: none;
            width: 150px;
            transition: all 0.2s ease;
        }

        .search-input:focus {
            border-color: var(--accent);
            background: rgba(255, 255, 255, 0.07);
        }

        .stage-flow {
            display: flex;
            align-items: center;
            gap: 4px;
            margin: 10px 0;
            overflow-x: auto;
            padding-bottom: 6px;
        }

        .stage-node {
            padding: 6px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            font-size: 0.72rem;
            text-align: center;
            flex-shrink: 0;
            transition: all 0.3s ease;
        }

        .stage-node.active {
            background: rgba(6, 182, 212, 0.15);
            border-color: var(--cyan);
            color: var(--cyan);
            font-weight: 600;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.1);
        }

        .stage-node.complete {
            background: rgba(16, 185, 129, 0.1);
            border-color: var(--emerald);
            color: var(--emerald);
        }

        .stage-arrow {
            color: var(--pencil);
            opacity: 0.4;
            font-size: 0.8rem;
        }

        @keyframes pulse-border {
            0% { border-color: rgba(16, 185, 129, 0.2); }
            50% { border-color: rgba(16, 185, 129, 0.5); }
            100% { border-color: rgba(16, 185, 129, 0.2); }
        }

        @keyframes pulse-dot {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.4; }
            100% { transform: scale(1); opacity: 1; }
        }

        .tab-btn {
            font-size: 0.74rem;
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid transparent;
            cursor: pointer;
            background: none;
            color: var(--pencil);
            transition: all 0.2s ease;
        }

        .tab-btn.active {
            color: #fff;
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--border-color);
        }

        .grid-compact td {
            padding: 5px 8px;
            font-size: 0.76rem;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-logo">
            <div class="logo-icon">V</div>
            <div class="logo-text">
                <h1>VoiceSum</h1>
                <span>Backend Developer & Monitoring Console</span>
            </div>
        </div>
        
        <div class="header-actions">
            <div id="connection-badge" class="badge badge-offline">
                <span class="dot"></span>
                <span class="label">DISCONNECTED</span>
            </div>
            
            <div class="badge" style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color)">
                Uptime: <strong id="uptime-val" style="margin-left:4px; color:#fff">00:00:00</strong>
            </div>

            <button class="btn" id="refresh-btn">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                Refresh
            </button>
        </div>
    </header>

    <div class="dashboard-grid">
        
        <!-- Metrics Row -->
        <div class="card col-md-3">
            <div class="card-header">
                <div class="card-title">CPU Utilization</div>
                <span id="cpu-core-count">Cores: 0</span>
            </div>
            <div class="metric-value">
                <span id="cpu-val">0</span><span class="metric-unit">%</span>
            </div>
            <div class="metric-subtitle" id="cpu-proc-val">Process CPU: 0.0%</div>
            <div class="sparkline-container">
                <canvas id="cpu-chart" class="sparkline-canvas"></canvas>
            </div>
        </div>

        <div class="card col-md-3">
            <div class="card-header">
                <div class="card-title">System RAM</div>
                <span id="ram-total-val">0 GB</span>
            </div>
            <div class="metric-value">
                <span id="ram-val">0</span><span class="metric-unit">%</span>
            </div>
            <div class="metric-subtitle" id="ram-proc-val">Process RSS: 0 MB</div>
            <div class="sparkline-container">
                <canvas id="ram-chart" class="sparkline-canvas"></canvas>
            </div>
        </div>

        <div class="card col-md-3">
            <div class="card-header">
                <div class="card-title">GPU core load</div>
                <span id="gpu-name-val">N/A</span>
            </div>
            <div class="metric-value">
                <span id="gpu-val">0</span><span class="metric-unit">%</span>
            </div>
            <div class="metric-subtitle" id="gpu-temp-power">Temp: 0°C · Power: 0W</div>
            <div class="sparkline-container">
                <canvas id="gpu-chart" class="sparkline-canvas"></canvas>
            </div>
        </div>

        <div class="card col-md-3">
            <div class="card-header">
                <div class="card-title">VRAM Usage</div>
                <span id="vram-total-val">0 MB</span>
            </div>
            <div class="metric-value">
                <span id="vram-val">0</span><span class="metric-unit">%</span>
            </div>
            <div class="metric-subtitle" id="vram-used-free">Allocated: 0 MB · Free: 0 MB</div>
            <div class="sparkline-container">
                <canvas id="vram-chart" class="sparkline-canvas"></canvas>
            </div>
        </div>

        <!-- Pipeline and Active Jobs -->
        <div class="card col-md-8">
            <div class="card-header">
                <div class="card-title">Active Processing Pipeline</div>
            </div>
            
            <div class="stage-flow">
                <div class="stage-node" id="stage-preprocess">Audio Preprocessing</div>
                <div class="stage-arrow">→</div>
                <div class="stage-node" id="stage-transcribe">Transcription</div>
                <div class="stage-arrow">→</div>
                <div class="stage-node" id="stage-align">Alignment</div>
                <div class="stage-arrow">→</div>
                <div class="stage-node" id="stage-diarize">Diarization</div>
                <div class="stage-arrow">→</div>
                <div class="stage-node" id="stage-identify">Speaker ID</div>
                <div class="stage-arrow">→</div>
                <div class="stage-node" id="stage-insights">LLM MoM Generation</div>
            </div>

            <div style="margin-top: 10px; flex: 1; overflow-y: auto; max-height: 180px;">
                <table class="grid-compact" id="active-jobs-table">
                    <thead>
                        <tr>
                            <th>Recording Filename</th>
                            <th>Current Stage</th>
                            <th>Active Task</th>
                            <th>Started</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="5" style="text-align: center; color: var(--pencil); font-style: italic;">No active recordings currently processing</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Models Resident Status -->
        <div class="card col-md-4">
            <div class="card-header">
                <div class="card-title">AI Model Resident Manager</div>
            </div>
            <div style="flex: 1; overflow-y: auto; max-height: 250px;">
                <table class="grid-compact" id="models-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Status</th>
                            <th>Device</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Populated by script -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Storage Diagnostics & DB status -->
        <div class="card col-md-4">
            <div class="card-header">
                <div class="card-title">Storage & SQLite Database</div>
            </div>
            <div style="flex: 1; overflow-y: auto; max-height: 280px; display:flex; flex-direction:column; gap:12px;">
                <div>
                    <h4 style="font-size:0.78rem; text-transform:uppercase; color:var(--pencil); margin-bottom:6px">Storage Directories Sizing</h4>
                    <table class="grid-compact" id="storage-table">
                        <tbody>
                            <!-- Sizing stats -->
                        </tbody>
                    </table>
                </div>
                <div>
                    <h4 style="font-size:0.78rem; text-transform:uppercase; color:var(--pencil); margin-bottom:6px">SQLite Metadata Rows</h4>
                    <table class="grid-compact" id="db-table">
                        <tbody>
                            <!-- Row counts -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- FAISS Vector index statistics & Uptime -->
        <div class="card col-md-4">
            <div class="card-header">
                <div class="card-title">Vector Store Indexes (FAISS)</div>
            </div>
            <div style="flex: 1; overflow-y: auto; max-height: 280px; display:flex; flex-direction:column; gap:12px;">
                <table class="grid-compact" id="vector-table">
                    <thead>
                        <tr>
                            <th>Index Scope</th>
                            <th>Indices</th>
                            <th>Vectors</th>
                            <th>Size</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Vector counts -->
                    </tbody>
                </table>
                
                <div style="border-top:1px solid var(--border-color); padding-top:10px;">
                    <h4 style="font-size:0.78rem; text-transform:uppercase; color:var(--pencil); margin-bottom:6px">System Platform Spec</h4>
                    <div style="font-size:0.76rem; line-height:1.4; color:var(--pencil);">
                        <div>OS platform: <strong style="color:var(--ink)" id="os-platform">Loading...</strong></div>
                        <div>Python build: <strong style="color:var(--ink)" id="py-version">Loading...</strong></div>
                        <div>Asyncio Loop Tasks: <strong style="color:var(--ink)" id="asyncio-count">0</strong> active</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Maintenance panel -->
        <div class="card col-md-4">
            <div class="card-header">
                <div class="card-title">Maintenance console</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; flex: 1;">
                <button class="btn" onclick="triggerMaintenance('clear_cuda_cache')">Clear CUDA Cache</button>
                <button class="btn" onclick="triggerMaintenance('clear_temp_files')">Clear Temp files</button>
                <button class="btn" onclick="triggerMaintenance('clear_vector_cache')">Clear Vector Cache</button>
                <button class="btn" onclick="triggerMaintenance('reload_models')">Reload Models</button>
                <button class="btn btn-destructive" style="grid-column: span 2" onclick="triggerMaintenance('unload_models')">Unload Resident Models</button>
                
                <div style="grid-column: span 2; display: flex; gap: 8px; margin-top: 6px;">
                    <a href="/dashboard/api/diagnostics/export" class="btn" style="flex:1; justify-content:center; text-decoration:none">Export Diagnostics</a>
                    <a href="/dashboard/api/diagnostics/logs/download" class="btn btn-accent" style="flex:1; justify-content:center; text-decoration:none">Download Logs</a>
                </div>
            </div>
        </div>

        <!-- API Endpoint Explorer -->
        <div class="card col-md-6">
            <div class="card-header">
                <div class="card-title">FastAPI API Endpoint Explorer</div>
                <input type="text" id="route-search" class="search-input" placeholder="Filter routes..." style="width: 140px; padding: 2px 8px">
            </div>
            <div style="flex: 1; overflow-y: auto; max-height: 250px;">
                <table class="grid-compact" id="endpoints-table">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Route Endpoint</th>
                            <th>Details Summary</th>
                            <th>Auth</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- List of endpoints -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Processing performance benchmarks -->
        <div class="card col-md-6">
            <div class="card-header">
                <div class="card-title">Pipeline performance timings</div>
            </div>
            <div style="flex:1; overflow-y:auto; max-height:250px;">
                <table class="grid-compact" id="perf-table">
                    <thead>
                        <tr>
                            <th>Pipeline Phase</th>
                            <th>Average Execution Duration</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Performance times -->
                    </tbody>
                </table>
                
                <div style="margin-top:10px; border-top:1px solid var(--border-color); padding-top:10px;">
                    <h4 style="font-size:0.78rem; text-transform:uppercase; color:var(--pencil); margin-bottom:6px">Active runtime parameters</h4>
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:6px; font-size:0.75rem; color:var(--pencil)">
                        <div>Chunk size: <strong style="color:var(--ink)" id="cfg-chunk">--</strong></div>
                        <div>Overlap: <strong style="color:var(--ink)" id="cfg-overlap">--</strong></div>
                        <div>Top-K context: <strong style="color:var(--ink)" id="cfg-k">--</strong></div>
                        <div>Similarity cut: <strong style="color:var(--ink)" id="cfg-cutoff">--</strong></div>
                        <div>LLM Model: <strong style="color:var(--ink)" id="cfg-llm">--</strong></div>
                        <div>Offline mode: <strong style="color:var(--ink)" id="cfg-offline">--</strong></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Logs Terminal Console -->
        <div class="card col-md-12">
            <div class="card-header">
                <div class="card-title">Live Server Log Streamer</div>
                
                <div class="console-actions">
                    <input type="text" id="log-search" class="search-input" placeholder="Search string...">
                    
                    <button class="tab-btn active" id="log-filter-all" onclick="setLogLevelFilter('ALL')">ALL</button>
                    <button class="tab-btn" id="log-filter-info" onclick="setLogLevelFilter('INFO')">INFO</button>
                    <button class="tab-btn" id="log-filter-warning" onclick="setLogLevelFilter('WARNING')">WARN</button>
                    <button class="tab-btn" id="log-filter-error" onclick="setLogLevelFilter('ERROR')">ERROR</button>
                    <button class="tab-btn" id="log-filter-debug" onclick="setLogLevelFilter('DEBUG')">DEBUG</button>
                    
                    <button class="btn" id="log-pause-btn" style="padding: 2px 8px; font-size:0.7rem">Pause</button>
                    <button class="btn" id="log-clear-btn" style="padding: 2px 8px; font-size:0.7rem">Clear</button>
                </div>
            </div>
            
            <div class="console">
                <div class="console-body" id="log-output-console">
                    <!-- Logs stream in here -->
                </div>
            </div>
        </div>

    </div>

    <script>
        // Custom Realtime Line Chart using HTML5 Canvas API
        class SparklineChart {
            constructor(canvasId, color = '#6366f1') {
                this.canvas = document.getElementById(canvasId);
                if (!this.canvas) return;
                this.ctx = this.canvas.getContext('2d');
                this.maxPoints = 40;
                this.color = color;
                this.data = [];
            }
            
            addPoint(val) {
                this.data.push(val);
                if (this.data.length > this.maxPoints) {
                    this.data.shift();
                }
                this.draw();
            }
            
            draw() {
                const ctx = this.ctx;
                const width = this.canvas.width = this.canvas.clientWidth * 2; // high-dpi
                const height = this.canvas.height = this.canvas.clientHeight * 2;
                
                ctx.clearRect(0, 0, width, height);
                if (this.data.length < 2) return;
                
                ctx.scale(2, 2);
                const w = width / 2;
                const h = height / 2;
                
                // Draw grid line at 50%
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, h / 2);
                ctx.lineTo(w, h / 2);
                ctx.stroke();
                
                // Draw chart line
                ctx.beginPath();
                ctx.strokeStyle = this.color;
                ctx.lineWidth = 2;
                ctx.shadowColor = this.color;
                ctx.shadowBlur = 4;
                
                const step = w / (this.maxPoints - 1);
                for (let i = 0; i < this.data.length; i++) {
                    const x = i * step;
                    // map 0..100 to canvas height
                    const y = h - (this.data[i] / 100) * h * 0.85 - (h * 0.07);
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
                ctx.shadowBlur = 0; // reset
                
                // Area fill
                ctx.lineTo((this.data.length - 1) * step, h);
                ctx.lineTo(0, h);
                ctx.closePath();
                const grad = ctx.createLinearGradient(0, 0, 0, h);
                grad.addColorStop(0, this.color + '22');
                grad.addColorStop(1, this.color + '00');
                ctx.fillStyle = grad;
                ctx.fill();
            }
        }

        // Initialize Realtime charts
        const charts = {
            cpu: new SparklineChart('cpu-chart', '#06b6d4'),
            ram: new SparklineChart('ram-chart', '#a855f7'),
            gpu: new SparklineChart('gpu-chart', '#ec4899'),
            vram: new SparklineChart('vram-chart', '#6366f1')
        };

        // State trackers
        let allLogs = [];
        let logFilterLevel = 'ALL';
        let logPaused = false;
        let endpoints = [];
        let socket = null;

        // Formats file sizes
        function formatBytes(bytes, decimals = 2) {
            if (!bytes || bytes === 0) return '0 Bytes';
            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
        }

        // Uptime Formatter
        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return [h, m, s].map(v => v.toString().padStart(2, '0')).join(':');
        }

        // Initialize Websocket status stream
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/dashboard/ws`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = () => {
                const badge = document.getElementById('connection-badge');
                badge.className = 'badge badge-live';
                badge.querySelector('.label').textContent = 'LIVE FEED';
            };
            
            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'initial_logs') {
                    allLogs = data.logs;
                    renderLogs();
                } else if (data.type === 'update') {
                    updateStatus(data.status);
                    if (data.new_logs && data.new_logs.length > 0) {
                        allLogs = allLogs.concat(data.new_logs);
                        if (allLogs.length > 2000) {
                            allLogs = allLogs.slice(allLogs.length - 2000);
                        }
                        renderLogs();
                    }
                }
            };
            
            socket.onclose = () => {
                const badge = document.getElementById('connection-badge');
                badge.className = 'badge badge-offline';
                badge.querySelector('.label').textContent = 'FALLBACK POLLING';
                // Fallback: trigger polling
                setTimeout(initWebSocket, 5000);
            };
        }

        // Poll API status fallback
        async function loadStatus() {
            try {
                const res = await fetch('/dashboard/api/status');
                const data = await res.json();
                updateStatus(data);
            } catch (e) {
                console.error("Failed to load status API", e);
            }
        }

        async function loadEndpoints() {
            try {
                const res = await fetch('/dashboard/api/endpoints');
                const data = await res.json();
                endpoints = data.endpoints || [];
                renderEndpoints();
            } catch (e) {
                console.error("Failed to load routes explorer", e);
            }
        }

        async function triggerMaintenance(action) {
            const label = action.replace(/_/g, ' ');
            if (action === 'unload_models' && !confirm("Warning: This will immediately drop resident model weights from RAM/VRAM. The next request to transcribe or summarize will experience latency during reloading. Proceed?")) {
                return;
            }
            try {
                const res = await fetch('/dashboard/api/maintenance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action })
                });
                const data = await res.json();
                alert(data.message || "Maintenance action triggered successfully!");
                loadStatus();
            } catch (e) {
                alert("Failed to run maintenance: " + e);
            }
        }

        // Render UI panels from data
        function updateStatus(status) {
            if (!status) return;

            // Uptime & System info
            document.getElementById('uptime-val').textContent = formatUptime(status.system.uptime);
            document.getElementById('cpu-val').textContent = status.system.cpu_percent.toFixed(1);
            document.getElementById('cpu-core-count').textContent = `Cores: ${status.system.cpu_cores_physical}P / ${status.system.cpu_cores_logical}L`;
            document.getElementById('cpu-proc-val').textContent = `Process CPU: ${status.system.process_cpu.toFixed(1)}%`;
            
            document.getElementById('ram-val').textContent = status.system.ram_percent.toFixed(1);
            document.getElementById('ram-total-val').textContent = `${(status.system.ram_total / (1024**3)).toFixed(1)} GB`;
            document.getElementById('ram-proc-val').textContent = `Process RSS: ${(status.system.process_ram / (1024**2)).toFixed(0)} MB`;
            
            // Add Sparkline Data
            charts.cpu.addPoint(status.system.cpu_percent);
            charts.ram.addPoint(status.system.ram_percent);

            // GPU Monitor
            if (status.gpu.cuda_available) {
                document.getElementById('gpu-val').textContent = status.gpu.utilization.toFixed(0);
                document.getElementById('gpu-name-val').textContent = status.gpu.name;
                document.getElementById('gpu-temp-power').textContent = `Temp: ${status.gpu.temperature}°C · Power: ${status.gpu.power_draw.toFixed(0)}W`;
                
                document.getElementById('vram-val').textContent = ((status.gpu.memory_used / status.gpu.memory_total) * 100).toFixed(0);
                document.getElementById('vram-total-val').textContent = `${status.gpu.memory_total.toFixed(0)} MB`;
                document.getElementById('vram-used-free').textContent = `Allocated: ${status.gpu.memory_used.toFixed(0)} MB · Free: ${status.gpu.memory_free.toFixed(0)} MB`;
                
                charts.gpu.addPoint(status.gpu.utilization);
                charts.vram.addPoint((status.gpu.memory_used / status.gpu.memory_total) * 100);
            } else {
                document.getElementById('gpu-name-val').textContent = 'CUDA Not Available';
                document.getElementById('gpu-temp-power').textContent = 'Offline fallback';
                document.getElementById('vram-total-val').textContent = 'N/A';
                document.getElementById('vram-used-free').textContent = 'Using system CPU RAM';
                charts.gpu.addPoint(0);
                charts.vram.addPoint(0);
            }

            // Platform Specs
            document.getElementById('os-platform').textContent = status.system.os;
            document.getElementById('py-version').textContent = status.system.python.split('\\n')[0];
            document.getElementById('asyncio-count').textContent = status.asyncio_tasks;

            // Storage Sizing Table
            const storageBody = [];
            for (const [key, meta] of Object.entries(status.storage)) {
                storageBody.push(`
                    <tr>
                        <td style="font-weight:600; text-transform:capitalize">${key.replace('_', ' ')}</td>
                        <td>${meta.files} files</td>
                        <td style="text-align:right">${formatBytes(meta.size)}</td>
                    </tr>
                `);
            }
            document.getElementById('storage-table').innerHTML = storageBody.join('');

            // DB stats
            const dbBody = [
                `<tr><td>SQLite path</td><td colspan="2" style="font-family:monospace; word-break:break-all; font-size:0.68rem">${status.database.path}</td></tr>`,
                `<tr><td>Total File size</td><td colspan="2">${formatBytes(status.database.size)}</td></tr>`
            ];
            for (const [table, count] of Object.entries(status.database.tables)) {
                dbBody.push(`
                    <tr>
                        <td style="font-family:monospace; padding-left:15px">${table}</td>
                        <td colspan="2" style="text-align:right; font-weight:600">${count} rows</td>
                    </tr>
                `);
            }
            document.getElementById('db-table').innerHTML = dbBody.join('');

            // Vector Sizing
            const vectorBody = [];
            for (const [scope, data] of Object.entries(status.vector_store)) {
                if (scope === 'total_size') continue;
                vectorBody.push(`
                    <tr>
                        <td style="font-weight:600; text-transform:capitalize">${scope.replace('_', ' ')}</td>
                        <td>${data.count} indexes</td>
                        <td>${data.vectors} items</td>
                        <td style="text-align:right">${formatBytes(data.size)}</td>
                    </tr>
                `);
            }
            vectorBody.push(`
                <tr style="border-top:1.5px solid var(--border-color); font-weight:700">
                    <td>Total Size</td>
                    <td colspan="3" style="text-align:right">${formatBytes(status.vector_store.total_size)}</td>
                </tr>
            `);
            document.getElementById('vector-table').querySelector('tbody').innerHTML = vectorBody.join('');

            // Active processing jobs
            const activeJobs = status.active_jobs;
            const nodes = {
                preprocess: document.getElementById('stage-preprocess'),
                transcribe: document.getElementById('stage-transcribe'),
                align: document.getElementById('stage-align'),
                diarize: document.getElementById('stage-diarize'),
                identify: document.getElementById('stage-identify'),
                insights: document.getElementById('stage-insights')
            };
            
            // Reset active nodes class
            Object.values(nodes).forEach(n => n.className = 'stage-node');

            if (activeJobs.length === 0) {
                document.getElementById('active-jobs-table').querySelector('tbody').innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; color: var(--pencil); font-style: italic;">No active recordings currently processing</td>
                    </tr>
                `;
            } else {
                const jobRows = activeJobs.map(job => {
                    // Set active visualizer stage node
                    const prog = job.progress || '';
                    if (prog === 'transcribing') nodes.transcribe.className = 'stage-node active';
                    else if (prog === 'diarizing') nodes.diarize.className = 'stage-node active';
                    else if (prog === 'identifying_speakers') nodes.identify.className = 'stage-node active';
                    else if (prog === 'generating_mom') nodes.insights.className = 'stage-node active';
                    else if (prog === 'summarizing_chunks') nodes.insights.className = 'stage-node active';
                    
                    const isTask = job.is_active ? `<span class="badge" style="background:rgba(6,180,212,0.12); color:var(--cyan); padding:2px 6px">Task Async</span>` : `<span class="badge" style="background:rgba(255,255,255,0.03); color:var(--pencil); padding:2px 6px">Queued</span>`;
                    const dateStr = job.created_at ? new Date(job.created_at).toLocaleTimeString() : '—';
                    
                    return `
                        <tr>
                            <td style="font-weight:600; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">${job.filename}</td>
                            <td style="color:var(--cyan); font-weight:600">${job.stage}</td>
                            <td>${isTask}</td>
                            <td>${dateStr}</td>
                            <td>
                                <button class="btn btn-destructive" style="padding:2px 6px; font-size:0.7rem" onclick="cancelProcessing('${job.id}')">Cancel</button>
                            </td>
                        </tr>
                    `;
                });
                document.getElementById('active-jobs-table').querySelector('tbody').innerHTML = jobRows.join('');
            }

            // AI Model list
            const modelRows = status.models.map(m => {
                const dotClass = m.status === 'Resident' || m.status.startsWith('Active') ? 'status-resident' : 'status-offline';
                return `
                    <tr>
                        <td style="font-weight:600">${m.name}</td>
                        <td>
                            <span class="status-dot ${dotClass}" style="margin-right:6px"></span>
                            ${m.status}
                        </td>
                        <td style="font-family:monospace">${m.device}</td>
                        
                    </tr>
                `;
            });
            document.getElementById('models-table').querySelector('tbody').innerHTML = modelRows.join('');

            // Performance table
            const perfRows = [
                `<tr><td>Speech Audio Transcription</td><td style="font-weight:600; text-align:right">${status.performance.avg_transcription.toFixed(1)}s avg</td></tr>`,
                `<tr><td>Text Alignment & Word-level Sync</td><td style="font-weight:600; text-align:right">${status.performance.avg_alignment.toFixed(1)}s avg</td></tr>`,
                `<tr><td>Speaker Region Diarization</td><td style="font-weight:600; text-align:right">${status.performance.avg_diarization.toFixed(1)}s avg</td></tr>`,
                `<tr><td>Speaker Voice Identification</td><td style="font-weight:600; text-align:right">${status.performance.avg_identification.toFixed(1)}s avg</td></tr>`,
                `<tr><td>LLM Document and Minutes Generation</td><td style="font-weight:600; text-align:right">${status.performance.avg_llm.toFixed(1)}s avg</td></tr>`,
                `<tr style="border-top:1.5px solid var(--border-color); font-weight:700"><td>Total Job End-to-End Pipeline</td><td style="text-align:right">${status.performance.avg_total.toFixed(1)}s avg</td></tr>`
            ];
            document.getElementById('perf-table').querySelector('tbody').innerHTML = perfRows.join('');

            // Runtime configs
            document.getElementById('cfg-chunk').textContent = status.config.chunk_size;
            document.getElementById('cfg-overlap').textContent = status.config.chunk_overlap;
            document.getElementById('cfg-k').textContent = `G:${status.config.k_global}/M:${status.config.k_meeting}/T:${status.config.k_transcript}`;
            document.getElementById('cfg-cutoff').textContent = status.config.similarity_cutoff;
            document.getElementById('cfg-llm').textContent = status.config.llm_model;
            document.getElementById('cfg-offline').textContent = status.config.offline_mode ? '100% OFFLINE' : 'ONLINE';
        }

        async function cancelProcessing(recordingId) {
            if (confirm("Confirm cancelling recording pipeline task? This will terminate transcription/insights immediately.")) {
                try {
                    const res = await fetch(`/dashboard/api/maintenance`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action: 'cancel_task', data: recordingId })
                    });
                    const data = await res.json();
                    alert(data.message);
                    loadStatus();
                } catch(e) {
                    alert("Failed to cancel: " + e);
                }
            }
        }

        // Endpoint explorer table render
        function renderEndpoints() {
            const query = document.getElementById('route-search').value.toLowerCase();
            const filtered = endpoints.filter(e => 
                e.path.toLowerCase().includes(query) || 
                e.name.toLowerCase().includes(query) ||
                (e.description && e.description.toLowerCase().includes(query))
            );
            
            const rows = filtered.map(e => {
                const methodsBadges = e.methods.map(m => {
                    let color = 'var(--pencil)';
                    if (m === 'GET') color = 'var(--cyan)';
                    else if (m === 'POST') color = 'var(--accent)';
                    else if (m === 'DELETE') color = 'var(--rose)';
                    return `<span class="badge" style="background:rgba(255,255,255,0.03); border:1px solid ${color}; color:${color}; font-size:0.65rem; padding:1px 5px">${m}</span>`;
                }).join(' ');
                
                const authBadge = e.auth_required 
                    ? `<span class="badge" style="background:rgba(245,158,11,0.1); color:var(--amber)">JWT</span>`
                    : `<span class="badge" style="background:rgba(16,185,129,0.1); color:var(--emerald)">Public</span>`;
                
                return `
                    <tr>
                        <td style="white-space:nowrap">${methodsBadges}</td>
                        <td style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#fff">${e.path}</td>
                        <td style="color:var(--pencil); font-size:0.75rem">${e.description || e.name}</td>
                        <td>${authBadge}</td>
                    </tr>
                `;
            });
            document.getElementById('endpoints-table').querySelector('tbody').innerHTML = rows.length > 0 ? rows.join('') : `
                <tr><td colspan="4" style="text-align:center; color:var(--pencil)">No endpoints match the search</td></tr>
            `;
        }

        // Live Log viewer handling
        function setLogLevelFilter(level) {
            logFilterLevel = level;
            // Update tab UI
            ['all', 'info', 'warning', 'error', 'debug'].forEach(id => {
                document.getElementById(`log-filter-${id}`).className = 'tab-btn';
            });
            document.getElementById(`log-filter-${level.toLowerCase()}`).className = 'tab-btn active';
            renderLogs();
        }

        function renderLogs() {
            if (logPaused) return;
            
            const searchQuery = document.getElementById('log-search').value.toLowerCase();
            const consoleEl = document.getElementById('log-output-console');
            
            const filtered = allLogs.filter(log => {
                // Filter level
                if (logFilterLevel !== 'ALL' && log.level !== logFilterLevel) return false;
                // Filter search query
                if (searchQuery && !log.message.toLowerCase().includes(searchQuery) && !log.name.toLowerCase().includes(searchQuery)) return false;
                return true;
            });

            const lines = filtered.map(log => {
                return `<div class="log-line log-${log.level}">${log.timestamp} [${log.level}] ${log.name} — ${log.message}</div>`;
            });

            consoleEl.innerHTML = lines.join('');
            
            // Auto scroll to bottom
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        // Setup Event Listeners
        document.getElementById('refresh-btn').addEventListener('click', () => {
            loadStatus();
            loadEndpoints();
        });

        document.getElementById('route-search').addEventListener('input', renderEndpoints);
        document.getElementById('log-search').addEventListener('input', renderLogs);
        
        const pauseBtn = document.getElementById('log-pause-btn');
        pauseBtn.addEventListener('click', () => {
            logPaused = !logPaused;
            pauseBtn.textContent = logPaused ? 'Resume' : 'Pause';
            if (!logPaused) renderLogs();
        });

        document.getElementById('log-clear-btn').addEventListener('click', () => {
            allLogs = [];
            renderLogs();
        });

        // Initialize dashboard loading
        loadStatus();
        loadEndpoints();
        initWebSocket();
        
        // Polling fallback interval
        setInterval(() => {
            if (!socket || socket.readyState !== WebSocket.OPEN) {
                loadStatus();
            }
        }, 3000);
        
    </script>
</body>
</html>
"""


async def get_active_jobs():
    """Query recordings table for jobs currently in pending/processing state."""
    from tasks.pipeline import active_tasks
    jobs = []
    async with get_db() as db:
        try:
            res = await db.execute(text(
                "SELECT id, filename, duration, status, progress, created_at FROM recordings "
                "WHERE status IN ('pending', 'processing')"
            ))
            rows = res.fetchall()
            for row in rows:
                rid = row[0]
                filename = row[1]
                duration = row[2]
                status = row[3]
                progress = row[4]
                created_at = row[5]

                # Map progress stage tags
                progress_map = {
                    "transcribing": "Transcribing Audio (WhisperX)",
                    "diarizing": "Diarizing Speakers (Pyannote)",
                    "identifying_speakers": "Identifying Speaker Voices (ECAPA-TDNN)",
                    "generating_mom": "Generating Minutes of Meeting (Qwen LLM)",
                    "summarizing_chunks": "Summarizing Transcript Segments (Qwen LLM)",
                }
                stage_desc = progress_map.get(progress, "Queued" if status == "pending" else "Processing")
                is_active = rid in active_tasks

                jobs.append({
                    "id": rid,
                    "filename": filename,
                    "duration": duration,
                    "status": status,
                    "progress": progress,
                    "stage": stage_desc,
                    "created_at": created_at,
                    "is_active": is_active
                })
        except Exception as e:
            logger.warning(f"[Dashboard] Failed to query active recordings: {e}")
    return jobs


def get_dir_size_and_count(dir_path: str | Path) -> tuple[int, int]:
    """Calculate directory size in bytes and file count recursive."""
    total_size = 0
    file_count = 0
    path = Path(dir_path)
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1

    for root, dirs, files in os.walk(str(path)):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
                file_count += 1
            except Exception:
                pass
    return total_size, file_count


async def get_db_stats() -> dict:
    """Return database rows stats and size."""
    stats = {}
    tables = [
        "users", "sessions", "login_attempts", "voice_profiles", "recordings",
        "recording_attachments", "global_context_documents", "dictionary"
    ]
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    db_abs_path = os.path.abspath(db_path)
    db_size = os.path.getsize(db_abs_path) if os.path.exists(db_abs_path) else 0

    stats["path"] = db_abs_path
    stats["size"] = db_size
    stats["tables"] = {}

    async with get_db() as db:
        for t in tables:
            try:
                res = await db.execute(text(f"SELECT COUNT(*) FROM {t}"))
                stats["tables"][t] = res.scalar() or 0
            except Exception:
                stats["tables"][t] = 0
    return stats


def get_vector_store_stats() -> dict:
    """Scan and compute metadata row counts across all vector directories."""
    store_dir = Path(settings.VECTOR_STORE_DIR)
    stats = {
        "global_context": {"count": 0, "vectors": 0, "size": 0},
        "meeting_context": {"count": 0, "vectors": 0, "size": 0},
        "transcript": {"count": 0, "vectors": 0, "size": 0},
        "total_size": 0
    }

    if not store_dir.exists():
        return stats

    for p in store_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        dir_size = sum(f.stat().st_size for f in p.glob('*') if f.is_file())
        stats["total_size"] += dir_size

        # Read meta file
        meta_file = p / "index_meta.json"
        vector_count = 0
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    vector_count = len(meta)
            except Exception:
                pass

        if name.startswith("global_context_"):
            stats["global_context"]["count"] += 1
            stats["global_context"]["vectors"] += vector_count
            stats["global_context"]["size"] += dir_size
        elif name.startswith("meeting_"):
            stats["meeting_context"]["count"] += 1
            stats["meeting_context"]["vectors"] += vector_count
            stats["meeting_context"]["size"] += dir_size
        elif name.startswith("transcript_"):
            stats["transcript"]["count"] += 1
            stats["transcript"]["vectors"] += vector_count
            stats["transcript"]["size"] += dir_size

    return stats


async def get_performance_metrics() -> dict:
    """Retrieve averages for transcription, alignment, diarization, and LLM steps."""
    metrics = {
        "avg_total": 0.0,
        "avg_transcription": 0.0,
        "avg_alignment": 0.0,
        "avg_diarization": 0.0,
        "avg_identification": 0.0,
        "avg_llm": 0.0,
        "count": 0
    }
    async with get_db() as db:
        try:
            res = await db.execute(text("""
                SELECT 
                    AVG(total_pipeline_sec),
                    AVG(transcription_sec),
                    AVG(alignment_sec),
                    AVG(diarization_sec),
                    AVG(identification_sec),
                    AVG(llm_mom_sec),
                    COUNT(*)
                FROM processing_analytics
                WHERE final_status = 'success'
            """))
            row = res.fetchone()
            if row and row[6] > 0:
                metrics["avg_total"] = round(row[0] or 0.0, 2)
                metrics["avg_transcription"] = round(row[1] or 0.0, 2)
                metrics["avg_alignment"] = round(row[2] or 0.0, 2)
                metrics["avg_diarization"] = round(row[3] or 0.0, 2)
                metrics["avg_identification"] = round(row[4] or 0.0, 2)
                metrics["avg_llm"] = round(row[5] or 0.0, 2)
                metrics["count"] = row[6]
        except Exception:
            pass
    return metrics


async def get_recent_activity() -> List[dict]:
    """Retrieve status timelines for the last 10 recordings."""
    activity = []
    async with get_db() as db:
        try:
            res = await db.execute(text(
                "SELECT id, filename, status, created_at FROM recordings "
                "ORDER BY created_at DESC LIMIT 10"
            ))
            rows = res.fetchall()
            for r in rows:
                activity.append({
                    "id": r[0],
                    "filename": r[1],
                    "status": r[2],
                    "timestamp": r[3]
                })
        except Exception:
            pass
    return activity


async def get_status_payload() -> dict:
    """Build system, GPU, models, storage, DB, FAISS, and activity status payload."""
    import psutil
    import platform
    import torch
    from main import STARTUP_TIME

    # 1. System specs
    cpu_logical = psutil.cpu_count()
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent()

    ram = psutil.virtual_memory()
    ram_total = ram.total
    ram_used = ram.used
    ram_percent = ram.percent

    proc = psutil.Process(os.getpid())
    proc_ram = proc.memory_info().rss
    proc_cpu = proc.cpu_percent()

    uptime = time.time() - STARTUP_TIME
    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    python_version = sys.version

    # 2. GPU spec (via nvidia-smi query)
    gpu_info = {
        "cuda_available": False,
        "name": "",
        "utilization": 0.0,
        "memory_used": 0.0,
        "memory_total": 0.0,
        "memory_free": 0.0,
        "temperature": 0.0,
        "power_draw": 0.0,
        "driver_version": "",
        "cuda_version": ""
    }

    if torch.cuda.is_available():
        gpu_info["cuda_available"] = True
        try:
            # Query nvidia-smi command directly
            cmd = ["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True).strip()
            parts = [p.strip() for p in out.split(",")]
            gpu_info["name"] = parts[0]
            gpu_info["driver_version"] = parts[1]
            gpu_info["utilization"] = float(parts[2])
            gpu_info["memory_used"] = float(parts[3])
            gpu_info["memory_total"] = float(parts[4])
            gpu_info["memory_free"] = float(parts[5])
            gpu_info["temperature"] = float(parts[6])
            gpu_info["power_draw"] = float(parts[7])
        except Exception:
            # Fallback PyTorch values
            try:
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_used"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
                gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_info["memory_free"] = gpu_info["memory_total"] - gpu_info["memory_used"]
            except Exception:
                pass

    # 3. Model management
    whisper_loaded = False
    if "services.transcription" in sys.modules:
        from services.transcription import _whisperx_model
        whisper_loaded = _whisperx_model is not None

    diarization_loaded = False
    if "services.diarization" in sys.modules:
        from services.diarization import _diarization_pipeline
        diarization_loaded = _diarization_pipeline is not None

    ecapa_loaded = False
    if "services.embedding" in sys.modules:
        from services.embedding import _ecapa_classifier
        ecapa_loaded = _ecapa_classifier is not None

    align_loaded = False
    align_cache_size = 0
    if "services.transcription" in sys.modules:
        from services.transcription import _align_model_cache
        align_loaded = len(_align_model_cache) > 0
        align_cache_size = len(_align_model_cache)

    llm_loaded = False
    if "services.ai_provider" in sys.modules:
        from services.ai_provider import QwenProvider
        llm_loaded = QwenProvider._model is not None or QwenProvider._pipeline is not None

    embedder_loaded = False
    if "services.text_embedding_service" in sys.modules:
        from services.text_embedding_service import _embedder
        embedder_loaded = _embedder is not None and _embedder._loaded

    overlap_loaded = False
    if "main" in sys.modules:
        import main
        overlap_loaded = getattr(main, "_overlap_model", None) is not None

    models = [
        {"name": "Transcription", "key": "whisper", "status": "Resident" if whisper_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and whisper_loaded else "CPU", "size": "Transcription Model"},
        {"name": "Diarization", "key": "diarization", "status": "Resident" if diarization_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and diarization_loaded else "CPU", "size": "Diarization Model"},
        {"name": "Speaker Identification", "key": "ecapa", "status": "Resident" if ecapa_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and ecapa_loaded else "CPU", "size": "Speaker ID Model"},
        {"name": "Alignment Cache", "key": "align", "status": f"Active ({align_cache_size})" if align_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and align_loaded else "CPU", "size": "Alignment Model"},
        {"name": "Language Model", "key": "local_llm", "status": "Resident" if llm_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and llm_loaded else "CPU", "size": "LLM Model"},
        {"name": "Text Embeddings", "key": "local_embeddings", "status": "Resident" if embedder_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and embedder_loaded else "CPU", "size": "Embedding Model"},
        {"name": "Overlap Classification", "key": "overlap", "status": "Resident" if overlap_loaded else "Offline", "device": "CUDA" if gpu_info["cuda_available"] and overlap_loaded else "CPU", "size": "Overlap Model"}
    ]

    # 4. Sizing
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    models_size, models_files = get_dir_size_and_count(settings.MODELS_DIR)
    uploads_size, uploads_files = get_dir_size_and_count(settings.UPLOAD_DIR)
    vector_size, vector_files = get_dir_size_and_count(settings.VECTOR_STORE_DIR)
    db_size, _ = get_dir_size_and_count(db_path)
    log_size, _ = get_dir_size_and_count("voicesum.log")

    storage = {
        "models": {"size": models_size, "files": models_files},
        "uploads": {"size": uploads_size, "files": uploads_files},
        "vector_store": {"size": vector_size, "files": vector_files},
        "database": {"size": db_size, "files": 1 if db_size else 0},
        "logs": {"size": log_size, "files": 1 if log_size else 0}
    }

    # 5. Database, FAISS, analytics summaries
    db_stats = await get_db_stats()
    vector_stats = get_vector_store_stats()
    perf = await get_performance_metrics()
    recent_activity = await get_recent_activity()
    active_jobs = await get_active_jobs()

    config = {
        "chunk_size": settings.RAG_CHUNK_SIZE,
        "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
        "k_global": settings.RAG_RETRIEVAL_K_GLOBAL,
        "k_meeting": settings.RAG_RETRIEVAL_K_MEETING,
        "k_transcript": settings.RAG_RETRIEVAL_K_TRANSCRIPT,
        "similarity_cutoff": settings.RAG_RELATIVE_SCORE_CUTOFF,
        "whisper_device": settings.WHISPER_DEVICE,
        "whisper_compute_type": settings.WHISPER_COMPUTE_TYPE,
        "llm_model": "LLM Model",
        "embedding_model": "Embedding Model",
        "speaker_threshold": settings.SPEAKER_SIMILARITY_THRESHOLD,
        "min_segment_dur": settings.MIN_SEGMENT_DURATION,
        "audio_preprocess": settings.AUDIO_PREPROCESS_BEFORE_ALIGNMENT,
        "offline_mode": settings.OFFLINE_MODE
    }

    return {
        "system": {
            "os": os_info,
            "python": python_version,
            "cpu_percent": cpu_percent,
            "cpu_cores_logical": cpu_logical,
            "cpu_cores_physical": cpu_physical,
            "ram_total": ram_total,
            "ram_used": ram_used,
            "ram_percent": ram_percent,
            "process_ram": proc_ram,
            "process_cpu": proc_cpu,
            "uptime": uptime
        },
        "gpu": gpu_info,
        "models": models,
        "active_jobs": active_jobs,
        "storage": storage,
        "database": db_stats,
        "vector_store": vector_stats,
        "performance": perf,
        "activity": recent_activity,
        "config": config,
        "asyncio_tasks": len(asyncio.all_tasks())
    }


@router.get("", response_class=HTMLResponse)
def get_dashboard_html():
    """Serves the standalone CSS grid control dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@router.get("/api/status")
async def get_status():
    """Returns granular metrics diagnostics."""
    payload = await get_status_payload()
    return JSONResponse(content=payload)


@router.get("/api/logs")
def get_logs(level: str = "ALL", search: str = "", limit: int = 500):
    """Query in-memory logs collection."""
    from main import in_memory_log_handler
    all_logs = list(in_memory_log_handler.records)
    filtered = []
    
    # filter log objects
    for log in all_logs:
        if level != "ALL" and log["level"] != level:
            continue
        if search:
            search_lower = search.lower()
            if search_lower not in log["message"].lower() and search_lower not in log["name"].lower():
                continue
        filtered.append(log)
        
    return JSONResponse(content={"logs": filtered[-limit:]})


@router.get("/api/endpoints")
def get_endpoints(request: Request):
    """Retrieve FastAPI registered routes mapping name, method, parameters and security context."""
    routes = []
    for r in request.app.routes:
        if hasattr(r, "methods"):
            auth_required = False
            # Check route dependencies for authentication token validation
            if hasattr(r, "dependencies"):
                for dep in r.dependencies:
                    if "get_current_user" in str(dep.dependency):
                        auth_required = True
            
            routes.append({
                "path": r.path,
                "methods": list(r.methods),
                "name": r.name,
                "description": getattr(r, "description", None) or getattr(r, "summary", None) or "",
                "auth_required": auth_required
            })
    return JSONResponse(content={"endpoints": routes})


@router.post("/api/maintenance")
async def post_maintenance(body: dict):
    """Triggers maintenance model, memory and CUDA operations."""
    action = body.get("action")
    target_job_id = body.get("data")
    
    if action == "unload_models":
        from tasks.pipeline import unload_all_models
        unload_all_models()
        return {"status": "success", "message": "All residency model weights unloaded successfully from RAM/VRAM."}
        
    elif action == "clear_cuda_cache":
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            return {"status": "success", "message": f"CUDA VRAM cache cleared. Residency memory: {allocated:.1f} MB"}
        return {"status": "success", "message": "CUDA not available — Cache empty skipped."}
        
    elif action == "clear_temp_files":
        import tempfile
        import shutil
        temp_dir = Path(tempfile.gettempdir()) / "voicesum_runtime"
        files_deleted = 0
        if temp_dir.exists() and temp_dir.is_dir():
            for child in temp_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    files_deleted += 1
                except Exception:
                    pass
        return {"status": "success", "message": f"Cleared voicesum_runtime temp directory. Cleaned {files_deleted} structures."}
        
    elif action == "clear_vector_cache":
        # Clear text embedder local FAISS caches
        if "services.text_embedding_service" in sys.modules:
            from services.text_embedding_service import unload_text_embedder
            unload_text_embedder()
        return {"status": "success", "message": "Vector embedder services caches unloaded."}
        
    elif action == "reload_models":
        # Warmup models
        try:
            from services.ai_provider import warm_up_model
            warm_up_model()
            from services.text_embedding_service import get_text_embedder
            get_text_embedder().load()
            return {"status": "success", "message": "Residency NLP and embedding models warmed up in background."}
        except Exception as e:
            return {"status": "error", "message": f"Reload warmup failed: {e}"}
            
    elif action == "cancel_task" and target_job_id:
        from tasks.pipeline import cancel_task
        cancelled = await cancel_task(target_job_id)
        if cancelled:
            return {"status": "success", "message": f"Cancelled processing task {target_job_id}."}
        return {"status": "error", "message": f"Job task {target_job_id} not found in active registry."}
        
    raise HTTPException(status_code=400, detail="Invalid action parameter.")


@router.get("/api/diagnostics/export")
async def export_diagnostics():
    """Generates diagnostics details JSON export."""
    status = await get_status_payload()
    from main import in_memory_log_handler
    logs = list(in_memory_log_handler.records)
    
    diagnostic_info = {
        "timestamp": time.time(),
        "status": status,
        "recent_logs": logs
    }
    
    # Write to a temp file in runtime/
    export_path = Path("voicesum_diagnostics.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_info, f, indent=2)
        
    return FileResponse(
        path=str(export_path),
        media_type="application/json",
        filename="voicesum_diagnostics.json"
    )


@router.get("/api/diagnostics/logs/download")
def download_logs():
    """Serves raw voicesum.log file."""
    log_file = "voicesum.log"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("--- Log started ---")
            
    return FileResponse(
        path=log_file,
        media_type="text/plain",
        filename="voicesum.log"
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Establishes active streaming socket pushing log chunks and system statistics."""
    await websocket.accept()
    
    from main import in_memory_log_handler
    initial_logs = list(in_memory_log_handler.records)
    
    # 1. Send all buffered startup logs
    await websocket.send_json({
        "type": "initial_logs",
        "logs": initial_logs
    })
    
    last_sent_idx = len(initial_logs)
    
    try:
        while True:
            status = await get_status_payload()
            
            # Extract new logs since last loop pass
            all_records = list(in_memory_log_handler.records)
            new_logs = []
            if len(all_records) > last_sent_idx:
                new_logs = all_records[last_sent_idx:]
                last_sent_idx = len(all_records)
            elif len(all_records) < last_sent_idx:
                last_sent_idx = len(all_records)
                
            await websocket.send_json({
                "type": "update",
                "status": status,
                "new_logs": new_logs
            })
            
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[Dashboard WS] Connection terminated with exception: {e}")
