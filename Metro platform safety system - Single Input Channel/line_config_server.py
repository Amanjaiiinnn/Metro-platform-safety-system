"""
line_config_server.py
──────────────────────
Flask web control dashboard for real-time video tuning, threshold adjustment,
and virtual line configuration.

Reads and writes config.json. Loads previously saved line coordinates,
thresholds, and image adjustments on startup.
"""

import json
import logging
import os
import platform
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import socket
import cv2
from flask import Flask, Response, jsonify, send_from_directory, request
from werkzeug.serving import run_simple

from config import is_schedule_off, get_current_time, set_virtual_clock, TIME_SOURCE

CONFIG_PATH  = "config.json"
BACKUP_PATH  = "config_backup.json"

_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metro Safety Control Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-dark: #0a0a14;
    --card-bg: rgba(26, 22, 46, 0.65);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-border-hover: rgba(255, 255, 255, 0.18);
    --text-primary: #f8fafc;
    --text-secondary: #a5adc7;
    --text-tertiary: #6b7395;

    --c-purple: #a855f7;
    --c-pink: #ec4899;
    --c-cyan: #22d3ee;
    --c-blue: #3b82f6;
    --c-green: #22c55e;
    --c-yellow: #facc15;
    --c-orange: #fb923c;
    --c-red: #f43f5e;

    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
  ::selection { background: var(--c-pink); color: white; }

  @keyframes floatBlob {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(3%, -4%) scale(1.08); }
    66% { transform: translate(-3%, 3%) scale(0.95); }
  }

  body {
    background: var(--bg-dark);
    color: var(--text-primary);
    min-height: 100vh;
    padding: clamp(14px, 3vw, 28px);
    display: flex;
    flex-direction: column;
    gap: 20px;
    position: relative;
    overflow-x: hidden;
  }

  body::before, body::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.35;
    z-index: 0;
    pointer-events: none;
  }
  body::before {
    width: 620px; height: 620px;
    top: -220px; left: -160px;
    background: radial-gradient(circle, var(--c-purple), transparent 70%);
    animation: floatBlob 18s ease-in-out infinite;
  }
  body::after {
    width: 560px; height: 560px;
    bottom: -220px; right: -140px;
    background: radial-gradient(circle, var(--c-cyan), transparent 70%);
    animation: floatBlob 22s ease-in-out infinite reverse;
  }

  .bg-blob-3 {
    content: '';
    position: fixed;
    width: 480px; height: 480px;
    top: 30%; right: 15%;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.22;
    background: radial-gradient(circle, var(--c-pink), transparent 70%);
    animation: floatBlob 26s ease-in-out infinite;
    z-index: 0;
    pointer-events: none;
  }

  header, .main-grid { position: relative; z-index: 1; }

  /* ---------- Header ---------- */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(20px);
    flex-wrap: wrap;
    gap: 12px;
    position: relative;
    overflow: hidden;
  }

  header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--c-purple), var(--c-pink), var(--c-cyan), var(--c-green), var(--c-yellow), var(--c-purple));
    background-size: 300% 100%;
    animation: rainbowShift 8s linear infinite;
  }

  @keyframes rainbowShift {
    0% { background-position: 0% 0%; }
    100% { background-position: 300% 0%; }
  }

  .logo-title {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .logo-icon {
    width: 42px;
    height: 42px;
    flex-shrink: 0;
    background: linear-gradient(135deg, var(--c-purple), var(--c-pink), var(--c-cyan));
    background-size: 200% 200%;
    animation: gradientMove 5s ease infinite;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 20px;
    box-shadow: 0 4px 24px rgba(168, 85, 247, 0.55), inset 0 1px 0 rgba(255,255,255,0.3);
  }

  @keyframes gradientMove {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }

  h1 {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.3px;
    line-height: 1.3;
    background: linear-gradient(90deg, #ffffff, #d8b4fe);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle { font-size: 12.5px; color: var(--text-secondary); font-weight: 500; margin-top: 1px; }

  .header-right { display: flex; align-items: center; gap: 10px; }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #4ade80;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.15);
  }

  .pulse-dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 10px #22c55e;
    animation: pulse 2s infinite;
    flex-shrink: 0;
  }

  @keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
  }

  .res-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(59, 130, 246, 0.14);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ---------- Layout ---------- */
  .main-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 370px;
    gap: 20px;
    align-items: start;
  }

  @media (max-width: 1100px) { .main-grid { grid-template-columns: 1fr; } }

  .viewport-card, .panel-card {
    position: relative;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    backdrop-filter: blur(20px);
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
  }

  .viewport-card::before, .panel-card::before {
    content: '';
    position: absolute;
    top: 0; left: 18px; right: 18px;
    height: 2px;
    border-radius: 2px;
  }

  .viewport-card::before { background: linear-gradient(90deg, var(--c-cyan), var(--c-blue)); }
  .panel-card.theme-pink::before { background: linear-gradient(90deg, var(--c-pink), var(--c-purple)); }
  .panel-card.theme-green::before { background: linear-gradient(90deg, var(--c-green), var(--c-cyan)); }
  .panel-card.theme-orange::before { background: linear-gradient(90deg, var(--c-orange), var(--c-yellow)); }

  .panel-card:hover, .viewport-card:hover {
    border-color: var(--card-border-hover);
    transform: translateY(-2px);
  }

  .panel-card.theme-pink:hover { box-shadow: 0 12px 32px rgba(236, 72, 153, 0.18); }
  .panel-card.theme-green:hover { box-shadow: 0 12px 32px rgba(34, 197, 94, 0.18); }
  .panel-card.theme-orange:hover { box-shadow: 0 12px 32px rgba(251, 146, 60, 0.18); }
  .viewport-card:hover { box-shadow: 0 12px 32px rgba(34, 211, 238, 0.16); }

  .viewport-head { display: flex; align-items: center; justify-content: space-between; }

  .viewport-title {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .viewport-title svg { color: var(--c-cyan); }

  .stage-wrapper {
    position: relative;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #000;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: 0 16px 40px rgba(0,0,0,0.55), 0 0 0 1px rgba(34, 211, 238, 0.15), inset 0 0 0 1px rgba(255,255,255,0.05);
  }

  #stage { position: relative; width: 640px; height: 480px; max-width: 100%; }
  #feed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: fill; filter: contrast(1.02) saturate(1.05); }
  #canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: transparent; cursor: crosshair; }

  .legend-bar { display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--card-border);
    padding: 5px 12px 5px 9px;
    border-radius: 20px;
  }

  .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 8px currentColor; }
  .drag-hint { text-align: center; font-size: 11.5px; color: var(--text-tertiary); font-weight: 500; }

  /* ---------- Sidebar ---------- */
  .controls-sidebar { display: flex; flex-direction: column; gap: 16px; }

  .panel-title {
    font-size: 12.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .panel-icon {
    width: 28px; height: 28px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .theme-pink .panel-icon { background: linear-gradient(135deg, rgba(236,72,153,0.35), rgba(168,85,247,0.35)); color: #f9a8d4; box-shadow: 0 0 16px rgba(236,72,153,0.3); }
  .theme-green .panel-icon { background: linear-gradient(135deg, rgba(34,197,94,0.35), rgba(34,211,238,0.35)); color: #86efac; box-shadow: 0 0 16px rgba(34,197,94,0.3); }
  .theme-orange .panel-icon { background: linear-gradient(135deg, rgba(251,146,60,0.35), rgba(250,204,21,0.35)); color: #fdba74; box-shadow: 0 0 16px rgba(251,146,60,0.3); }

  .slider-group { display: flex; flex-direction: column; gap: 8px; }

  .slider-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 600;
  }

  .slider-header .label-with-icon { display: flex; align-items: center; gap: 7px; }
  .slider-header .label-with-icon svg { width: 14px; height: 14px; }

  .slider-val {
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    padding: 2px 9px;
    border-radius: 7px;
    min-width: 44px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
  }

  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
    background: #23233a;
  }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--accent, var(--c-purple));
    border: 3px solid #16162a;
    cursor: pointer;
    box-shadow: 0 0 0 3px var(--accent, var(--c-purple)), 0 0 14px var(--accent, var(--c-purple));
    transition: transform 0.15s;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }

  input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--accent, var(--c-purple));
    border: 3px solid #16162a;
    cursor: pointer;
    box-shadow: 0 0 0 3px var(--accent, var(--c-purple)), 0 0 14px var(--accent, var(--c-purple));
  }
  input[type=range]::-moz-range-track { height: 6px; border-radius: 3px; background: #23233a; }

  .divider { height: 1px; background: var(--card-border); margin: 2px 0; }

  .input-number-group {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 13px; color: var(--text-secondary); font-weight: 600;
  }

  .stepper {
    display: flex; align-items: center;
    background: #16162a;
    border: 1px solid rgba(251, 146, 60, 0.3);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  .stepper button {
    all: unset;
    width: 30px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    color: var(--c-orange);
    cursor: pointer;
    font-size: 16px;
    font-weight: 700;
    transition: background 0.15s;
  }
  .stepper button:hover { background: rgba(251, 146, 60, 0.15); }

  input[type=number] {
    width: 58px;
    background: transparent;
    border: none;
    border-left: 1px solid rgba(251, 146, 60, 0.25);
    border-right: 1px solid rgba(251, 146, 60, 0.25);
    color: var(--text-primary);
    padding: 6px 4px;
    text-align: center;
    font-size: 13.5px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    -moz-appearance: textfield;
  }
  input[type=number]::-webkit-outer-spin-button, input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  input[type=number]:focus { outline: none; }

  button.btn-primary {
    width: 100%;
    padding: 14px 16px;
    border: none;
    border-radius: var(--radius-sm);
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    background: linear-gradient(135deg, var(--c-purple), var(--c-pink));
    background-size: 200% 200%;
    color: white;
    box-shadow: 0 4px 20px rgba(236, 72, 153, 0.4);
  }

  button.btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(236, 72, 153, 0.55);
    background-position: 100% 50%;
  }
  button.btn-primary:active:not(:disabled) { transform: translateY(0); }
  button.btn-primary:disabled { opacity: 0.75; cursor: default; }

  button.btn-primary.saved {
    background: linear-gradient(135deg, var(--c-green), var(--c-cyan));
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.45);
  }

  #status-msg {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    font-size: 12px; color: var(--text-tertiary);
    text-align: center; min-height: 18px; font-weight: 600;
    transition: color 0.2s;
  }
  #status-msg.ok { color: var(--c-green); }

  @media (max-width: 480px) {
    header { flex-direction: column; align-items: flex-start; }
    .header-right { width: 100%; justify-content: space-between; }
  }
</style>
</head>
<body>
  <div class="bg-blob-3"></div>

  <header>
    <div class="logo-title">
      <div class="logo-icon">M</div>
      <div>
        <h1>Metro Safety Tuning &amp; Control Center</h1>
        <div class="subtitle">Real-time image tuning &amp; virtual line configuration</div>
      </div>
    </div>
    <div class="header-right">
      <div class="res-badge" id="resBadge">640 × 480</div>
      <div class="status-badge">
        <div class="pulse-dot"></div>
        Live Pipeline Connected
      </div>
    </div>
  </header>

  <div class="main-grid">
    <!-- Left: Viewport & Interactive Handles -->
    <div class="viewport-card">
      <div class="viewport-head">
        <div class="viewport-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
          Live Camera Feed
        </div>
      </div>
      <div class="stage-wrapper">
        <div id="stage">
          <img id="feed" src="/video_feed" alt="Live Feed">
          <canvas id="canvas" width="640" height="480"></canvas>
        </div>
      </div>

      <div class="legend-bar">
        <div class="legend-item"><div class="dot" style="background:#ff8c00; color:#ff8c00;"></div>P1 Endpoint</div>
        <div class="legend-item"><div class="dot" style="background:#00d4ff; color:#00d4ff;"></div>P2 Endpoint</div>
        <div class="legend-item"><div class="dot" style="background:#ff3355; color:#ff3355;"></div>Alert Trigger Side</div>
      </div>
      <div class="drag-hint">Drag any handle on the feed to reposition it — changes save automatically</div>
    </div>

    <!-- Right: Sidebar Controls -->
    <div class="controls-sidebar">

      <!-- Image Adjustments Panel -->
      <div class="panel-card theme-pink">
        <div class="panel-title">
          <div class="panel-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
          </div>
          Image Adjustments
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#facc15;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/></svg>
              Brightness
            </span>
            <span class="slider-val" id="val-brightness" style="background:rgba(250,204,21,0.15); color:#facc15;">0</span>
          </div>
          <input type="range" id="slider-brightness" min="-100" max="100" value="0" style="--accent:#facc15;">
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#fb923c;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20z" fill="currentColor" stroke="none"/></svg>
              Contrast
            </span>
            <span class="slider-val" id="val-contrast" style="background:rgba(251,146,60,0.15); color:#fb923c;">0</span>
          </div>
          <input type="range" id="slider-contrast" min="-100" max="100" value="0" style="--accent:#fb923c;">
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#f472b6;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>
              Exposure
            </span>
            <span class="slider-val" id="val-exposure" style="background:rgba(244,114,182,0.15); color:#f472b6;">0</span>
          </div>
          <input type="range" id="slider-exposure" min="-100" max="100" value="0" style="--accent:#f472b6;">
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#c084fc;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2s7 7.5 7 12.5a7 7 0 1 1-14 0C5 9.5 12 2 12 2z"/></svg>
              Saturation
            </span>
            <span class="slider-val" id="val-saturation" style="background:rgba(192,132,252,0.15); color:#c084fc;">0</span>
          </div>
          <input type="range" id="slider-saturation" min="-100" max="100" value="0" style="--accent:#c084fc;">
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#22d3ee;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 21V10M4 6V3M12 21v-7M12 10V3M20 21v-4M20 13V3"/><path d="M1 10h6M9 6h6M17 13h6"/></svg>
              Gamma
            </span>
            <span class="slider-val" id="val-gamma" style="background:rgba(34,211,238,0.15); color:#22d3ee;">0</span>
          </div>
          <input type="range" id="slider-gamma" min="-100" max="100" value="0" style="--accent:#22d3ee;">
        </div>
      </div>

      <!-- Detection Thresholds Panel -->
      <div class="panel-card theme-green">
        <div class="panel-title">
          <div class="panel-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>
          </div>
          Detection Thresholds
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#4ade80;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="7" r="4"/><path d="M4 21v-1a8 8 0 0 1 16 0v1"/></svg>
              Person Confidence
            </span>
            <span class="slider-val" id="val-person-conf" style="background:rgba(74,222,128,0.15); color:#4ade80;">50%</span>
          </div>
          <input type="range" id="slider-person-conf" min="5" max="100" value="50" style="--accent:#4ade80;">
        </div>

        <div class="slider-group">
          <div class="slider-header">
            <span class="label-with-icon" style="color:#38bdf8;">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="12" rx="2"/><path d="M4 12h16M8 20l-2-4M16 20l2-4"/><circle cx="8" cy="8" r="1" fill="currentColor"/><circle cx="16" cy="8" r="1" fill="currentColor"/></svg>
              Train Confidence
            </span>
            <span class="slider-val" id="val-train-conf" style="background:rgba(56,189,248,0.15); color:#38bdf8;">45%</span>
          </div>
          <input type="range" id="slider-train-conf" min="5" max="100" value="45" style="--accent:#38bdf8;">
        </div>
      </div>

      <!-- Buffer Settings & Save -->
      <div class="panel-card theme-orange">
        <div class="panel-title">
          <div class="panel-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3 8 4-16 3 8h4"/></svg>
          </div>
          Buffer &amp; Save
        </div>

        <div class="input-number-group">
          <label for="threshold">Line buffer zone (px)</label>
          <div class="stepper">
            <button type="button" id="thresholdDown" aria-label="Decrease">−</button>
            <input type="number" id="threshold" value="0" min="0" step="1">
            <button type="button" id="thresholdUp" aria-label="Increase">+</button>
          </div>
        </div>

        <div class="divider"></div>

        <button class="btn-primary" id="saveBtn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/></svg>
          <span id="saveBtnLabel">Save Config</span>
        </button>
        <div id="status-msg"></div>
      </div>

    </div>
  </div>

<script>
let p1 = [100, 240], p2 = [540, 240], alertPt = [320, 400], frameW = 640, frameH = 480;
let dragging = null;
let hovered = null;
let saveTimer = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  for (let x = 0; x < canvas.width; x += 64) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
  for (let y = 0; y < canvas.height; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }

  ctx.strokeStyle = '#ff8c00';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#ff8c00';
  ctx.shadowBlur = 8;
  ctx.setLineDash([8, 4]);
  ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;

  drawHandle(p1, '#ff8c00', 'P1', hovered === 'p1' || dragging === 'p1');
  drawHandle(p2, '#00d4ff', 'P2', hovered === 'p2' || dragging === 'p2');
  drawHandle(alertPt, '#ff3355', 'ALERT', hovered === 'alert' || dragging === 'alert');
}

function drawHandle(pt, color, label, active) {
  if (active) {
    ctx.beginPath(); ctx.arc(pt[0], pt[1], 18, 0, 2 * Math.PI);
    ctx.fillStyle = color + '33';
    ctx.fill();
  }
  ctx.shadowColor = color;
  ctx.shadowBlur = active ? 16 : 8;
  ctx.beginPath(); ctx.arc(pt[0], pt[1], active ? 11 : 10, 0, 2 * Math.PI);
  ctx.fillStyle = color; ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5; ctx.stroke();

  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  const ty = pt[1] - 18;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(10,10,20,0.65)';
  ctx.fillRect(pt[0] - tw / 2 - 5, ty - 12, tw + 10, 16);
  ctx.fillStyle = color;
  ctx.fillText(label, pt[0], ty);
  ctx.textAlign = 'left';
}

function hitTest(x, y) {
  for (const [name, pt] of [['p1', p1], ['p2', p2], ['alert', alertPt]]) {
    if (Math.hypot(x - pt[0], y - pt[1]) <= 18) return name;
  }
  return null;
}

function canvasCoords(e) {
  const r = canvas.getBoundingClientRect();
  const scaleX = canvas.width / r.width;
  const scaleY = canvas.height / r.height;
  return { x: (e.clientX - r.left) * scaleX, y: (e.clientY - r.top) * scaleY };
}

canvas.addEventListener('mousedown', e => {
  const { x, y } = canvasCoords(e);
  dragging = hitTest(x, y);
  canvas.style.cursor = dragging ? 'grabbing' : 'crosshair';
});

canvas.addEventListener('mousemove', e => {
  const { x, y } = canvasCoords(e);
  if (!dragging) {
    hovered = hitTest(x, y);
    canvas.style.cursor = hovered ? 'grab' : 'crosshair';
    draw();
    return;
  }
  const cx = Math.max(0, Math.min(Math.round(x), canvas.width - 1));
  const cy = Math.max(0, Math.min(Math.round(y), canvas.height - 1));
  if (dragging === 'p1') p1 = [cx, cy];
  else if (dragging === 'p2') p2 = [cx, cy];
  else if (dragging === 'alert') alertPt = [cx, cy];
  draw();
  scheduleAutoSave();
});

canvas.addEventListener('mouseleave', () => { hovered = null; draw(); });

window.addEventListener('mouseup', () => {
  dragging = null;
  canvas.style.cursor = hovered ? 'grab' : 'crosshair';
});

async function loadConfig() {
  try {
    const res = await fetch('/get_lines');
    const data = await res.json();
    p1 = data.p1 || p1;
    p2 = data.p2 || p2;
    alertPt = data.alert_side_pt || alertPt;
    frameW = data.frame_width || 640;
    frameH = data.frame_height || 480;

    canvas.width = frameW; canvas.height = frameH;
    const stage = document.getElementById('stage');
    stage.style.width = frameW + 'px';
    stage.style.height = frameH + 'px';
    document.getElementById('resBadge').textContent = frameW + ' × ' + frameH;

    document.getElementById('threshold').value = data.threshold_pixels || 0;

    setSliderVal('brightness', data.brightness || 0);
    setSliderVal('contrast', data.contrast || 0);
    setSliderVal('exposure', data.exposure || 0);
    setSliderVal('saturation', data.saturation || 0);
    setSliderVal('gamma', data.gamma || 0);

    const personPct = Math.round((data.person_conf_th !== undefined ? data.person_conf_th : 0.50) * 100);
    const trainPct = Math.round((data.train_conf_th !== undefined ? data.train_conf_th : 0.45) * 100);

    setSliderVal('person-conf', personPct, '%');
    setSliderVal('train-conf', trainPct, '%');

  } catch (e) { console.error('Failed to load config', e); }
  draw();
}

function updateSliderFill(input) {
  const min = parseFloat(input.min) || 0;
  const max = parseFloat(input.max) || 100;
  const val = parseFloat(input.value) || 0;
  const pct = ((val - min) / (max - min)) * 100;
  const accent = getComputedStyle(input).getPropertyValue('--accent').trim() || '#a855f7';
  input.style.background = `linear-gradient(to right, ${accent} 0%, ${accent} ${pct}%, #23233a ${pct}%, #23233a 100%)`;
}

function setSliderVal(id, val, suffix='') {
  const elem = document.getElementById(`slider-${id}`);
  const disp = document.getElementById(`val-${id}`);
  if (elem) { elem.value = val; updateSliderFill(elem); }
  if (disp) disp.textContent = val + suffix;
}

function bindSlider(id, suffix='') {
  const slider = document.getElementById(`slider-${id}`);
  const label = document.getElementById(`val-${id}`);
  if (!slider) return;
  slider.addEventListener('input', () => {
    label.textContent = slider.value + suffix;
    updateSliderFill(slider);
    scheduleAutoSave();
  });
}

['brightness', 'contrast', 'exposure', 'saturation', 'gamma'].forEach(id => bindSlider(id));
bindSlider('person-conf', '%');
bindSlider('train-conf', '%');

const thresholdInput = document.getElementById('threshold');
thresholdInput.addEventListener('input', scheduleAutoSave);
document.getElementById('thresholdUp').addEventListener('click', () => {
  thresholdInput.value = (parseInt(thresholdInput.value) || 0) + 1;
  scheduleAutoSave();
});
document.getElementById('thresholdDown').addEventListener('click', () => {
  thresholdInput.value = Math.max(0, (parseInt(thresholdInput.value) || 0) - 1);
  scheduleAutoSave();
});

function scheduleAutoSave() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(saveConfig, 40);
}

async function saveConfig() {
  const body = {
    p1, p2, alert_side_pt: alertPt,
    threshold_pixels: parseFloat(document.getElementById('threshold').value) || 0,
    frame_width: frameW, frame_height: frameH,
    brightness: parseInt(document.getElementById('slider-brightness').value) || 0,
    contrast: parseInt(document.getElementById('slider-contrast').value) || 0,
    exposure: parseInt(document.getElementById('slider-exposure').value) || 0,
    saturation: parseInt(document.getElementById('slider-saturation').value) || 0,
    gamma: parseInt(document.getElementById('slider-gamma').value) || 0,
    person_conf_th: Math.max(0.05, (parseInt(document.getElementById('slider-person-conf').value) || 50) / 100.0),
    train_conf_th:  Math.max(0.05, (parseInt(document.getElementById('slider-train-conf').value)  || 45) / 100.0),
  };

  const statusElem = document.getElementById('status-msg');

  try {
    const res = await fetch('/update_lines', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.status === 'saved') {
      statusElem.classList.add('ok');
      statusElem.textContent = 'Saved at ' + new Date().toLocaleTimeString();
    }
  } catch (e) {
    statusElem.classList.remove('ok');
    statusElem.textContent = 'Save failed — check connection';
    console.error('Save error', e);
  }
}

const saveBtn = document.getElementById('saveBtn');
const saveBtnLabel = document.getElementById('saveBtnLabel');

saveBtn.addEventListener('click', async () => {
  saveBtn.disabled = true;
  const original = saveBtnLabel.textContent;
  saveBtnLabel.textContent = 'Saving…';
  await saveConfig();
  saveBtn.classList.add('saved');
  saveBtnLabel.textContent = 'Saved ✓';
  setTimeout(() => {
    saveBtn.classList.remove('saved');
    saveBtnLabel.textContent = original;
    saveBtn.disabled = false;
  }, 1200);
});

loadConfig();
</script>
</body>
</html>
"""

_DEFAULT_CONFIG = {
    "p1": [0, 240],
    "p2": [639, 240],
    "alert_side_pt": [320, 400],
    "threshold_pixels": 0.0,
    "frame_width": 640,
    "frame_height": 480,
    # Image adjustments (all -100..100; exposure in EV*50, gamma offset)
    "brightness": 0,
    "contrast": 0,
    "saturation": 0,
    "exposure": 0,
    "gamma": 0,
    # Detection thresholds
    "person_conf_th": 0.50,
    "train_conf_th": 0.45,
    # Alert / audio
    "alert_cooldown_seconds": 30,
    # Spoken-alert playback volume as a percentage (0-150). 100 = each
    # clip's own recorded level, unchanged.
    "alert_volume": 100,
    # Which spoken-alert language(s) are on. Any subset of "english",
    # "hindi", "regional" -- checkboxes on the dashboard, not a single
    # dropdown value anymore. Empty list = alert still prints to console,
    # just no audio. See config.py's VOICES_DIR comment for the WAV
    # layout each language folder needs.
    "tts_languages": ["english"],
    "tts_rate": 160,
    # Train schedule (no-detection window, HH:MM strings)
    "train_off_start": "23:30",
    "train_off_end": "05:00",
}

_VALID_TTS_LANGUAGES = ("english", "hindi", "regional")


def _sanitize_languages(value) -> list:
    """Keep only known language codes, de-duped, order preserved.

    Falls back to the default (English only) if the incoming value isn't
    even a list -- e.g. an old config.json still has the pre-checkbox
    "tts_lang" string format. An explicit EMPTY list is left as-is on
    purpose: that's the "all languages unchecked -> silent alerts" state,
    not a missing/invalid value.
    """
    if not isinstance(value, list):
        return list(_DEFAULT_CONFIG["tts_languages"])
    out = []
    for item in value:
        code = str(item).strip().lower()
        if code in _VALID_TTS_LANGUAGES and code not in out:
            out.append(code)
    return out


def _sanitize_volume(value) -> int:
    """Clamp the incoming alert-volume percentage to 0-200.

    Same defensive spirit as _sanitize_languages above: request JSON is
    untrusted input (a stale UI, a bad manual /update_lines call, etc.),
    so this is the one place that guarantees whatever ends up in
    config.json -- and from there, AlertEngine.volume -- is always a sane
    number. AlertEngine._apply_volume() clamps again as a second line of
    defense, but doing it here too keeps config.json itself honest.
    """
    try:
        pct = int(round(float(value)))
    except (TypeError, ValueError):
        return _DEFAULT_CONFIG["alert_volume"]
    return max(0, min(200, pct))


app = Flask(__name__)


_active_config = dict(_DEFAULT_CONFIG)

def _init_active_config():
    global _active_config
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                _active_config = json.load(f)
        else:
            _active_config = dict(_DEFAULT_CONFIG)
    except Exception:
        _active_config = dict(_DEFAULT_CONFIG)

_init_active_config()


def _read_config() -> dict:
    return dict(_active_config)


def _create_backup() -> None:
    """Copy the current config.json to config_backup.json.
    Called after every successful save so the backup always reflects
    the last intentionally saved state.
    """
    try:
        if os.path.exists(CONFIG_PATH):
            import shutil
            shutil.copy2(CONFIG_PATH, BACKUP_PATH)
    except Exception as exc:
        logging.warning("[Config] Could not create backup: %s", exc)


def _restore_from_backup() -> bool:
    """Overwrite config.json with config_backup.json and reload active config. Returns True on success."""
    try:
        if not os.path.exists(BACKUP_PATH):
            return False
        import shutil
        tmp = CONFIG_PATH + ".tmp"
        shutil.copy2(BACKUP_PATH, tmp)
        os.replace(tmp, CONFIG_PATH)
        _init_active_config()
        return True
    except Exception as exc:
        logging.warning("[Config] Restore from backup failed: %s", exc)
        return False


_resolution_lock = threading.Lock()
_current_resolution = {"width": None, "height": None}


def set_frame_resolution(width: int, height: int) -> None:
    with _resolution_lock:
        _current_resolution["width"] = int(width)
        _current_resolution["height"] = int(height)


def _get_current_resolution():
    with _resolution_lock:
        return _current_resolution["width"], _current_resolution["height"]


# Directory that contains index.html (same folder as this server file)
_UI_DIR = str(Path(__file__).parent)


def _no_cache(response):
    """Force the browser to always re-fetch instead of reusing a stale
    cached copy of the dashboard. Without this, updating index.html on
    disk (which happens often while iterating on the console) doesn't
    reliably show up in the browser even after a normal refresh -- the
    old JS just keeps running silently, which is exactly what made the
    topbar clock look like it was still using the browser's own time
    instead of the fix that was already on disk.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return _no_cache(send_from_directory(_UI_DIR, "index.html"))


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any static file (css, js, images) from the UI directory."""
    return _no_cache(send_from_directory(_UI_DIR, filename))


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/get_lines")
def get_lines():
    data = _read_config()
    width, height = _get_current_resolution()

    if width and height:
        old_w = data.get("frame_width") or width
        old_h = data.get("frame_height") or height
        if old_w and old_h and (old_w, old_h) != (width, height):
            sx = width / old_w
            sy = height / old_h
            data["p1"] = [data["p1"][0] * sx, data["p1"][1] * sy]
            data["p2"] = [data["p2"][0] * sx, data["p2"][1] * sy]
            data["alert_side_pt"] = [
                data["alert_side_pt"][0] * sx,
                data["alert_side_pt"][1] * sy,
            ]
            data["threshold_pixels"] = data.get("threshold_pixels", 0.0) * ((sx + sy) / 2.0)
        data["frame_width"] = width
        data["frame_height"] = height

    with _health_lock:
        data["camera_status"] = _camera_health["status"]
        data["camera_status_msg"] = _camera_health["message"]

    return jsonify(data)


@app.route("/camera_status")
def camera_status():
    with _health_lock:
        payload = dict(_camera_health)
    cfg = _read_config()
    payload["schedule_active"] = not is_schedule_off(
        cfg.get("train_off_start", ""), cfg.get("train_off_end", "")
    )
    now = get_current_time()
    payload["device_time"] = now.strftime("%H:%M:%S")
    # Same explicit-UTC convention as /set_device_time's encode and
    # config.py's utcfromtimestamp() decode -- see the comment there. A
    # bare now.timestamp() here was the other half of the topbar-clock bug:
    # it re-encoded through the board's local OS timezone, and the browser
    # then decoded that epoch through ITS OWN local timezone (a real,
    # legitimate Date conversion on the JS side) -- two different-timezone
    # interpretations of the same "naive digits" value compounded into a
    # double shift (e.g. +5:30 twice) instead of cancelling out.
    payload["device_time_epoch"] = now.replace(tzinfo=timezone.utc).timestamp()
    return jsonify(payload)


_DEVICE_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


@app.route("/set_device_time", methods=["POST"])
def set_device_time():
    """Set 'what time it is' for the whole app, from the caller's clock.

    The i.MX93 board has no internet access (no NTP) and no
    battery-backed RTC. The dashboard is normally opened from a
    phone/laptop that DOES have a correct clock, so this lets the
    operator push that correct time down -- either automatically ("Sync
    device clock" ON in the Scheduling tab) or from a manually-typed
    time (sync OFF).

    Expects JSON: {"datetime": "YYYY-MM-DD HH:MM:SS"}, local wall-clock,
    no timezone conversion attempted (the board has no reliable timezone
    info to convert against either).

    Behavior depends on config.TIME_SOURCE:
      "virtual" -- anchors the in-app virtual clock (config.py) to this
                   value. Never touches the OS clock at all -- this is
                   the path to use while `date -s` isn't reliably
                   sticking on this board (see TIME_SOURCE's comment in
                   config.py for why).
      "system"  -- runs `date -s` as before, requires root (confirmed
                   available here) and an OS clock that actually holds
                   the value once set.
    """
    data = request.json or {}
    dt_str = str(data.get("datetime", "")).strip()

    if not _DEVICE_TIME_RE.match(dt_str):
        return jsonify({
            "status": "error",
            "message": "Expected 'datetime' as 'YYYY-MM-DD HH:MM:SS'",
        }), 400

    if TIME_SOURCE == "virtual":
        try:
            # .replace(tzinfo=timezone.utc) before .timestamp(), NOT a bare
            # .timestamp() on the naive datetime: dt_str's digits are the
            # operator's own local wall-clock time (from the browser's
            # Date object -- see index.html's buildDeviceDatetimeString()),
            # with no timezone attached. A bare .timestamp() call on a naive
            # datetime interprets it through the BOARD's OS timezone, not
            # the operator's -- on this board that silently mismatched IST
            # (browser) vs UTC (board OS), shifting the stored epoch by
            # 5:30 versus the true instant. Anchoring it to UTC explicitly
            # here (and decoding the same way in config.py's
            # get_current_time()) makes epoch_at_sync a fixed,
            # OS-timezone-independent encoding of "these wall-clock digits"
            # -- it never touches the board's local TZ setting at all.
            epoch = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            set_virtual_clock(epoch)
            return jsonify({"status": "set", "datetime": dt_str, "mode": "virtual"})
        except Exception as exc:
            logging.warning("[Config] Failed to set virtual clock: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 500

    if platform.system() != "Linux":
        # PC/dev/Windows -- nothing to set, but don't fail the request so
        # the dashboard's save-flash still behaves normally during dev.
        return jsonify({"status": "skipped", "message": f"Not Linux ({platform.system()})"})

    try:
        result = subprocess.run(
            ["date", "-s", dt_str],
            check=True,
            capture_output=True,
            text=True,
        )
        # Logged even on success -- previously only failures were logged,
        # which made it impossible to tell from the terminal whether a
        # sync click actually ran `date -s` at all versus silently being
        # reverted by something else afterward.
        logging.info("[Config] date -s '%s' -> rc=%s stdout=%r stderr=%r",
                      dt_str, result.returncode, result.stdout.strip(), result.stderr.strip())
        return jsonify({"status": "set", "datetime": dt_str, "mode": "system"})
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or str(exc)).strip()
        logging.warning("[Config] Failed to set device time: %s", err)
        return jsonify({"status": "error", "message": err}), 500
    except Exception as exc:
        logging.warning("[Config] Failed to set device time: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/restore_backup", methods=["POST"])
def restore_backup():
    """Revert config.json to the last startup backup. Called by UI after 30s of unsaved changes."""
    ok = _restore_from_backup()
    if ok:
        return jsonify({"status": "restored"})
    return jsonify({"status": "no_backup"}), 404


@app.route("/preview_lines", methods=["POST"])
def preview_lines():
    """Update active configuration IN MEMORY only for real-time live preview.
    Does NOT write to config.json or config_backup.json on disk.
    """
    data = request.json or {}
    width, height = _get_current_resolution()

    def _get(key, default):
        return data.get(key, _active_config.get(key, default))

    _active_config.update({
        "p1":             _get("p1",             _DEFAULT_CONFIG["p1"]),
        "p2":             _get("p2",             _DEFAULT_CONFIG["p2"]),
        "alert_side_pt": _get("alert_side_pt",  _DEFAULT_CONFIG["alert_side_pt"]),
        "threshold_pixels": float(_get("threshold_pixels", 0.0)),
        "frame_width":  width  or int(_get("frame_width",  _DEFAULT_CONFIG["frame_width"])),
        "frame_height": height or int(_get("frame_height", _DEFAULT_CONFIG["frame_height"])),
        "brightness": int(_get("brightness", 0)),
        "contrast":   int(_get("contrast",   0)),
        "exposure":   int(_get("exposure",   0)),
        "saturation": int(_get("saturation", 0)),
        "gamma":      int(_get("gamma",      0)),
        "person_conf_th": max(0.05, float(_get("person_conf_th", 0.50))),
        "train_conf_th":  max(0.05, float(_get("train_conf_th",  0.45))),
        "alert_cooldown_seconds": int(_get("alert_cooldown_seconds", 30)),
        "alert_volume": _sanitize_volume(_get("alert_volume", _DEFAULT_CONFIG["alert_volume"])),
        "tts_languages": _sanitize_languages(_get("tts_languages", _DEFAULT_CONFIG["tts_languages"])),
        "tts_rate":  int(_get("tts_rate", 160)),
        "train_off_start": str(_get("train_off_start", "23:30")),
        "train_off_end":   str(_get("train_off_end",   "05:00")),
    })

    return jsonify({"status": "preview_updated"})


@app.route("/update_lines", methods=["POST"])
def update_lines():
    global _active_config
    data = request.json or {}
    existing = _active_config
    width, height = _get_current_resolution()
    save_both = data.get("save_both", True)

    def _get(key, default):
        return data.get(key, existing.get(key, default))

    config = {
        # Line geometry (stored in frame-space coordinates)
        "p1":             _get("p1",             _DEFAULT_CONFIG["p1"]),
        "p2":             _get("p2",             _DEFAULT_CONFIG["p2"]),
        "alert_side_pt": _get("alert_side_pt",  _DEFAULT_CONFIG["alert_side_pt"]),
        "threshold_pixels": float(_get("threshold_pixels", 0.0)),
        "frame_width":  width  or int(_get("frame_width",  _DEFAULT_CONFIG["frame_width"])),
        "frame_height": height or int(_get("frame_height", _DEFAULT_CONFIG["frame_height"])),
        # Image adjustments (all integers, -100..100)
        "brightness": int(_get("brightness", 0)),
        "contrast":   int(_get("contrast",   0)),
        "exposure":   int(_get("exposure",   0)),
        "saturation": int(_get("saturation", 0)),
        "gamma":      int(_get("gamma",      0)),
        # Detection thresholds
        "person_conf_th": max(0.05, float(_get("person_conf_th", 0.50))),
        "train_conf_th":  max(0.05, float(_get("train_conf_th",  0.45))),
        # Alerts / audio
        "alert_cooldown_seconds": int(_get("alert_cooldown_seconds", 30)),
        "alert_volume": _sanitize_volume(_get("alert_volume", _DEFAULT_CONFIG["alert_volume"])),
        "tts_languages": _sanitize_languages(_get("tts_languages", _DEFAULT_CONFIG["tts_languages"])),
        "tts_rate":  int(_get("tts_rate", 160)),
        # Train schedule
        "train_off_start": str(_get("train_off_start", "23:30")),
        "train_off_end":   str(_get("train_off_end",   "05:00")),
    }

    _active_config = config

    # Atomic write to config.json — real-time preview on disk for video pipeline
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, CONFIG_PATH)

    # Only mirror to config_backup.json when explicit Save is requested
    if save_both:
        _create_backup()

    return jsonify({"status": "saved"})


frame_queue = queue.Queue(maxsize=2)
_health_lock = threading.Lock()
_camera_health = {"status": "live", "message": "Live", "last_update": 0.0}


def set_camera_health(status_code: str, message: str) -> None:
    with _health_lock:
        _camera_health["status"] = status_code
        _camera_health["message"] = message
        _camera_health["last_update"] = time.time()


def check_camera_health(frame) -> tuple[str, str]:
    if frame is None or frame.size == 0:
        return "black", "Black Frame"
    try:
        small = cv2.resize(frame, (320, 240))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mean_val = float(cv2.mean(gray)[0])
        std_dev  = float(cv2.meanStdDev(gray)[1][0][0])
        lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 1. Full black or dark feed
        if mean_val < 15.0:
            return "black", "Black Frame"

        # 2. Hand/finger covering lens, solid obstruction, or heavy out-of-focus blur
        if lap_var < 20.0 or std_dev < 8.0:
            return "blur", "Blur / Obstructed"

        return "live", "Live"
    except Exception:
        return "black", "Black Frame"


def update_frame(frame) -> None:
    """Push a frame to the Flask live-preview MJPEG stream.

    Camera health (black/blur/obstructed) is intentionally NOT checked
    here anymore. It used to be checked a second time on this frame, but
    by the time this is called `frame` already has boxes, the FPS
    readout, and the yellow line drawn on it -- running the blur/black
    detector on annotated pixels instead of the raw camera frame gives
    a wrong reading (the overlay can mask a genuinely blurred camera, or
    trip a false "blur" status), and it silently clobbered the correct
    result that video_widget.py already computed on the raw frame this
    same iteration via check_camera_health()/set_camera_health(). That
    single raw-frame call is the only camera-health check that should
    happen per frame.
    """
    if not frame_queue.full():
        frame_queue.put(frame)


def _generate_mjpeg():
    while True:
        frame = frame_queue.get()

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def start_server(host: str = "0.0.0.0", port: int = 5050) -> threading.Thread:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Bootstrap: if no backup exists yet (first ever run), seed it from config.json
    # so the model has a valid file to read from before the user first clicks Save.
    if not os.path.exists(BACKUP_PATH) and os.path.exists(CONFIG_PATH):
        _create_backup()

    thread = threading.Thread(
        target=lambda: run_simple(host, port, app, threaded=True, use_reloader=False),
        daemon=True,
    )
    thread.start()
    ip = get_local_ip()
    print("\n***************************************")
    if ip:
        print(f"Dashboard running on: http://{ip}:{port}")
    else:
        print("*Network access : Not connected\n")
    print("***************************************\n")
    return thread


if __name__ == "__main__":
    start_server()
    print("[LineConfigServer] Running standalone on http://0.0.0.0:5050")
    while True:
        time.sleep(1)