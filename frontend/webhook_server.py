#!/usr/bin/env python3
"""
Simple FastAPI webhook server to receive battery data from Flower
and serve it to the React frontend via REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import asyncio
from datetime import datetime
import json

app = FastAPI(title="Battery Data Webhook Server", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],  # React dev servers
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory storage for battery data
latest_battery_data: Dict[str, Any] = {}
battery_history: list = []

class BatteryWebhookData(BaseModel):
    type: str
    round: int
    battery_levels: Dict[str, Any]
    timestamp: float

@app.get("/")
async def root():
    return {"message": "Battery Webhook Server Running", "status": "ok"}

@app.post("/api/webhook/battery")
async def receive_battery_webhook(data: BatteryWebhookData):
    """Receive battery data from Flower server"""
    global latest_battery_data, battery_history
    
    try:
        # Store the latest data
        latest_battery_data = {
            "type": data.type,
            "round": data.round,
            "battery_levels": data.battery_levels,
            "timestamp": datetime.now().isoformat(),
            "received_at": datetime.now().timestamp()
        }
        
        # Add to history (keep last 50 entries)
        battery_history.append(latest_battery_data.copy())
        if len(battery_history) > 50:
            battery_history.pop(0)
        
        print(f"🔋 Received battery data for round {data.round}")
        print(f"   Satellites: {len(data.battery_levels)}")
        for sat_id, info in data.battery_levels.items():
            battery_level = info.get('battery', 0)
            status = info.get('status', 'unknown')
            print(f"   {sat_id}: {battery_level:.1f}% ({status})")
        
        return {"status": "success", "message": f"Received battery data for round {data.round}"}
        
    except Exception as e:
        print(f"❌ Error processing battery webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/battery/latest")
async def get_latest_battery_data():
    """Get the latest battery data for frontend"""
    if not latest_battery_data:
        return {"status": "no_data", "message": "No battery data received yet"}
    
    return {
        "status": "success",
        "data": latest_battery_data
    }

@app.get("/api/battery/history")
async def get_battery_history():
    """Get battery data history"""
    return {
        "status": "success", 
        "data": battery_history,
        "count": len(battery_history)
    }

@app.get("/api/battery/status")
async def get_battery_status():
    """Get current battery status summary"""
    if not latest_battery_data:
        return {"status": "no_data", "message": "No battery data available"}
    
    battery_levels = latest_battery_data.get("battery_levels", {})
    
    # Calculate summary statistics
    batteries = [info.get("battery", 0) for info in battery_levels.values()]
    if not batteries:
        return {"status": "no_data", "message": "No battery levels found"}
    
    summary = {
        "round": latest_battery_data.get("round", 0),
        "total_satellites": len(batteries),
        "avg_battery": sum(batteries) / len(batteries),
        "min_battery": min(batteries),
        "max_battery": max(batteries),
        "satellites_online": len([b for b in batteries if b >= 30]),
        "satellites_low": len([b for b in batteries if b < 30]),
        "last_update": latest_battery_data.get("timestamp"),
    }
    
    return {"status": "success", "data": summary}

if __name__ == "__main__":
    print("🚀 Starting Battery Webhook Server on http://localhost:8000")
    print("📡 Ready to receive battery data from Flower...")
    uvicorn.run(app, host="0.0.0.0", port=8000)