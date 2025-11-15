# Federated Learning Satellite System with 3D Visualization

This project demonstrates **battery-aware federated learning** on a satellite constellation with real-time 3D visualization using React and Three.js.

## 🚀 Features

- **10 Virtual Satellites** in 3D Earth orbit with realistic orbital mechanics
- **Battery-Aware Federated Learning** using Flower framework 
- **Real-time Battery Simulation** with solar charging and day/night cycles
- **Webhook-based Communication** between FL server and React frontend
- **3D Satellite Visualization** with battery indicator dots and federated learning status
- **Terminal-style Battery Console** showing satellite status
- **Live Federated Learning Panel** displaying which satellites participate in training

## 🛰️ Architecture

```
┌─────────────────┐    webhook    ┌─────────────────┐    API     ┌─────────────────┐
│  Flower Server  │─────────────>│ FastAPI Webhook │<──────────│  React Frontend │
│  (Fed Learning) │               │    Server       │           │ (3D Visualization)│
└─────────────────┘               └─────────────────┘           └─────────────────┘
```

## 🔧 Setup Instructions

### 1. Backend Setup (Flower + FastAPI)

```bash
# Install dependencies 
cd /path/to/project
uv sync

# Start the webhook server (Terminal 1)
cd frontend
python webhook_server.py

# Start federated learning server (Terminal 2) 
flwr run --run-config fraction-train=0.5
```

### 2. Frontend Setup (React + Three.js)

```bash
# Install Node dependencies
cd frontend
npm install

# Start development server (Terminal 3)
npm run dev
```

### 3. Access the Application

- **React Frontend**: http://localhost:8080
- **Webhook API**: http://localhost:8000
- **Webhook Status**: http://localhost:8000/api/battery/latest

## 🎮 How to Use

1. **Start all services** following setup instructions
2. **Open React frontend** at http://localhost:8080
3. **Click "Show Battery View"** to see real-time battery console and FL status
4. **Watch the 3D satellites** with color-coded battery indicator dots:
   - 🟢 Green: >70% battery
   - 🟡 Yellow: 50-69% battery  
   - 🟠 Orange: 30-49% battery
   - 🔴 Red: 15-29% battery
   - 🔴 Dark Red: <15% battery

5. **Observe federated learning**:
   - Green badges: Selected for training this round
   - Gray badges: Available but not selected
   - Red badges: Low battery, cannot train

## ⚙️ Configuration

### Battery Parameters (`pyproject.toml`)
```toml
# Battery simulation config
battery-enabled = true
initial-battery = 80.0        # Starting battery level
charge-rate = 6.0            # Solar charging rate per round
train-cost = 12.0            # Battery cost for training  
comm-cost = 4.0              # Battery cost for communication
min-battery-threshold = 30.0  # Minimum battery to participate
day-night-cycle = true        # Enable day/night charging simulation
orbit-period = 6             # Orbital period in rounds
```

### Federated Learning Parameters
```toml
num-server-rounds = 3         # Total FL rounds
fraction-train = 0.5          # Fraction of available satellites selected per round
local-epochs = 1              # Local training epochs
lr = 0.001                    # Learning rate
```

## 📊 Real-time Displays

### Battery Console View
- Shows all 10 satellites with battery levels, training status, and day/night indicator
- Updates every 500ms with real battery simulation data
- Color-coded status indicators (✓=Ready, ✗=Low battery)

### Federated Learning Status Panel  
- Displays current FL round number
- Shows which satellites are selected for training
- Updates in real-time via webhook when FL rounds execute

### 3D Satellite Visualization
- Realistic polar and equatorial orbits
- Blinking battery indicator dots on each satellite
- Click satellites to view detailed telemetry
- Smooth orbital animations with Three.js

## 🔄 Data Flow

1. **Flower server** runs federated learning with battery constraints
2. **WebhookBatteryAwareFedAvg** strategy sends battery/FL data to webhook server
3. **FastAPI webhook server** receives and stores the data
4. **React frontend** polls webhook server every 200ms for updates
5. **3D visualization** updates satellite colors and FL status in real-time

## 🛠️ Technical Stack

- **Backend**: Flower FL framework, FastAPI, Pydantic, uvicorn
- **Frontend**: React 18, TypeScript, Vite, Three.js, React Three Fiber
- **3D Graphics**: @react-three/fiber, @react-three/drei, Three.js
- **UI Components**: Radix UI, Tailwind CSS, Lucide icons
- **Communication**: REST API with webhook pattern

## 🎯 Key Files

- `eurosat/server_app.py` - Main Flower server with webhook integration
- `frontend/webhook_server.py` - FastAPI webhook server  
- `frontend/src/components/EarthViewer.tsx` - 3D satellite visualization
- `frontend/src/components/BatteryConsoleView.tsx` - Battery status console
- `eurosat/battery_aware_strategy.py` - Battery-aware federated averaging
- `eurosat/battery_simulation.py` - Satellite battery simulation

This system demonstrates how federated learning can work with resource-constrained edge devices (satellites) while providing rich real-time visualization of the training process.