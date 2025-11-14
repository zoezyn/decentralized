import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, useGLTF, Stars } from "@react-three/drei";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { useApi } from "@/contexts/ApiContext";

function Earth() {
  const { scene } = useGLTF('/models/Earth_1_12756.glb');
  
  // Clone the scene to avoid issues with multiple renders
  const clonedScene = scene.clone();
  
  return (
    <primitive 
      object={clonedScene} 
      scale={0.01}
      rotation={[0.1, 0, 0]}
      position={[0, 0, 0]}
    />
  );
}

type SatelliteConfig = {
  id: string;
  name: string;
  orbitRadius: number;
  baseOrbitSpeed: number;
  baseRotationSpeed: number;
  color: string;
  initialAngle?: number;
};

function OrbitingCube({
  config,
  onSelect,
}: {
  config: SatelliteConfig;
  onSelect: (satelliteId: string) => void;
}) {
  const cubeRef = useRef<THREE.Group>(null);
  const orbitAngleRef = useRef(config.initialAngle ?? 0);
  const orbitSpeedRef = useRef(config.baseOrbitSpeed * 2);
  const rotationSpeedRef = useRef(config.baseRotationSpeed * 2);

  const targetOrbitSpeed = config.baseOrbitSpeed;
  const targetRotationSpeed = config.baseRotationSpeed;

  const handlePointerOver = useCallback(() => {
    if (typeof document !== "undefined") {
      document.body.style.cursor = "pointer";
    }
  }, []);

  const handlePointerOut = useCallback(() => {
    if (typeof document !== "undefined") {
      document.body.style.cursor = "auto";
    }
  }, []);

  const handleClick = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();
      onSelect(config.id);
    },
    [config.id, onSelect],
  );

  useEffect(
    () => () => {
      if (typeof document !== "undefined") {
        document.body.style.cursor = "auto";
      }
    },
    [],
  );

  useFrame((_, delta) => {
    if (!cubeRef.current) {
      return;
    }

    orbitSpeedRef.current = THREE.MathUtils.damp(orbitSpeedRef.current, targetOrbitSpeed, 2.5, delta);
    orbitAngleRef.current += orbitSpeedRef.current * delta;

    const x = Math.cos(orbitAngleRef.current) * config.orbitRadius;
    const z = Math.sin(orbitAngleRef.current) * config.orbitRadius;
    cubeRef.current.position.set(x, 0, z);

    rotationSpeedRef.current = THREE.MathUtils.damp(rotationSpeedRef.current, targetRotationSpeed, 3, delta);
    const rotationStep = rotationSpeedRef.current * delta;
    cubeRef.current.rotation.x += rotationStep;
    cubeRef.current.rotation.y += rotationStep * 0.8;
  });

  return (
    <group
      ref={cubeRef}
      onClick={handleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      <mesh>
        <boxGeometry args={[0.5, 0.5, 0.5]} />
        <meshStandardMaterial 
          color={config.color}
          emissive={config.color}
          emissiveIntensity={0.4}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>
    </group>
  );
}

function Loader() {
  return (
    <mesh>
      <sphereGeometry args={[3, 32, 32]} />
      <meshStandardMaterial color="#0ea5e9" wireframe opacity={0.3} transparent />
    </mesh>
  );
}

export default function EarthViewer() {
  const [selectedSatelliteId, setSelectedSatelliteId] = useState<string | null>(null);
  const { fetchData } = useApi();

  const satellites: SatelliteConfig[] = [
    {
      id: "sat-forest",
      name: "Forest Canopy Watcher",
      orbitRadius: 5.4,
      baseOrbitSpeed: 0.38,
      baseRotationSpeed: 0.55,
      color: "#22c55e",
      initialAngle: 0,
    },
    {
      id: "sat-brushland",
      name: "Brushland Sentinel",
      orbitRadius: 5.9,
      baseOrbitSpeed: 0.34,
      baseRotationSpeed: 0.5,
      color: "#84cc16",
      initialAngle: Math.PI / 5,
    },
    {
      id: "sat-annual-crop",
      name: "Crop Cycle Tracker",
      orbitRadius: 6.4,
      baseOrbitSpeed: 0.32,
      baseRotationSpeed: 0.48,
      color: "#facc15",
      initialAngle: (2 * Math.PI) / 5,
    },
    {
      id: "sat-pasture",
      name: "Pasture Guardian",
      orbitRadius: 6.9,
      baseOrbitSpeed: 0.3,
      baseRotationSpeed: 0.45,
      color: "#f97316",
      initialAngle: (3 * Math.PI) / 5,
    },
    {
      id: "sat-permanent-crop",
      name: "Orchard Monitor",
      orbitRadius: 7.4,
      baseOrbitSpeed: 0.28,
      baseRotationSpeed: 0.42,
      color: "#fb923c",
      initialAngle: (4 * Math.PI) / 5,
    },
    {
      id: "sat-highway",
      name: "Urban Artery Surveyor",
      orbitRadius: 7.9,
      baseOrbitSpeed: 0.26,
      baseRotationSpeed: 0.46,
      color: "#6366f1",
      initialAngle: Math.PI,
    },
    {
      id: "sat-industrial",
      name: "Industrial Beacon",
      orbitRadius: 8.4,
      baseOrbitSpeed: 0.24,
      baseRotationSpeed: 0.44,
      color: "#a855f7",
      initialAngle: (6 * Math.PI) / 5,
    },
    {
      id: "sat-residential",
      name: "Residential Skyline Sentinel",
      orbitRadius: 8.9,
      baseOrbitSpeed: 0.22,
      baseRotationSpeed: 0.4,
      color: "#ec4899",
      initialAngle: (7 * Math.PI) / 5,
    },
    {
      id: "sat-river",
      name: "Riverway Sentinel",
      orbitRadius: 9.4,
      baseOrbitSpeed: 0.2,
      baseRotationSpeed: 0.38,
      color: "#0ea5e9",
      initialAngle: (8 * Math.PI) / 5,
    },
    {
      id: "sat-lake",
      name: "Coastal Blueguard",
      orbitRadius: 9.9,
      baseOrbitSpeed: 0.18,
      baseRotationSpeed: 0.35,
      color: "#38bdf8",
      initialAngle: (9 * Math.PI) / 5,
    },
  ];

  const telemetryBySatellite: Record<string, { label: string; value: string; detail: string }[]> = {
    "sat-forest": [
      { label: "Battery", value: "88%", detail: "Solar arrays optimized for dense canopy angles" },
      { label: "Canopy Health", value: "94%", detail: "NDVI index steady across reserve sectors" },
      { label: "Alert Tier", value: "Low", detail: "No wildfire signatures detected this orbit" },
    ],
    "sat-brushland": [
      { label: "Battery", value: "85%", detail: "Dust shedding cycle finished 2 min ago" },
      { label: "Shrubland Moisture", value: "41%", detail: "Below seasonal average by 6%" },
      { label: "Thermal Anomalies", value: "2 hotspots", detail: "Monitoring western ridges" },
    ],
    "sat-annual-crop": [
      { label: "Battery", value: "92%", detail: "Panel tracking aligned to early sunrise" },
      { label: "Crop Vigor", value: "89%", detail: "Normalized difference moisture rising" },
      { label: "Data Rate", value: "1.6 Gbps", detail: "Yield forecasts streaming to agronomists" },
    ],
    "sat-pasture": [
      { label: "Battery", value: "79%", detail: "Cloud cover reduced solar gain by 8%" },
      { label: "Biomass Density", value: "73%", detail: "Grazing pressure within plan" },
      { label: "Livestock Sensors", value: "177 active", detail: "Telemetry synced to ranch hubs" },
    ],
    "sat-permanent-crop": [
      { label: "Battery", value: "84%", detail: "High-latitude orbit maintaining reserves" },
      { label: "Orchard Stress Index", value: "12%", detail: "Irrigation anomalies localized" },
      { label: "Pollination Bands", value: "Green", detail: "Bee activity within optimal range" },
    ],
    "sat-highway": [
      { label: "Battery", value: "87%", detail: "Regenerative braking on reaction wheels engaged" },
      { label: "Traffic Flow", value: "62 km/h avg", detail: "Morning peak easing along corridor" },
      { label: "Incident Flags", value: "1 advisory", detail: "Debris near interchange 47" },
    ],
    "sat-industrial": [
      { label: "Battery", value: "90%", detail: "Thermal load balanced via radiator sweep" },
      { label: "Emission Index", value: "0.64 ppm", detail: "Stacks within compliance" },
      { label: "Downlink Rate", value: "1.2 Gbps", detail: "High-resolution mapping uplink" },
    ],
    "sat-residential": [
      { label: "Battery", value: "83%", detail: "Night cycle demand offset by super-capacitors" },
      { label: "Urban Growth", value: "4.1% YoY", detail: "New construction zones mapped" },
      { label: "Heat Islands", value: "Moderate", detail: "Cooling interventions recommended" },
    ],
    "sat-river": [
      { label: "Battery", value: "89%", detail: "Hydroreflective calibration nominal" },
      { label: "Flow Rate", value: "3,800 m3/s", detail: "Snowmelt surge trending upward" },
      { label: "Sediment Load", value: "Low", detail: "Water clarity favorable for habitats" },
    ],
    "sat-lake": [
      { label: "Battery", value: "93%", detail: "Marine-grade panels at peak efficiency" },
      { label: "Algal Bloom Risk", value: "Minimal", detail: "Chlorophyll-a concentration stable" },
      { label: "Coastal Coverage", value: "97%", detail: "Buoy network fully synchronized" },
    ],
  };

  const statusBySatellite: Record<string, string[]> = {
    "sat-forest": [
      "Next pass over boreal reserve in 11 minutes",
      "Infrared fire watch grid reporting all clear",
      "Ground rangers synced via low-bandwidth mesh",
    ],
    "sat-brushland": [
      "Brushland thermal mapping queued for sunset revisit",
      "Wind vectors updated to anticipate flare spread",
      "Drone partner missions scheduled for ridge survey",
    ],
    "sat-annual-crop": [
      "Yield forecast package disseminated to co-ops",
      "Soil moisture probes cross-validated via satlink",
      "Cloud shadows compensated in growth analytics",
    ],
    "sat-pasture": [
      "Pasture rotation recommendations issued to ranch HQ",
      "Collar telemetry ingest at 98% completeness",
      "Storm-front advisory triggered for southern plains",
    ],
    "sat-permanent-crop": [
      "Orchard frost alert protocol on standby overnight",
      "Pollen corridor imagery dispatched to agronomists",
      "Servo calibration scheduled for dawn overflight",
    ],
    "sat-highway": [
      "Dynamic traffic signage request sent to DOT hub",
      "Road surface degradation flagged at sector A17",
      "Lidar sweep ready for tonight's maintenance window",
    ],
    "sat-industrial": [
      "Compliance report uploaded to environmental bureau",
      "Spectral emissions trending below quarterly cap",
      "Cooling tower plume models refreshed hourly",
    ],
    "sat-residential": [
      "Urban expansion layer synced with planning teams",
      "Heat resiliency blueprint updated for district 9",
      "Noise pollution sensors calibrated with sat imagery",
    ],
    "sat-river": [
      "Floodplain alert threshold at watch status",
      "River-mouth dredging schedule adjusted with new data",
      "Fish migration beacons aligned with turbidity map",
    ],
    "sat-lake": [
      "Coastal erosion watchlist transmitted to harbor masters",
      "Maritime traffic density trending moderate",
      "Next offshore lidar bathymetry sweep in 23 minutes",
    ],
  };

  const handleSatelliteSelect = useCallback((satelliteId: string) => {
    setSelectedSatelliteId(satelliteId);
  }, []);

  return (
    <div className="w-full h-screen relative">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 45 }}
        className="bg-background"
      >
        <Suspense fallback={<Loader />}>
          <ambientLight intensity={1.2} />
          <directionalLight position={[10, 10, 5]} intensity={2.8} />
          <pointLight position={[-10, -10, -5]} intensity={1.5} color="#0ea5e9" />
          <hemisphereLight args={["#8cc0ff", "#050505", 0.6]} />
          
          <Earth />
          {satellites.map((satellite) => (
            <OrbitingCube key={satellite.id} config={satellite} onSelect={handleSatelliteSelect} />
          ))}
          
          <Stars
            radius={100} 
            depth={50} 
            count={5000} 
            factor={4} 
            saturation={0} 
            fade 
            speed={1}
          />
          
          <OrbitControls 
            enableZoom={true}
            enablePan={false}
            minDistance={5}
            maxDistance={40}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Suspense>
      </Canvas>
      <Sheet open={Boolean(selectedSatelliteId)} onOpenChange={(open) => !open && setSelectedSatelliteId(null)}>
        {selectedSatelliteId ? (
          <SheetContent side="right" className="sm:max-w-md space-y-6">
            <SheetHeader>
              <SheetTitle>{satellites.find((s) => s.id === selectedSatelliteId)?.name}</SheetTitle>
              <SheetDescription>Realtime status for the selected orbital monitor.</SheetDescription>
            </SheetHeader>
            <div className="grid grid-cols-1 gap-4">
              {(telemetryBySatellite[selectedSatelliteId] ?? []).map((item) => (
                <div key={item.label} className="rounded-lg border border-border/80 bg-secondary/30 p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">{item.label}</p>
                  <p className="mt-1 text-2xl font-semibold text-primary">{item.value}</p>
                  <p className="mt-2 text-sm text-muted-foreground">{item.detail}</p>
                </div>
              ))}
            </div>
            <div className="space-y-3">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Operational Notes</h3>
              <ul className="space-y-2">
                {(statusBySatellite[selectedSatelliteId] ?? []).map((note) => (
                  <li key={note} className="text-sm text-foreground/90">
                    - {note}
                  </li>
                ))}
              </ul>
            </div>
          </SheetContent>
        ) : null}
      </Sheet>
    </div>
  );
}
