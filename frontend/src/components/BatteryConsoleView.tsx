import { useApi } from "@/contexts/ApiContext";
import { useEffect, useState } from "react";

interface BatteryInfo {
  battery: number;
  in_sunlight: boolean;
  can_train: boolean;
  status: string;
}

interface BatteryConsoleViewProps {
  isVisible: boolean;
}

export default function BatteryConsoleView({ isVisible }: BatteryConsoleViewProps) {
  const [batteryData, setBatteryData] = useState<Record<string, BatteryInfo>>({});
  const [currentRound, setCurrentRound] = useState<number>(0);
  const { fetchData } = useApi();

  useEffect(() => {
    if (!isVisible) return;

    const fetchBatteryData = async () => {
      try {
        const response = await fetchData('/api/battery/latest');
        if (response.status === 'success' && response.data?.battery_levels) {
          setBatteryData(response.data.battery_levels);
          if (response.data.round) {
            setCurrentRound(response.data.round);
          }
        }
      } catch (error) {
        console.log('Battery fetch error:', error);
      }
    };

    const interval = setInterval(fetchBatteryData, 500);
    fetchBatteryData();

    return () => clearInterval(interval);
  }, [isVisible, fetchData]);

  if (!isVisible) return null;

  // Get battery level color
  const getBatteryColor = (level: number): string => {
    if (level >= 70) return "text-green-300";
    if (level >= 50) return "text-yellow-300";  
    if (level >= 30) return "text-orange-300";
    if (level >= 15) return "text-red-300";
    return "text-red-500";
  };

  // Get battery bar color
  const getBatteryBarColor = (level: number): string => {
    if (level >= 70) return "text-green-400";
    if (level >= 50) return "text-yellow-400";  
    if (level >= 30) return "text-orange-400";
    if (level >= 15) return "text-red-400";
    return "text-red-500";
  };

  // Format battery bar
  const getBatteryBar = (level: number): string => {
    const barLength = Math.floor(level / 5); // 20 char bar max
    const filledBars = "█".repeat(barLength);
    const emptyBars = "░".repeat(20 - barLength);
    return filledBars + emptyBars;
  };

  return (
    <div className="absolute top-16 left-4 z-10 bg-black/95 text-green-400 p-4 rounded-lg border border-green-400/30 font-mono text-xs max-w-md">
      <div className="mb-3">
        <div className="text-green-200 text-center mb-1">🔋 SATELLITE BATTERY STATUS</div>
        <div className="text-center text-xs text-green-600">Round {currentRound}</div>
        <div className="text-center text-xs text-gray-400">Battery Threshold: 30%</div>
      </div>
      
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {Object.entries(batteryData)
          .sort(([a], [b]) => {
            const aNum = parseInt(a.replace('sat-', ''));
            const bNum = parseInt(b.replace('sat-', ''));
            return aNum - bNum;
          })
          .map(([satId, info]) => {
            const batteryLevel = Math.round(info.battery);
            const canTrain = info.can_train;
            const status = canTrain ? "✓" : "✗";
            const statusColor = canTrain ? "text-green-400" : "text-red-400";
            const sunIcon = info.in_sunlight ? "☀️" : "🌙";
            
            return (
              <div key={satId} className="flex items-center space-x-2">
                <span className="w-8 text-gray-300">{satId.replace('sat-', 'S')}</span>
                <span className={`w-4 ${statusColor}`}>{status}</span>
                <span className={`w-12 ${getBatteryColor(batteryLevel)} text-right`}>
                  {batteryLevel}%
                </span>
                <span className={`${getBatteryBarColor(batteryLevel)} font-mono text-xs`}>
                  [{getBatteryBar(batteryLevel)}]
                </span>
                <span className="w-6">{sunIcon}</span>
              </div>
            );
          })}
      </div>
      
      <div className="mt-3 pt-2 border-t border-green-400/30 text-xs">
        <div className="flex justify-between">
          <span>Legend:</span>
          <span className="text-gray-400">✓=Ready ✗=Low ☀️=Day 🌙=Night</span>
        </div>
      </div>
    </div>
  );
}