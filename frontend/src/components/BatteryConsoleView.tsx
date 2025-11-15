import { useEffect, useState } from 'react';
import { useApi } from '@/contexts/ApiContext';

interface BatteryData {
  battery: number;
  in_sunlight: boolean;
  can_train: boolean;
  status: string;
}

interface BatteryUpdate {
  type: string;
  round: number;
  battery_levels: Record<string, BatteryData>;
  timestamp: string;
}

interface BatteryResponse {
  status: string;
  data?: BatteryUpdate;
  message?: string;
}

export default function BatteryConsoleView({ isVisible = false }: { isVisible?: boolean }) {
  const [batteryData, setBatteryData] = useState<BatteryUpdate | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { fetchData } = useApi();

  useEffect(() => {
    if (!isVisible) return;

    const fetchBatteryData = async () => {
      try {
        setIsLoading(true);
        const response: BatteryResponse = await fetchData('/api/battery/latest');
        
        if (response.status === 'success' && response.data) {
          setBatteryData(response.data);
        }
      } catch (error) {
        console.log('Battery data not available yet:', error);
      } finally {
        setIsLoading(false);
      }
    };

    // Fetch immediately
    fetchBatteryData();

    // Set up polling for updates
    const interval = setInterval(fetchBatteryData, 200); // Update every 200ms

    return () => clearInterval(interval);
  }, [isVisible, fetchData]);

  if (!isVisible || !batteryData) {
    return null;
  }

  const satellites = Object.entries(batteryData.battery_levels);
  const roundNumber = batteryData.round;

  return (
    <div className="absolute top-4 right-4 z-10 bg-black/90 text-green-400 p-4 rounded-lg border border-green-400/30 font-mono text-sm max-w-md">
      <div className="mb-2 text-center">
        <div className="text-green-200">🛰️ FEDERATED LEARNING ROUND {roundNumber} 🛰️</div>
        <div className="text-xs text-green-600">Battery Status Report</div>
      </div>
      
      <div className="space-y-1">
        {satellites.map(([satId, data]) => {
          const satNumber = satId.replace('sat-', '');
          const batteryPercent = Math.round(data.battery);
          const barLength = Math.floor(batteryPercent / 5); // 20 chars max
          const barFilled = '█'.repeat(barLength);
          const barEmpty = '░'.repeat(20 - barLength);
          
          // Determine color based on battery level
          let batteryColor = 'text-red-400';
          if (batteryPercent >= 80) batteryColor = 'text-green-400';
          else if (batteryPercent >= 50) batteryColor = 'text-yellow-400';
          else if (batteryPercent >= 30) batteryColor = 'text-orange-400';
          
          const sunIcon = data.in_sunlight ? '☀️' : '🌙';
          const statusIcon = data.can_train ? '✓' : '✗';
          const trainingIndicator = data.status === 'operational' ? '🚀' : '⚠️';
          
          return (
            <div key={satId} className="flex items-center gap-2 text-xs">
              <span className="w-12">Sat {satNumber}</span>
              <span className="w-4">{sunIcon}</span>
              <span className="w-4">{statusIcon}</span>
              <span className={`w-4 ${batteryColor}`}>[</span>
              <span className={batteryColor}>{barFilled}</span>
              <span className="text-gray-600">{barEmpty}</span>
              <span className={`w-4 ${batteryColor}`}>]</span>
              <span className={`w-12 ${batteryColor}`}>{batteryPercent.toString().padStart(3, ' ')}%</span>
              <span className="w-4">{trainingIndicator}</span>
            </div>
          );
        })}
      </div>
      
      <div className="mt-2 pt-2 border-t border-green-400/30 text-xs text-green-600">
        Last Update: {new Date(batteryData.timestamp).toLocaleTimeString()}
        {isLoading && <span className="ml-2 animate-pulse">⟳</span>}
      </div>
    </div>
  );
}