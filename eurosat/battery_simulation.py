"""
Battery Simulation for Satellite Federated Learning
Simulates solar charging, training costs, and communication costs
"""

from typing import Dict, List, Tuple
import random


class BatterySimulator:
    """Simulates battery levels for satellite constellation"""
    
    def __init__(
        self,
        num_satellites: int = 10,
        initial_battery: float = 80.0,
        charge_rate: float = 3.0,
        train_cost: float = 15.0,
        comm_cost: float = 5.0,
        min_battery_threshold: float = 30.0,
        day_night_cycle: bool = True,
        orbit_period: int = 6,  # rounds per orbit (day/night cycle)
    ):
        """
        Initialize battery simulator
        
        Args:
            num_satellites: Number of satellites in constellation
            initial_battery: Starting battery level (0-100%)
            charge_rate: Battery charge per round when in sunlight (%)
            train_cost: Battery drain for training (%)
            comm_cost: Battery drain for communication (%)
            min_battery_threshold: Minimum battery to participate (%)
            day_night_cycle: Whether to simulate day/night cycles
            orbit_period: Rounds per complete orbit (for day/night)
        """
        self.num_satellites = num_satellites
        self.charge_rate = charge_rate
        self.train_cost = train_cost
        self.comm_cost = comm_cost
        self.min_battery_threshold = min_battery_threshold
        self.day_night_cycle = day_night_cycle
        self.orbit_period = orbit_period
        
        # Initialize batteries with some variation
        self.batteries = {
            i: initial_battery + random.uniform(-10, 10) 
            for i in range(num_satellites)
        }
        
        # Ensure all start above threshold
        for i in self.batteries:
            self.batteries[i] = max(self.min_battery_threshold + 5, 
                                   min(100, self.batteries[i]))
        
        # Track statistics
        self.history = []
        self.skipped_due_to_battery = {i: 0 for i in range(num_satellites)}
    
    def is_in_sunlight(self, round_num: int, satellite_id: int) -> bool:
        """
        Determine if satellite is in sunlight
        
        Simple model: Each satellite has offset in orbit, 
        half the orbit is in sunlight, half in shadow
        """
        if not self.day_night_cycle:
            return True
        
        # Each satellite has different orbit phase
        orbit_position = (round_num + satellite_id) % self.orbit_period
        return orbit_position < (self.orbit_period // 2)
    
    def get_available_clients(self, round_num: int) -> List[int]:
        """
        Get list of satellite IDs that have enough battery to train
        
        Args:
            round_num: Current training round
            
        Returns:
            List of satellite IDs with battery > threshold
        """
        available = []
        for sat_id in range(self.num_satellites):
            if self.batteries[sat_id] >= self.min_battery_threshold:
                available.append(sat_id)
            else:
                self.skipped_due_to_battery[sat_id] += 1
        
        return available
    
    def charge_batteries(self, round_num: int):
        """
        Charge all satellite batteries based on sunlight exposure
        
        Args:
            round_num: Current training round
        """
        for sat_id in range(self.num_satellites):
            if self.is_in_sunlight(round_num, sat_id):
                # Charge battery
                charge = self.charge_rate
                self.batteries[sat_id] = min(100, self.batteries[sat_id] + charge)
            # else: in shadow, no charging
    
    def consume_battery(self, satellite_ids: List[int]):
        """
        Consume battery for satellites that participated in training
        
        Args:
            satellite_ids: List of satellites that trained this round
        """
        for sat_id in satellite_ids:
            # Training + communication cost
            total_cost = self.train_cost + self.comm_cost
            self.batteries[sat_id] = max(0, self.batteries[sat_id] - total_cost)
    
    def step(self, round_num: int, selected_satellites: List[int]) -> Dict:
        """
        Simulate one round: charge batteries, consume for selected clients
        
        Args:
            round_num: Current round number
            selected_satellites: IDs of satellites selected for training
            
        Returns:
            Dictionary with battery statistics
        """
        # First charge all batteries
        self.charge_batteries(round_num)
        
        # Then consume for selected satellites
        self.consume_battery(selected_satellites)
        
        # Collect statistics
        stats = {
            'round': round_num,
            'avg_battery': sum(self.batteries.values()) / len(self.batteries),
            'min_battery': min(self.batteries.values()),
            'max_battery': max(self.batteries.values()),
            'available_clients': len(self.get_available_clients(round_num)),
            'selected_clients': len(selected_satellites),
            'batteries': dict(self.batteries),  # Copy of current state
        }
        
        self.history.append(stats)
        return stats
    
    def get_battery_level(self, satellite_id: int) -> float:
        """Get current battery level for a satellite"""
        return self.batteries[satellite_id]
    
    def get_statistics(self) -> Dict:
        """Get overall battery usage statistics"""
        return {
            'current_batteries': dict(self.batteries),
            'skipped_counts': dict(self.skipped_due_to_battery),
            'total_skipped': sum(self.skipped_due_to_battery.values()),
            'avg_battery': sum(self.batteries.values()) / len(self.batteries),
            'satellites_below_threshold': sum(
                1 for b in self.batteries.values() 
                if b < self.min_battery_threshold
            ),
        }
    
    def print_status(self, round_num: int):
        """Print current battery status"""
        print(f"\n🔋 Battery Status - Round {round_num}")
        print("=" * 70)
        
        for sat_id in range(self.num_satellites):
            battery = self.batteries[sat_id]
            in_sun = "☀️" if self.is_in_sunlight(round_num, sat_id) else "🌑"
            status = "✓" if battery >= self.min_battery_threshold else "✗"
            
            # Battery bar
            bar_length = int(battery / 5)  # 20 chars = 100%
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"  Sat {sat_id} {in_sun} {status} [{bar}] {battery:5.1f}%")
        
        print("=" * 70)
        available = self.get_available_clients(round_num)
        print(f"Available: {len(available)}/{self.num_satellites} satellites")
        print(f"Average battery: {sum(self.batteries.values()) / len(self.batteries):.1f}%")


if __name__ == "__main__":
    # Test the battery simulator
    print("Testing Battery Simulator\n")
    
    sim = BatterySimulator(
        num_satellites=10,
        initial_battery=80,
        charge_rate=3,
        train_cost=15,
        comm_cost=5,
        min_battery_threshold=30,
        day_night_cycle=True,
        orbit_period=6
    )
    
    # Simulate 20 rounds
    for round_num in range(1, 21):
        sim.print_status(round_num)
        
        # Get available clients
        available = sim.get_available_clients(round_num)
        
        # Select 50% of available clients
        num_select = max(1, len(available) // 2)
        selected = random.sample(available, min(num_select, len(available)))
        
        print(f"\nSelected for training: {selected}\n")
        
        # Simulate round
        stats = sim.step(round_num, selected)
        
        input("Press Enter for next round...")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    final_stats = sim.get_statistics()
    print(f"Average battery: {final_stats['avg_battery']:.1f}%")
    print(f"Total training rounds skipped: {final_stats['total_skipped']}")
    print(f"\nSkipped per satellite:")
    for sat_id, count in final_stats['skipped_counts'].items():
        print(f"  Satellite {sat_id}: {count} rounds skipped")

