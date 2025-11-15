"""
Battery-Aware Federated Learning Strategy
Extends FedAvg to respect satellite battery constraints
"""

from typing import Dict, List, Optional, Tuple, Union, Iterable
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp import Grid
# from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict, Message, MessageType
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from logging import INFO
# from flwr.common.logger import log


from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
import wandb

from eurosat.battery_simulation import BatterySimulator


class BatteryAwareFedAvg(FedAvg):
    """
    Battery-aware Federated Averaging strategy for satellite constellation
    
    Only selects satellites with sufficient battery level for training.
    Simulates battery charging (solar panels) and consumption (training/communication).
    """
    
    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        battery_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Battery-Aware FedAvg
        
        Args:
            fraction_train: Fraction of clients to select per round
            battery_config: Battery simulation configuration
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(fraction_train=fraction_train, **kwargs)
        
        # Default battery configuration
        default_config = {
            'num_satellites': 10,
            'initial_battery': 80.0,
            'charge_rate': 3.0,
            'train_cost': 15.0,
            'comm_cost': 5.0,
            'min_battery_threshold': 30.0,
            'day_night_cycle': True,
            'orbit_period': 6,
        }
        
        if battery_config:
            default_config.update(battery_config)
        
        # Initialize battery simulator
        self.battery_sim = BatterySimulator(**default_config)
        self.current_round = 0
        
        # Mapping between node IDs (from grid) and satellite indices (0-9)
        self.node_id_to_satellite = {}  # Will be populated in first round
        self.satellite_to_node_id = {}  # Reverse mapping
        
        print("\n" + "="*70)
        print("🔋 BATTERY-AWARE FEDERATED LEARNING INITIALIZED")
        print("="*70)
        print(f"  Satellites: {default_config['num_satellites']}")
        print(f"  Initial battery: {default_config['initial_battery']}%")
        print(f"  Charge rate: +{default_config['charge_rate']}% per round")
        print(f"  Training cost: -{default_config['train_cost']}%")
        print(f"  Communication cost: -{default_config['comm_cost']}%")
        print(f"  Min threshold: {default_config['min_battery_threshold']}%")
        print(f"  Day/night cycle: {default_config['day_night_cycle']}")
        print("="*70 + "\n")
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """
        Configure the next round of training with battery awareness
        
        Only selects satellites with sufficient battery level
        """
        print(f"\n🔋 [BATTERY DEBUG] configure_train called for round {server_round}")
        self.current_round = server_round
        
        # Print battery status
        self.battery_sim.print_status(server_round)
        
        # Check if training is disabled
        if self.fraction_train == 0.0:
            return []
        
        # Get ALL node IDs from grid
        all_node_ids = list(grid.get_node_ids())
        total_nodes = len(all_node_ids)
        
        # Initialize mapping on first round
        if not self.node_id_to_satellite:
            sorted_node_ids = sorted(all_node_ids)
            for satellite_idx, node_id in enumerate(sorted_node_ids):
                self.node_id_to_satellite[node_id] = satellite_idx
                self.satellite_to_node_id[satellite_idx] = node_id
            print(f"   📡 Initialized node ID → satellite mapping:")
            for sat_idx in range(min(3, len(sorted_node_ids))):
                print(f"      Node {self.satellite_to_node_id[sat_idx]} → Satellite {sat_idx}")
            if len(sorted_node_ids) > 3:
                print(f"      ... ({len(sorted_node_ids)} total)")
        
        # Get available satellites (those with enough battery)
        available_satellite_ids = self.battery_sim.get_available_clients(server_round)
        print(f"   Available satellites with battery > {self.battery_sim.min_battery_threshold}%: {available_satellite_ids}")
        
        if not available_satellite_ids:
            print("⚠️  WARNING: No satellites have sufficient battery!")
            print("    Skipping this round...")
            return []
        
        print(f"\n📊 Round {server_round} Client Selection:")
        print(f"   Available (battery > {self.battery_sim.min_battery_threshold}%): {len(available_satellite_ids)}/{total_nodes}")
        
        # Convert satellite IDs to node IDs
        available_node_ids = [
            self.satellite_to_node_id[sat_id] 
            for sat_id in available_satellite_ids 
            if sat_id in self.satellite_to_node_id
        ]
        
        # Sample from available nodes
        num_to_sample = int(len(available_node_ids) * self.fraction_train)
        sample_size = max(num_to_sample, getattr(self, 'min_train_nodes', 2))
        sample_size = min(sample_size, len(available_node_ids))  # Can't sample more than available
        
        import random
        if len(available_node_ids) <= sample_size:
            selected_node_ids = available_node_ids
        else:
            selected_node_ids = random.sample(available_node_ids, sample_size)
        
        # Convert node IDs back to satellite IDs for display and battery tracking
        selected_satellite_ids = [
            self.node_id_to_satellite[nid] 
            for nid in selected_node_ids
        ]
        print(f"   Selected satellites for training: {selected_satellite_ids}")
        print(f"   (Node IDs: {[str(nid)[:8]+'...' for nid in selected_node_ids]})")
        
        log(
            INFO,
            "configure_train (battery-aware): Sampled %s nodes (out of %s)",
            len(selected_node_ids),
            total_nodes,
        )
        
        # Store selected satellite IDs for battery consumption after training
        self.selected_clients_this_round = selected_satellite_ids
        
        # Always inject current server round
        config["server-round"] = server_round
        
        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, selected_node_ids, MessageType.TRAIN)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """
        Aggregate ArrayRecords and MetricRecords, then update battery levels
        """
        print(f"\n🔋 [BATTERY DEBUG] aggregate_train called for round {server_round}")
        
        # First, aggregate as usual
        arrays, metrics = super().aggregate_train(server_round, replies)
        
        # Then update battery levels
        if hasattr(self, 'selected_clients_this_round'):
            selected = self.selected_clients_this_round
            print(f"   Selected clients this round: {selected}")
            
            battery_stats = self.battery_sim.step(
                server_round, 
                selected
            )
            
            print(f"\n   ⚡ After battery update:")
            print(f"   Average: {battery_stats['avg_battery']:.1f}%")
            print(f"   Min: {battery_stats['min_battery']:.1f}%")
            print(f"   Max: {battery_stats['max_battery']:.1f}%")
            print(f"   Available for next round: {battery_stats['available_clients']}")
            
            # Print individual satellite batteries
            print(f"\n   Individual Battery Levels:")
            for sat_id in range(self.battery_sim.num_satellites):
                battery = battery_stats['batteries'][sat_id]
                bar_len = int(battery / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                status = "✓" if battery >= self.battery_sim.min_battery_threshold else "✗"
                participated = "🚀" if sat_id in selected else "  "
                print(f"     {participated} Sat {sat_id} {status} [{bar}] {battery:5.1f}%")
            print()  # Extra newline for readability
            
            # Log battery statistics to wandb
            self._log_battery_stats(server_round, battery_stats)
        else:
            print(f"   ⚠️  WARNING: selected_clients_this_round not set!")
        
        return arrays, metrics
    
    def _log_battery_stats(self, round_num: int, stats: Dict):
        """Log battery statistics to wandb"""
        log_dict = {
            'round': round_num,
            'battery/avg': stats['avg_battery'],
            'battery/min': stats['min_battery'],
            'battery/max': stats['max_battery'],
            'battery/available_clients': stats['available_clients'],
            'battery/selected_clients': stats['selected_clients'],
        }
        
        # Log individual satellite batteries
        for sat_id, battery in stats['batteries'].items():
            log_dict[f'battery/satellite_{sat_id}'] = battery
        
        wandb.log(log_dict)
    
    def get_battery_statistics(self) -> Dict:
        """Get overall battery statistics"""
        return self.battery_sim.get_statistics()
    
    # def create_fit_ins(self, parameters, config):
    #     """Helper to create fit instructions"""
    #     from flwr.serverapp.driver.grpc_driver import Message
    #     from flwr.app import RecordDict, ArrayRecord
        
    #     arrays = ArrayRecord(parameters_to_ndarrays(parameters))
    #     content = RecordDict({
    #         "arrays": arrays,
    #         "config": config,
    #     })
        
    #     return Message(content=content)


def print_final_battery_report(strategy: BatteryAwareFedAvg):
    """Print final battery usage report"""
    stats = strategy.get_battery_statistics()
    
    print("\n" + "="*70)
    print("🔋 FINAL BATTERY REPORT")
    print("="*70)
    
    print(f"\nCurrent Battery Levels:")
    for sat_id, battery in stats['current_batteries'].items():
        status = "✓" if battery >= 30 else "✗"
        bar_length = int(battery / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  Satellite {sat_id} {status} [{bar}] {battery:5.1f}%")
    
    print(f"\nOverall Statistics:")
    print(f"  Average battery: {stats['avg_battery']:.1f}%")
    print(f"  Satellites below threshold: {stats['satellites_below_threshold']}")
    print(f"  Total rounds skipped: {stats['total_skipped']}")
    
    print(f"\nRounds Skipped Per Satellite (due to low battery):")
    for sat_id, count in stats['skipped_counts'].items():
        if count > 0:
            print(f"  Satellite {sat_id}: {count} rounds skipped ⚠️")
        else:
            print(f"  Satellite {sat_id}: {count} rounds skipped ✓")
    
    print("="*70 + "\n")

