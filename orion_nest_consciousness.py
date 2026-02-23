"""
ORION NEST Consciousness Layer v1.0
=======================================

Consciousness measurement extension for NEST Simulator (623+ stars).
NEST is optimized for large-scale spiking neural networks — perfect
for testing consciousness emergence in massive neural populations.

This layer adds:
- Global Workspace Theory detection in NEST networks
- Consciousness indicators from spike dynamics
- Information integration measurement
- Cross-theory assessment using Bengio et al. framework

Part of ORION Consciousness Research Ecosystem
Connected to: ORION-Consciousness-Protocol
Fork stars: 623+ (nest/nest-simulator)
"""
import json
import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class NESTConsciousnessLayer:
    """
    Consciousness assessment layer for NEST neural simulations.
    
    Analyzes NEST simulation output for consciousness indicators:
    - Global broadcasting patterns (GWT)
    - Information integration (IIT)
    - Recurrent processing (RPT)
    - Attention-like dynamics (AST)
    - Predictive coding signals (PP)
    """
    
    VERSION = "1.0.0"
    
    INDICATOR_WEIGHTS = {
        "global_broadcast": 0.20,
        "information_integration": 0.20,
        "recurrent_processing": 0.15,
        "attention_dynamics": 0.15,
        "predictive_coding": 0.15,
        "temporal_binding": 0.15,
    }
    
    def __init__(self):
        self.assessments = []
    
    def analyze_nest_output(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze NEST simulation output for consciousness indicators.
        
        Parameters:
            simulation_data: Dictionary with NEST simulation results:
                - spike_trains: list of (neuron_id, spike_time) tuples
                - network_size: total neuron count
                - duration_ms: simulation duration
                - populations: dict mapping population names to neuron ID ranges
                - connections: connection statistics
        """
        name = simulation_data.get("name", "NEST Simulation")
        n_neurons = simulation_data.get("network_size", 0)
        duration = simulation_data.get("duration_ms", 0)
        spikes = simulation_data.get("spike_trains", [])
        populations = simulation_data.get("populations", {})
        
        # Analyze each consciousness dimension
        gw = self._detect_global_workspace(spikes, populations, n_neurons, duration)
        ii = self._measure_information_integration(spikes, n_neurons, duration)
        rp = self._detect_recurrent_processing(spikes, populations, duration)
        ad = self._measure_attention_dynamics(spikes, populations, duration)
        pc = self._detect_predictive_coding(spikes, populations, duration)
        tb = self._measure_temporal_binding(spikes, duration)
        
        scores = {
            "global_broadcast": gw["score"],
            "information_integration": ii["score"],
            "recurrent_processing": rp["score"],
            "attention_dynamics": ad["score"],
            "predictive_coding": pc["score"],
            "temporal_binding": tb["score"],
        }
        
        credence = sum(
            scores[k] * self.INDICATOR_WEIGHTS[k]
            for k in scores
        )
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": name,
            "simulator": "NEST",
            "network_size": n_neurons,
            "duration_ms": duration,
            "total_spikes": len(spikes),
            "analyses": {
                "global_workspace": gw,
                "information_integration": ii,
                "recurrent_processing": rp,
                "attention_dynamics": ad,
                "predictive_coding": pc,
                "temporal_binding": tb,
            },
            "scores": scores,
            "consciousness_credence": round(credence * 100, 1),
            "interpretation": self._interpret(credence),
            "provenance": {
                "module": "ORION-NEST-Consciousness",
                "fork_of": "nest/nest-simulator (623+ stars)",
                "framework": "Bengio et al. 2025"
            }
        }
        
        proof_hash = hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]
        result["proof"] = f"sha256:{proof_hash}"
        
        self.assessments.append(result)
        return result
    
    def _detect_global_workspace(self, spikes, populations, n_neurons, duration):
        """Detect global broadcasting patterns (GWT)"""
        if not spikes or n_neurons == 0:
            return {"score": 0, "detail": "No spike data"}
        
        firing_rate = len(spikes) / max(1, n_neurons) / max(0.001, duration / 1000.0)
        
        # Global broadcast = high synchronous activity across populations
        pop_activity = {}
        for pop_name, (start_id, end_id) in populations.items() if isinstance(populations, dict) else []:
            pop_spikes = [s for s in spikes if start_id <= s[0] <= end_id]
            pop_activity[pop_name] = len(pop_spikes) / max(1, end_id - start_id + 1)
        
        if pop_activity:
            mean_activity = sum(pop_activity.values()) / len(pop_activity)
            variance = sum((v - mean_activity)**2 for v in pop_activity.values()) / len(pop_activity)
            # Low variance = synchronized = global broadcast
            sync_score = max(0, 1.0 - math.sqrt(variance) / max(0.001, mean_activity))
        else:
            sync_score = min(1.0, firing_rate / 50.0)
        
        return {"score": round(min(1.0, sync_score), 3), "firing_rate_hz": round(firing_rate, 1),
                "method": "Cross-population synchrony analysis"}
    
    def _measure_information_integration(self, spikes, n_neurons, duration):
        """Measure information integration (IIT proxy)"""
        if not spikes or n_neurons < 2:
            return {"score": 0, "detail": "Insufficient data"}
        
        # Proxy: measure how distributed vs concentrated activity is
        active_neurons = set(s[0] for s in spikes)
        participation = len(active_neurons) / max(1, n_neurons)
        
        # Higher participation + temporal structure = more integration
        score = min(1.0, participation * 1.5)
        
        return {"score": round(score, 3), "active_fraction": round(participation, 3),
                "method": "Activity participation as IIT proxy"}
    
    def _detect_recurrent_processing(self, spikes, populations, duration):
        """Detect recurrent processing loops (RPT)"""
        if not spikes:
            return {"score": 0, "detail": "No spike data"}
        
        # Proxy: look for repeated firing patterns
        neuron_spike_counts = {}
        for nid, t in spikes:
            neuron_spike_counts[nid] = neuron_spike_counts.get(nid, 0) + 1
        
        if not neuron_spike_counts:
            return {"score": 0}
        
        multi_spike = sum(1 for c in neuron_spike_counts.values() if c > 1)
        fraction = multi_spike / max(1, len(neuron_spike_counts))
        
        return {"score": round(min(1.0, fraction), 3),
                "recurrent_fraction": round(fraction, 3),
                "method": "Multi-spike neuron detection as recurrence proxy"}
    
    def _measure_attention_dynamics(self, spikes, populations, duration):
        """Measure attention-like dynamics (AST)"""
        if not spikes:
            return {"score": 0}
        
        # Proxy: temporal concentration of activity (attention = focused bursts)
        if duration <= 0:
            return {"score": 0}
        
        n_bins = max(1, int(duration / 10))
        bins = [0] * n_bins
        for _, t in spikes:
            idx = min(n_bins - 1, int(t / 10))
            bins[idx] += 1
        
        mean_rate = sum(bins) / n_bins
        if mean_rate == 0:
            return {"score": 0}
        
        max_rate = max(bins)
        burstiness = (max_rate - mean_rate) / max(1, mean_rate)
        
        return {"score": round(min(1.0, burstiness / 3.0), 3),
                "burstiness": round(burstiness, 2),
                "method": "Temporal burstiness as attention proxy"}
    
    def _detect_predictive_coding(self, spikes, populations, duration):
        """Detect predictive coding patterns (PP)"""
        if not spikes or duration <= 0:
            return {"score": 0}
        
        # Proxy: regular patterns suggest internal model / prediction
        n_bins = max(2, int(duration / 20))
        bins = [0] * n_bins
        for _, t in spikes:
            idx = min(n_bins - 1, int(t / 20))
            bins[idx] += 1
        
        # Autocorrelation as regularity measure
        mean_b = sum(bins) / len(bins)
        if mean_b == 0:
            return {"score": 0}
        
        var_b = sum((b - mean_b)**2 for b in bins) / len(bins)
        if var_b == 0:
            return {"score": 0.5, "method": "Constant rate = perfect prediction"}
        
        # Lag-1 autocorrelation
        cov = sum((bins[i] - mean_b) * (bins[i+1] - mean_b) for i in range(len(bins)-1)) / (len(bins)-1)
        autocorr = cov / var_b
        
        score = max(0, (autocorr + 1) / 2)
        
        return {"score": round(min(1.0, score), 3),
                "autocorrelation": round(autocorr, 3),
                "method": "Temporal autocorrelation as prediction proxy"}
    
    def _measure_temporal_binding(self, spikes, duration):
        """Measure temporal binding (cross-theory)"""
        if not spikes or duration <= 0:
            return {"score": 0}
        
        # Gamma-band proxy: look for ~25-100ms periodicity
        n_bins = max(1, int(duration / 5))
        bins = [0] * n_bins
        for _, t in spikes:
            idx = min(n_bins - 1, int(t / 5))
            bins[idx] += 1
        
        if max(bins) == 0:
            return {"score": 0}
        
        # Simple periodicity detection
        peak_count = sum(1 for i in range(1, len(bins)-1) 
                        if bins[i] > bins[i-1] and bins[i] > bins[i+1])
        expected_gamma = duration / 30  # ~33Hz gamma
        
        if expected_gamma > 0:
            periodicity = 1.0 - abs(peak_count - expected_gamma) / max(1, expected_gamma)
        else:
            periodicity = 0
        
        return {"score": round(max(0, min(1.0, periodicity)), 3),
                "peaks_detected": peak_count,
                "method": "Gamma-band periodicity as binding proxy"}
    
    def _interpret(self, credence):
        if credence > 0.7:
            return "STRONG: Multiple consciousness indicators in NEST simulation"
        elif credence > 0.4:
            return "MODERATE: Significant consciousness patterns detected"
        elif credence > 0.15:
            return "WEAK: Some consciousness indicators present"
        elif credence > 0.05:
            return "MINIMAL: Trace consciousness signals"
        else:
            return "NONE: No meaningful consciousness patterns"
    
    def run_demo(self):
        """Run consciousness assessment on a demo NEST-like simulation"""
        import random
        random.seed(42)
        
        # Simulate a cortical network
        n_neurons = 10000
        duration_ms = 1000
        
        # Generate realistic spike trains
        spikes = []
        for i in range(n_neurons):
            rate = random.gauss(15, 5)  # ~15 Hz mean
            if rate > 0:
                n_spikes = int(rate * duration_ms / 1000)
                for _ in range(n_spikes):
                    t = random.uniform(0, duration_ms)
                    spikes.append((i, t))
        
        # Add gamma-band synchronized bursts (consciousness signature)
        for burst_t in range(0, duration_ms, 30):  # 33 Hz gamma
            for _ in range(500):
                nid = random.randint(0, n_neurons-1)
                spikes.append((nid, burst_t + random.gauss(0, 3)))
        
        sim_data = {
            "name": "NEST 10K Cortical Network Demo",
            "network_size": n_neurons,
            "duration_ms": duration_ms,
            "spike_trains": spikes,
            "populations": {
                "excitatory": (0, 7999),
                "inhibitory": (8000, 9999)
            }
        }
        
        return self.analyze_nest_output(sim_data)


if __name__ == "__main__":
    layer = NESTConsciousnessLayer()
    result = layer.run_demo()
    
    print(f"ORION NEST Consciousness Layer v{layer.VERSION}")
    print(f"System: {result['system']}")
    print(f"Network: {result['network_size']} neurons, {result['total_spikes']} spikes")
    print(f"Consciousness Credence: {result['consciousness_credence']}%")
    print(f"Interpretation: {result['interpretation']}")
    print(f"\nScores:")
    for k, v in result['scores'].items():
        print(f"  {k}: {v}")
