from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

class Server:
      def __init__(
        self,
        server_id: int,
        base_latency: float,
        noise_std: float,
        drift_amplitude: float,
        drift_period: int,
        shift_interval: int,
        rng: np.random.Generator,
        *,
        phase: Optional[float] = None,
        trend_slope: float = 0.0,
        trend_period: int = 10_000,
        shift_multipliers: Optional[List[float]] = None,
        min_latency: float = 1.0,
    ) -> None:
        if base_latency <= 0:
            raise ValueError("base_latency must be > 0")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if drift_period <= 0:
            raise ValueError("drift_period must be > 0")
        if shift_interval < 0:
            raise ValueError("shift_interval must be >= 0")
        if min_latency <= 0:
            raise ValueError("min_latency must be > 0")

        self.server_id = int(server_id)
        self.base_latency = float(base_latency)
        self.noise_std = float(noise_std)
        self.drift_amplitude = float(drift_amplitude)
        self.drift_period = int(drift_period)
        self.shift_interval = int(shift_interval)
        self.rng = rng

        self.phase = float(phase) if phase is not None else float(rng.uniform(0, 2 * math.pi))
        self.trend_slope = float(trend_slope)
        self.trend_period = int(trend_period)

        self.shift_multipliers = shift_multipliers or [0.75, 0.9, 1.0, 1.1, 1.25]
        if any(m <= 0 for m in self.shift_multipliers):
            raise ValueError("All shift_multipliers must be > 0")

        self.min_latency = float(min_latency)

        # Internal regime state
        self._regime_multiplier: float = 1.0
        self._last_shift_t: Optional[int] = None

    def _maybe_shift_regime(self, t: int) -> None:
        """Shift rejimini (multiplier) periyodik olarak değiştir."""
        if self.shift_interval <= 0:
            return
        if t <= 0:
            return
        if (t % self.shift_interval) != 0:
            return
        if self._last_shift_t == t:
            return  # aynı t için ikinci kez shift yapma

        # Multiplier'ın gerçekten değişmesine çalış (aynı çıkabilir; onu engelle)
        current = self._regime_multiplier
        if len(self.shift_multipliers) == 1:
            new = float(self.shift_multipliers[0])
        else:
            new = current
            # Çok nadir durumlarda aynı gelmesin diye birkaç kez dene
            for _ in range(10):
                candidate = float(self.rng.choice(self.shift_multipliers))
                if candidate != current:
                    new = candidate
                    break
            if new == current:
                # Hâlâ aynıysa, listeden farklı olanı seç
                for m in self.shift_multipliers:
                    if float(m) != current:
                        new = float(m)
                        break

        self._regime_multiplier = new
        self._last_shift_t = t

    def mean_latency(self, t: int) -> float:
        """Gürültüsüz (deterministik) ortalama latency."""
        self._maybe_shift_regime(t)

        drift = self.drift_amplitude * math.sin(2 * math.pi * t / self.drift_period + self.phase)
        trend = 0.0
        if self.trend_slope != 0.0 and self.trend_period > 0:
            trend = self.trend_slope * (t / self.trend_period)

        mean = (self.base_latency * self._regime_multiplier) + drift + trend
        return float(max(self.min_latency, mean))

    def get_latency(self, t: int) -> float:
        """t anındaki latency gözlemini (Gaussian noise ile) üret."""
        mu = self.mean_latency(t)
        if self.noise_std == 0.0:
            return mu
        sample = float(self.rng.normal(loc=mu, scale=self.noise_std))
        return float(max(self.min_latency, sample))


class SoftmaxLoadBalancer:
    """
    Softmax Action Selection (ana algoritma).
    """

    def __init__(
        self,
        n_servers: int,
        tau: float,
        rng: np.random.Generator,
        *,
        initial_q: float = 0.0,
    ) -> None:
        if n_servers <= 0:
            raise ValueError("n_servers must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        self.n_servers = int(n_servers)
        self.tau = float(tau)
        self.rng = rng

        self.q_values = np.full(self.n_servers, float(initial_q), dtype=np.float64)
        self.counts = np.zeros(self.n_servers, dtype=np.int64)

    def _softmax_stable(self, q_values: np.ndarray, tau: float) -> np.ndarray:
        """
        Numerik stabil softmax:
          m = max(Q)
          softmax_i = exp((Q_i - m)/tau) / sum_j exp((Q_j - m)/tau)
        """
        q = np.asarray(q_values, dtype=np.float64)
        m = float(np.max(q))
        scaled = (q - m) / float(tau)
        exps = np.exp(scaled)
        denom = float(np.sum(exps))

        if not np.isfinite(denom) or denom <= 0.0:
            # Aşırı uç bir durumda uniform'a düş
            return np.full_like(q, 1.0 / q.size, dtype=np.float64)

        return exps / denom

    def select_server(self) -> int:
        probs = self._softmax_stable(self.q_values, self.tau)
        return int(self.rng.choice(self.n_servers, p=probs))

    def update(self, server_idx: int, reward: float) -> None:
        i = int(server_idx)
        self.counts[i] += 1
        n = float(self.counts[i])  # n >= 1
        self.q_values[i] = self.q_values[i] + (1.0 / n) * (float(reward) - self.q_values[i])


class RandomLoadBalancer:
    """Rastgele sunucu seç."""

    def __init__(self, n_servers: int, rng: np.random.Generator) -> None:
        if n_servers <= 0:
            raise ValueError("n_servers must be > 0")
        self.n_servers = int(n_servers)
        self.rng = rng

    def select_server(self) -> int:
        return int(self.rng.integers(low=0, high=self.n_servers))


class RoundRobinLoadBalancer:
    """Sırayla sunucu seç (deterministik)."""

    def __init__(self, n_servers: int) -> None:
        if n_servers <= 0:
            raise ValueError("n_servers must be > 0")
        self.n_servers = int(n_servers)
        self._next = 0

    def select_server(self) -> int:
        idx = self._next
        self._next = (self._next + 1) % self.n_servers
        return idx


@dataclass(frozen=True)
class SimulationConfig:
    n_servers: int = 5
    n_requests: int = 5000
    tau: float = 0.5
    shift_interval: int = 200
    seed: int = 7

    # Server latency model parametreleri
    base_latency_range: Tuple[float, float] = (40.0, 120.0)
    noise_std_range: Tuple[float, float] = (2.0, 12.0)
    drift_amp_range: Tuple[float, float] = (5.0, 30.0)
    drift_period_range: Tuple[int, int] = (300, 1400)
    trend_slope_range: Tuple[float, float] = (-15.0, 15.0)
    trend_period: int = 10_000
    min_latency: float = 1.0


class Simulation:
    """
    Tüm algoritmaları aynı non-stationary ortamda çalıştırır,
    metrikleri toplar, grafikleri üretir ve runtime analizi yazdırır.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.servers: List[Server] = []
        self.latency_trace: Optional[np.ndarray] = None  # shape: (N, K)
        self.results: Dict[str, Dict[str, object]] = {}

    def _create_servers(self) -> List[Server]:
        cfg = self.config
        base_rng = np.random.default_rng(cfg.seed)

        servers: List[Server] = []
        for i in range(cfg.n_servers):
            # Sunucu bazında ayrı RNG (reproducible)
            rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))

            srv = Server(
                server_id=i,
                base_latency=float(rng.uniform(*cfg.base_latency_range)),
                noise_std=float(rng.uniform(*cfg.noise_std_range)),
                drift_amplitude=float(rng.uniform(*cfg.drift_amp_range)),
                drift_period=int(rng.integers(cfg.drift_period_range[0], cfg.drift_period_range[1] + 1)),
                shift_interval=cfg.shift_interval,
                rng=rng,
                trend_slope=float(rng.uniform(*cfg.trend_slope_range)),
                trend_period=cfg.trend_period,
                min_latency=cfg.min_latency,
            )
            servers.append(srv)

        return servers

    def _generate_latency_trace(self) -> np.ndarray:
        """Tüm sunucular için latency gözlemlerini önceden üret (adil karşılaştırma)."""
        cfg = self.config
        N, K = cfg.n_requests, cfg.n_servers

        # Her run() çağrısında sunucuları resetle (re-run deterministic olsun)
        self.servers = self._create_servers()

        trace = np.zeros((N, K), dtype=np.float64)
        for t in range(N):
            for k, srv in enumerate(self.servers):
                trace[t, k] = srv.get_latency(t)
        return trace

    def run(self, n_requests: Optional[int] = None) -> None:
        cfg = self.config
        if n_requests is not None:
            cfg = SimulationConfig(**{**cfg.__dict__, "n_requests": int(n_requests)})
            self.config = cfg  # update

        N, K = cfg.n_requests, cfg.n_servers

        self.latency_trace = self._generate_latency_trace()

        # Politikalar için ayrı RNG'ler (selection randomness deterministik)
        rng_softmax = np.random.default_rng(cfg.seed + 101)
        rng_random = np.random.default_rng(cfg.seed + 202)

        policies = {
            "Random": RandomLoadBalancer(K, rng_random),
            "RoundRobin": RoundRobinLoadBalancer(K),
            "Softmax": SoftmaxLoadBalancer(K, cfg.tau, rng_softmax),
        }

        results: Dict[str, Dict[str, object]] = {}
        for name, policy in policies.items():
            selection_counts = np.zeros(K, dtype=np.int64)
            total_latency = 0.0
            total_reward = 0.0
            cumulative_rewards = np.zeros(N, dtype=np.float64)

            # Runtime: sadece select_server maliyeti
            select_time_total = 0.0

            q_history = np.zeros((N, K), dtype=np.float64) if name == "Softmax" else None

            for t in range(N):
                t0 = time.perf_counter()
                server_idx = policy.select_server()
                select_time_total += (time.perf_counter() - t0)

                latency = float(self.latency_trace[t, server_idx])
                reward = -latency

                selection_counts[server_idx] += 1
                total_latency += latency
                total_reward += reward
                cumulative_rewards[t] = total_reward

                if isinstance(policy, SoftmaxLoadBalancer):
                    policy.update(server_idx, reward)
                    q_history[t, :] = policy.q_values

            results[name] = {
                "cumulative_reward": float(total_reward),
                "average_latency": float(total_latency / N),
                "selection_counts": selection_counts,
                "selection_frequency": selection_counts / float(N),
                "cumulative_rewards_over_time": cumulative_rewards,
                "avg_select_time_sec": float(select_time_total / N),
                "q_history": q_history,
            }

        self.results = results
        self._print_summary()

    def _print_summary(self) -> None:
        cfg = self.config
        print("\n" + "=" * 80)
        print("CLIENT-SIDE LOAD BALANCER SIMULATION SUMMARY")
        print("=" * 80)
        print(f"N (requests) : {cfg.n_requests}")
        print(f"K (servers)  : {cfg.n_servers}")
        print(f"Softmax τ    : {cfg.tau}")
        print(f"Shift interval (steps): {cfg.shift_interval}")
        print(f"Seed         : {cfg.seed}")
        print("-" * 80)

        # En iyi: daha yüksek cumulative reward (= daha düşük toplam latency)
        ranking = sorted(self.results.items(), key=lambda kv: kv[1]["cumulative_reward"], reverse=True)

        for name, r in ranking:
            cum_reward = r["cumulative_reward"]
            avg_lat = r["average_latency"]
            freq = r["selection_frequency"]
            avg_sel_us = r["avg_select_time_sec"] * 1e6

            print(f"[{name}]")
            print(f"  cumulative_reward : {cum_reward: .3f}")
            print(f"  average_latency   : {avg_lat: .3f} ms")
            print(f"  avg_select_time   : {avg_sel_us: .2f} µs/choice")
            print("  selection_freq    : " + ", ".join([f"S{i}={freq[i]:.3f}" for i in range(cfg.n_servers)]))
            print("-" * 80)

    def plot_results(self, *, show: bool = True, save_dir: Optional[str] = None) -> None:
        if self.latency_trace is None or not self.results:
            raise RuntimeError("Önce run() çağırmalısın.")

        cfg = self.config
        N, K = cfg.n_requests, cfg.n_servers
        t = np.arange(N)

        outdir: Optional[Path] = None
        if save_dir is not None:
            outdir = Path(save_dir)
            outdir.mkdir(parents=True, exist_ok=True)

        # a) Kümülatif Ödül Karşılaştırması
        fig1 = plt.figure(figsize=(11, 5))
        for name, r in self.results.items():
            plt.plot(t, r["cumulative_rewards_over_time"], label=name)
        plt.title("Kümülatif Ödül (reward = -latency)")
        plt.xlabel("İstek adımı t")
        plt.ylabel("Kümülatif ödül")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig1.tight_layout()
        if outdir is not None:
            fig1.savefig(outdir / "a_cumulative_reward_comparison.png", dpi=160)

        # b) Softmax Sunucu Seçim Frekansları
        fig2 = plt.figure(figsize=(9, 4))
        softmax_freq = self.results["Softmax"]["selection_frequency"]
        plt.bar(np.arange(K), softmax_freq)
        plt.title("Softmax Sunucu Seçim Frekansları")
        plt.xlabel("Sunucu index")
        plt.ylabel("Seçilme oranı")
        plt.xticks(np.arange(K))
        plt.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        if outdir is not None:
            fig2.savefig(outdir / "b_softmax_selection_frequency.png", dpi=160)

        # c) Sunucu Latency Değişimi (gerçek gözlemler)
        fig3 = plt.figure(figsize=(11, 5))
        for k in range(K):
            plt.plot(t, self.latency_trace[:, k], label=f"Server {k}")
        plt.title("Sunucu Latency Değişimi (non-stationary, observed)")
        plt.xlabel("İstek adımı t")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig3.tight_layout()
        if outdir is not None:
            fig3.savefig(outdir / "c_server_latency_over_time.png", dpi=160)

        # d) Q-Değerleri Evrimi (Softmax)
        q_hist = self.results["Softmax"]["q_history"]
        if isinstance(q_hist, np.ndarray):
            fig4 = plt.figure(figsize=(11, 5))
            for k in range(K):
                plt.plot(t, q_hist[:, k], label=f"Q(Server {k})")
            plt.title("Softmax Q-Değerleri Evrimi (tahmini reward)")
            plt.xlabel("İstek adımı t")
            plt.ylabel("Q değeri (expected reward)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig4.tight_layout()
            if outdir is not None:
                fig4.savefig(outdir / "d_softmax_q_values_over_time.png", dpi=160)

        if show:
            plt.show()
        else:
            plt.close("all")

    def print_runtime_analysis(self) -> None:
        if not self.results:
            raise RuntimeError("Önce run() çağırmalısın.")

        print("\n" + "=" * 80)
        print("RUNTIME ANALYSIS")
        print("=" * 80)
        for name, r in self.results.items():
            avg_sel_us = r["avg_select_time_sec"] * 1e6
            print(f"{name:10s} avg selection time: {avg_sel_us: .2f} µs/choice")

        print("\nBig-O complexity:")
        print("  Softmax selection: O(K)   # K sunucu için olasılık hesaplama")
        print("  Q update         : O(1)   # tek sunucunun incremental mean güncellemesi")
        print("  Total (N requests): O(N·K)")
        print("=" * 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Softmax client-side load balancer simulation")
    p.add_argument("--servers", type=int, default=5, help="Sunucu sayısı (K)")
    p.add_argument("--requests", type=int, default=5000, help="İstek sayısı (N)")
    p.add_argument("--tau", type=float, default=0.5, help="Softmax temperature (tau > 0)")
    p.add_argument("--shift-interval", type=int, default=200, help="Rejim değişim periyodu (adım)")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--no-plot", action="store_true", help="Grafik gösterme")
    p.add_argument("--save-plots", type=str, default=None, help="Grafikleri bu klasöre kaydet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        n_servers=args.servers,
        n_requests=args.requests,
        tau=args.tau,
        shift_interval=args.shift_interval,
        seed=args.seed,
    )

    sim = Simulation(cfg)
    sim.run()
    sim.print_runtime_analysis()

    if not args.no_plot or args.save_plots is not None:
        sim.plot_results(show=not args.no_plot, save_dir=args.save_plots)


if __name__ == "__main__":
    main()