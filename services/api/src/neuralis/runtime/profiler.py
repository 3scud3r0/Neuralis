from pathlib import Path

import torch


class ResearchProfiler:
    """A scoped real PyTorch profiler for trusted user applications."""

    def __init__(self, output: Path, wait=1, warmup=1, active=3, include_cuda=False, record_shapes=True):
        if wait < 0 or warmup < 0 or not 1 <= active <= 20:
            raise ValueError("Profiler capture must contain 1 to 20 active steps")
        if include_cuda and not torch.cuda.is_available():
            raise ValueError("CUDA profiling requested on a runtime without available CUDA")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if include_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.traces = []
        self.profiler = torch.profiler.profile(activities=activities,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            record_shapes=record_shapes, profile_memory=True, with_stack=False, on_trace_ready=self._save_trace)

    def _save_trace(self, profiler):
        destination = self.output / f"trace-{len(self.traces) + 1:03d}.json"
        profiler.export_chrome_trace(str(destination))
        self.traces.append(str(destination))

    def step(self):
        self.profiler.step()

    def report(self, limit=50):
        if not 1 <= limit <= 500:
            raise ValueError("Profiler report budget must be 1 to 500 operators")
        averages = list(self.profiler.key_averages())
        averages.sort(key=lambda event: event.self_cpu_time_total, reverse=True)
        return {"operators": [{"name": event.key, "calls": event.count, "self_cpu_time_us": event.self_cpu_time_total,
                               "cpu_time_us": event.cpu_time_total, "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                               "cpu_memory_bytes": event.cpu_memory_usage} for event in averages[:limit]],
                "traces": self.traces, "units": "microseconds and bytes", "source": "torch.profiler"}

    def __enter__(self):
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self.profiler.__exit__(exc_type, exc, traceback)
