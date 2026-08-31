import sys, time, warnings, yaml, numpy as np
warnings.filterwarnings("ignore")
import coremltools as ct
cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
chunk = cfg["audio"]["chunk_size"]; sr = cfg["audio"]["sample_rate"]
spec = np.random.randn(1, 2050, 801, 2).astype(np.float32) * 0.1
for units, label in ((ct.ComputeUnit.ALL, "ALL (ANE+GPU+CPU)"),
                     (ct.ComputeUnit.CPU_AND_GPU, "CPU+GPU")):
    try:
        t0 = time.perf_counter()
        m = ct.models.MLModel("mel_band_roformer.mlpackage", compute_units=units)
        ni = list(m.get_spec().description.input)[0].name
        load = time.perf_counter() - t0
        m.predict({ni: spec})
        t0 = time.perf_counter()
        for _ in range(3): m.predict({ni: spec})
        per = (time.perf_counter() - t0) / 3
        print(f"{label:20s} load={load:5.1f}s  chunk={per:6.2f}s  "
              f"= {chunk/sr/per:5.2f}x realtime", flush=True)
    except Exception as e:
        print(f"{label:20s} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
