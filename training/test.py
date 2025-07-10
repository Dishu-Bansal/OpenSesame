import requests
import time
import numpy as np

latencies = []

for _ in range(100):
    start = time.time()
    r = requests.post("http://localhost:5000/predict")  # no data needed
    end = time.time()
    latencies.append((end - start) * 1000)  # convert to ms

latencies = np.array(latencies)
print(f"Median: {np.percentile(latencies, 50):.2f} ms")
print(f"P95: {np.percentile(latencies, 95):.2f} ms")
