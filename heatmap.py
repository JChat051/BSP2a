import numpy as np
import matplotlib.pyplot as plt

d = np.load("features.npz", allow_pickle=True)
scores, names = d["learned_scores"], list(d["class_names"])
lt = d["learned_times"]
top = np.argsort(scores.mean(axis=0))[::-1][:10]
plt.figure(figsize=(10, 4))
extent = (float(lt[0]), float(lt[-1]), 0.0, float(len(top)))
plt.imshow(scores[:, top].T, aspect="auto", origin="lower",
extent=extent, cmap="magma")
plt.yticks(np.arange(len(top)) + 0.5, [names[i] for i in top])
plt.xlabel("time (s)"); plt.colorbar(label="score"); plt.show()