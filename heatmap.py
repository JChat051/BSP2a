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

for n in ["Music", "Speech", "Drum", "Singing", "Silence", "Percussion"]:
    s = scores[:, names.index(n)]
    print(f"{n:12s} mean {s.mean():.2f} std {s.std():.2f}")

idx = names.index("Music") 
music = np.interp(t, lt, scores[:, idx]) 

def smooth(x, alpha=0.1): 
    out = np.zeros_like(x); v = 0.0
    for i, target in enumerate(x):
        v += (target - v) * alpha
        out[i] = v
    return out

music_s = smooth(music)