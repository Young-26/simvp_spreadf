from ionogram_manifest import IonogramManifestDataset
import matplotlib.pyplot as plt
import os

idx = 0

ds = IonogramManifestDataset(
    manifest_path=r"D:\BaiduNetdiskDownload\spreadf_sequence_builder\outputs\gao\manifests\gao_train.jsonl",
    image_mode="RGB",
    image_size=448,
)

sample = ds[idx]
meta = ds.samples[idx]   # 直接取 manifest 中这一条原始记录

x = sample["x"]   # [8, C, 448, 448]
y = sample["y"]   # [2, C, 448, 448]

# 这里改成你最终确定的 F 区边界
top, bottom = 186, 410
x_local = x[:, :, top:bottom, :]
y_local = y[:, :, top:bottom, :]

print("=" * 80)
print(f"sample idx: {idx}")

# 如果 __getitem__ 返回了这些字段，就打印
if "sequence_id" in sample:
    print("sequence_id:", sample["sequence_id"])
if "label" in sample:
    print("label:", sample["label"])
if "year" in sample:
    print("year:", sample["year"])
if "source" in sample:
    print("source:", sample["source"])
if "timestamps" in sample:
    print("timestamps:", sample["timestamps"])

print("\n--- manifest raw record keys ---")
print(list(meta.keys()))

# 打印输入/目标图片路径
if "input_paths" in meta:
    print("\n[input_paths]")
    for i, p in enumerate(meta["input_paths"]):
        print(f"  x[{i}] -> {p}")

if "target_paths" in meta:
    print("\n[target_paths]")
    for i, p in enumerate(meta["target_paths"]):
        print(f"  y[{i}] -> {p}")

# 如果想只看文件名，不看完整路径
if "input_paths" in meta:
    print("\n[input file names]")
    for i, p in enumerate(meta["input_paths"]):
        print(f"  x[{i}] -> {os.path.basename(p)}")

if "target_paths" in meta:
    print("\n[target file names]")
    for i, p in enumerate(meta["target_paths"]):
        print(f"  y[{i}] -> {os.path.basename(p)}")

print("\nshapes:")
print("x shape      :", tuple(x.shape))
print("x_local shape:", tuple(x_local.shape))
print("y shape      :", tuple(y.shape))
print("y_local shape:", tuple(y_local.shape))
print("=" * 80)

# 可视化第一帧输入和切出来的局部图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(x[0].permute(1, 2, 0).numpy())
plt.axhline(top, color="r")
plt.axhline(bottom - 1, color="r")
plt.title("global x[0]")

plt.subplot(1, 2, 2)
plt.imshow(x_local[0].permute(1, 2, 0).numpy())
plt.title("local x_local[0]")

plt.tight_layout()
plt.show()