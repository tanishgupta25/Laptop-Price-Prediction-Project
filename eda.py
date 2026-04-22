"""
eda.py
------
Exploratory Data Analysis for the Laptop Price Prediction project.
Saves visualisation plots to ./plots/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="darkgrid", palette="mako")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("laptop_augmented.csv", encoding="utf-8")
print(f"Dataset shape: {df.shape}")
print("\nMissing values:\n", df.isnull().sum())


# ── Feature engineering (mirrors train_model.py) ──────────────────────────────
def parse_ram(ram_str):
    try:
        return int(str(ram_str).replace("GB", "").strip())
    except Exception:
        return np.nan


def parse_weight(w_str):
    try:
        return float(str(w_str).replace("kg", "").strip())
    except Exception:
        return np.nan


def extract_cpu_brand(cpu_str):
    cpu_str = str(cpu_str)
    if "Intel" in cpu_str:
        return "Intel"
    elif "AMD" in cpu_str:
        return "AMD"
    return "Other"


def extract_cpu_ghz(cpu_str):
    try:
        tokens = str(cpu_str).split()
        for t in tokens:
            if "GHz" in t:
                return float(t.replace("GHz", ""))
    except Exception:
        pass
    return 2.0


def extract_ssd(mem_str):
    ssd = 0
    parts = str(mem_str).upper().split("+")
    for p in parts:
        nums = [int(x) for x in p.split() if x.isdigit()]
        if nums and ("SSD" in p or "FLASH" in p or "NVME" in p):
            size = nums[0]
            if "TB" in p:
                size *= 1024
            ssd += size
    return ssd


def extract_hdd(mem_str):
    hdd = 0
    parts = str(mem_str).upper().split("+")
    for p in parts:
        nums = [int(x) for x in p.split() if x.isdigit()]
        if nums and "HDD" in p:
            size = nums[0]
            if "TB" in p:
                size *= 1024
            hdd += size
    return hdd


df["RAM_GB"] = df["Ram"].apply(parse_ram)
df["Weight_kg"] = df["Weight"].apply(parse_weight)
df["CPU_Brand"] = df["Cpu"].apply(extract_cpu_brand)
df["CPU_GHz"] = df["Cpu"].apply(extract_cpu_ghz)
df["SSD_GB"] = df["Memory"].apply(extract_ssd)
df["HDD_GB"] = df["Memory"].apply(extract_hdd)

# ── 1. Price Distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Laptop Price Distribution", fontsize=16, fontweight="bold")

axes[0].hist(df["Price_euros"], bins=50, color="#5E81F4", edgecolor="white", alpha=0.9)
axes[0].set_xlabel("Price (€)", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Histogram")

axes[1].hist(np.log1p(df["Price_euros"]), bins=50, color="#81F45E", edgecolor="white", alpha=0.9)
axes[1].set_xlabel("log(Price + 1)", fontsize=12)
axes[1].set_title("Log-transformed Histogram")

plt.tight_layout()
plt.savefig("plots/price_distribution.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/price_distribution.png")


# ── 2. Price by Company (Boxplot) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby("Company")["Price_euros"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Company", y="Price_euros", order=order,
            palette="mako", ax=ax)
ax.set_title("Price Distribution by Company", fontsize=16, fontweight="bold")
ax.set_xlabel("Company", fontsize=12)
ax.set_ylabel("Price (€)", fontsize=12)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("plots/price_by_company.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/price_by_company.png")


# ── 3. Price by Type ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
order2 = df.groupby("TypeName")["Price_euros"].median().sort_values(ascending=False).index
sns.violinplot(data=df, x="TypeName", y="Price_euros", order=order2,
               palette="crest", inner="quartile", ax=ax)
ax.set_title("Price Distribution by Laptop Type", fontsize=16, fontweight="bold")
ax.set_xlabel("Type", fontsize=12)
ax.set_ylabel("Price (€)", fontsize=12)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("plots/price_by_type.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/price_by_type.png")


# ── 4. RAM vs Price ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="RAM_GB", y="Price_euros",
            palette="flare", ax=ax)
ax.set_title("RAM vs Price", fontsize=16, fontweight="bold")
ax.set_xlabel("RAM (GB)", fontsize=12)
ax.set_ylabel("Price (€)", fontsize=12)
plt.tight_layout()
plt.savefig("plots/ram_vs_price.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/ram_vs_price.png")


# ── 5. Feature vs Price scatter grid ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Key Features vs Price", fontsize=16, fontweight="bold")

for ax, (col, label) in zip(axes, [
    ("RAM_GB", "RAM (GB)"), ("CPU_GHz", "CPU Speed (GHz)"), ("SSD_GB", "SSD (GB)")
]):
    sub = df[[col, "Price_euros"]].dropna()
    sub = sub[np.isfinite(sub[col]) & np.isfinite(sub["Price_euros"])]
    ax.scatter(sub[col], sub["Price_euros"], alpha=0.3, s=15, color="#5E81F4")
    if len(sub) > 1 and sub[col].std() > 0:
        m, b = np.polyfit(sub[col], sub["Price_euros"], 1)
        x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="#F4815E", linewidth=2)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Price (€)", fontsize=11)
    ax.set_title(f"{label} vs Price")

plt.tight_layout()
plt.savefig("plots/feature_vs_price.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/feature_vs_price.png")


# ── 6. Correlation Heatmap ────────────────────────────────────────────────────
numeric_cols = ["RAM_GB", "CPU_GHz", "SSD_GB", "HDD_GB",
                "Weight_kg", "Inches", "Price_euros"]
corr = df[numeric_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="mako",
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("✔ Saved: plots/correlation_heatmap.png")

print("\n✅  EDA complete — all plots saved to ./plots/")
