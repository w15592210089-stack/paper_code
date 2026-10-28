import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 分布滞后模型
df = pd.read_csv(r'FIFE_NAME', encoding='utf-8')

# The last 54 columns are the 9 stages x 6 metrics each
output_cols = df.columns[-54:]
num_stages = 9
metrics_per_stage = 6

# Extract metrics for each stage into a list of DataFrames
stages = []
for i in range(num_stages):
    stage_cols = output_cols[i * metrics_per_stage: (i + 1) * metrics_per_stage]
    stage_df = df[stage_cols].copy()
    stage_df.columns = ['Rough', 'Height', 'Verticality1', 'Verticality2', 'Verticality3', 'Verticality4']
    stages.append(stage_df)

# Concatenate all stages' metrics for global PCA
all_metrics = pd.concat(stages, ignore_index=True)

# Filter out rows with non-finite values
all_metrics = all_metrics.replace([np.inf, -np.inf], np.nan).dropna()

# Standardize the data
scaler = StandardScaler()
scaled_metrics = scaler.fit_transform(all_metrics)

# Perform PCA
# 原代码中 PCA 部分替换成：
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=1, random_state=42)
fa.fit(scaled_metrics)
# ============ 因子方差贡献 / 共同度 ============
loadings = fa.components_          # shape: (n_components, n_features)
n_features = loadings.shape[1]
n_components = loadings.shape[0]

# 1) 各因子 SS loadings（方差贡献）
ss_loadings = (loadings ** 2).sum(axis=1)
ss_total = ss_loadings.sum()
ss_ratio = ss_loadings / ss_total          # 占共同方差的比例
ss_ratio_of_p = ss_loadings / n_features   # 占总标准化方差的比例（≈ 方差解释率）

# 2) 各变量共同度（communality）
communalities = (loadings ** 2).sum(axis=0)

# 3) 独特方差（uniqueness）
uniqueness = 1 - communalities

# ---- 输出 ----
print("\n========== Factor Variance Contribution ==========")
print(f"Number of factors: {n_components}")
print(f"Number of variables: {n_features}")

print("\n[Factor SS Loadings]")
for j in range(n_components):
    print(f"  Factor {j+1}: SS = {ss_loadings[j]:.3f} | "
          f"{ss_ratio[j]*100:.2f}% of common variance | "
          f"{ss_ratio_of_p[j]*100:.2f}% of total variance")
print(f"  Total common variance explained: {ss_total:.3f} "
      f"({ss_total/n_features*100:.2f}% of total variance)")

print("\n[Communalities per Variable]")
print(f"  {'Variable':<15}{'h²':>8}{'Uniqueness':>12}")
for name, h2, u in zip(all_metrics.columns, communalities, uniqueness):
    print(f"  {name:<15}{h2:>8.3f}{u:>12.3f}")

print(f"\n  Mean communality: {communalities.mean():.3f}")
print(f"  Min / Max communality: {communalities.min():.3f} / {communalities.max():.3f}")
# 载荷（factor loadings）
print("Factor Loadings:")
for metric, loading in zip(all_metrics.columns, fa.components_[0]):
    print(f"  - {metric}: {loading:.3f}")

# 方差解释（没有explained_variance_ratio_，但可以用得分方差近似）
pc1_stages = []
for stage in stages:
    stage_clean = stage.replace([np.inf, -np.inf], np.nan).dropna()
    scaled_stage = scaler.transform(stage_clean)
    scores = fa.transform(scaled_stage)[:, 0]
    pc1_stages.append(pd.Series(scores, name=f'FA1_Stage_{len(pc1_stages)+1}'))

# Assuming all experiments have the same number after cleaning, concat to DataFrame
pc1_df = pd.concat(pc1_stages, axis=1).dropna()  # Ensure no NaNs


# Function to fit distributed lag model
def fit_lag_model(target_stage):
    y = pc1_df.iloc[:, target_stage - 1]  # 0-indexed
    X = pc1_df.iloc[:, :target_stage - 1]
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()
    return model


# Fit models for stages 2 to 9
models = {}
for t in range(2, 10):
    models[t] = fit_lag_model(t)

# Print results in table format
print("\nDistributed Lag Models Results:")
for t in range(2, 10):
    model = models[t]
    print(f"\nTarget Stage: {t}")
    print("| Lag | Coefficient | p-value | 95% CI |")
    print("|-----|-------------|---------|--------|")
    for lag in range(1, t):
        coef = model.params[lag]
        pval = model.pvalues[lag]
        ci_low, ci_high = model.conf_int().iloc[lag]
        print(f"| {lag} | {coef:.3f} | {pval:.3f} | ({ci_low:.3f}, {ci_high:.3f}) |")
    print(f"R²: {model.rsquared:.3f}")

# 创建所有滞后系数的热力图
all_coeffs = []
target_stages = []

for t in range(2, 10):
    model = models[t]
    coeffs = []
    for lag in range(1, 9):  # 最多8个滞后
        if lag < t:
            coeff = model.params.get(lag, np.nan)
            pval = model.pvalues.get(lag, 1.0)  # 默认p值=1（不显著）
            # 标记显著性的系数（p<0.05）
            coeffs.append(coeff if pval < 0.05 else np.nan)
        else:
            coeffs.append(np.nan)
    all_coeffs.append(coeffs)
    target_stages.append(f'Stage {t}')

all_coeffs = np.array(all_coeffs)

# 创建热力图
plt.figure(figsize=(12, 8))
im = plt.imshow(all_coeffs, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, label='Coefficient Value')
plt.xlabel('Lag (Prior Stage)')
plt.ylabel('Target Stage')
plt.title('Lag Coefficients Heatmap (Significant Only, p<0.05)', fontsize=14)

# 设置刻度
plt.xticks(range(8), [f'Lag {i+1}' for i in range(8)])
plt.yticks(range(len(target_stages)), target_stages)

# 在热力图上添加数值
for i in range(len(target_stages)):
    for j in range(8):
        if not np.isnan(all_coeffs[i, j]):
            plt.text(j, i, f'{all_coeffs[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(all_coeffs[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.show()
# # 创建子图，显示所有阶段的系数变化
# fig, axes = plt.subplots(3, 3, figsize=(15, 12))
# axes = axes.flatten()
#
# for idx, t in enumerate(range(2, 10)):
#     model = models[t]
#     lags = range(1, t)
#     coefs = model.params[1:].values
#     pvals = model.pvalues[1:].values
#
#     # 显著和不显著的颜色区分
#     colors = ['red' if pval < 0.05 else 'lightgray' for pval in pvals]
#
#     ax = axes[idx]
#     bars = ax.bar(lags, coefs, color=colors, edgecolor='black')
#
#     # 添加误差线（置信区间）
#     ci = model.conf_int().iloc[1:]
#     errors = (ci.iloc[:, 1] - ci.iloc[:, 0]) / 2
#     ax.errorbar(lags, coefs, yerr=errors, fmt='none', color='black', capsize=3)
#
#     ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#     ax.set_xlabel('Lag')
#     ax.set_ylabel('Coefficient')
#     ax.set_title(f'Target Stage {t} (R²={model.rsquared:.3f})')
#     ax.set_xticks(lags)
#     ax.grid(True, alpha=0.3)
#
#     # 在显著柱子上方添加星号
#     for lag, pval in zip(lags, pvals):
#         if pval < 0.05:
#             ax.text(lag, coefs[lag - 1] + (0.02 if coefs[lag - 1] >= 0 else -0.02),
#                     '*', ha='center', fontsize=12, fontweight='bold')
#
# # 隐藏最后一个空的子图
# for i in range(len(range(2, 10)), len(axes)):
#     axes[i].set_visible(False)
#
# plt.suptitle('Lag Coefficients for All Target Stages (* indicates p<0.05)', fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()
# # 可视化PCA载荷
# plt.figure(figsize=(10, 6))
# metrics = all_metrics.columns
# loadings = pca.components_[0]
#
# bars = plt.bar(metrics, loadings, color='skyblue', edgecolor='black')
#
# # 添加数值标签
# for bar, loading in zip(bars, loadings):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
#             f'{loading:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
#
# plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
# plt.xlabel('Metrics')
# plt.ylabel('Loading on PC1')
# plt.title(f'PCA Loadings on First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
# plt.grid(True, alpha=0.3, axis='y')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# # 计算每个滞后期的累计影响（平均绝对系数）
# max_lags = 8
# cumulative_effects = np.zeros(max_lags)
#
# for t in range(2, 10):
#     model = models[t]
#     for lag in range(1, min(t, max_lags + 1)):
#         if lag in model.params and model.pvalues[lag] < 0.05:
#             cumulative_effects[lag-1] += abs(model.params[lag])
#
# # 标准化
# cumulative_effects = cumulative_effects / (len(range(2, 10)) - 1)
#
# plt.figure(figsize=(10, 6))
# lags = range(1, max_lags + 1)
# bars = plt.bar(lags, cumulative_effects, color='orange', edgecolor='black', alpha=0.7)
#
# plt.xlabel('Lag (Prior Stage)')
# plt.ylabel('Average Absolute Coefficient (Significant Only)')
# plt.title('Cumulative Influence of Prior Stages Across All Target Stages')
# plt.xticks(lags)
# plt.grid(True, alpha=0.3, axis='y')
#
# # 添加数值标签
# for bar, effect in zip(bars, cumulative_effects):
#     plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
#             f'{effect:.3f}', ha='center', va='bottom', fontsize=9)
#
# plt.tight_layout()
# plt.show()