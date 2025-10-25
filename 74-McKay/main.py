import numpy as np
from scipy import stats


def mckay_ci(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cv = std / mean

    # chi-square quantiles
    u1 = stats.chi2.ppf(1 - alpha / 2, n - 1)
    u2 = stats.chi2.ppf(alpha / 2, n - 1)

    # McKay confidence interval
    lcl = cv / np.sqrt((u1 / (n - 1)) * cv**2 + u1 / (n - 1))
    ucl = cv / np.sqrt((u2 / (n - 1)) * cv**2 + u2 / (n - 1))

    return lcl, ucl


# Model performance metrics (illustrative data)
acc_result = [0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.87, 0.86, 0.84, 0.88]

# Calculate CV and confidence interval
mean_acc = np.mean(acc_result)
std_acc = np.std(acc_result, ddof=1)
cv = std_acc / mean_acc
lcl, ucl = mckay_ci(acc_result)

# Print results
print("=" * 60)
print("Coefficient of Variation Analysis - McKay Method")
print("=" * 60)
print(f"Sample Size: {len(acc_result)}")
print(f"Mean: {mean_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")
print("-" * 60)
print(f"Coefficient of Variation (CV): {cv:.4f} ({cv*100:.2f}%)")
print(f"95% Confidence Interval: [{lcl:.4f}, {ucl:.4f}]")
print(f"Interval Width: {ucl-lcl:.4f}")
print("=" * 60)

# Interpretation
print("Interpretation:")
if cv < 0.05:
    print("    Low variability - Very stable")
elif cv < 0.1:
    print("    Moderate variability - Reasonably stable")
else:
    print("    High variability - Caution needed")
print(f"True CV is between {lcl:.4f} and {ucl:.4f} with 95% confidence")
