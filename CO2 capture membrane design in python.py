import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Constants for the equations
R = 8.314  # Gas constant, J/(mol*K)

# Molecular weights and atomic radii (from the table provided)
molecular_data = {
    "Water": {"M": 18.015, "radius": 2.8},
    "Carbon": {"M": 12.011, "radius": 1.7},
    "Hydrogen": {"M": 1.008, "radius": 1.1},
    "Oxygen": {"M": 15.999, "radius": 1.52},
    "Nitrogen": {"M": 14.007, "radius": 1.55},
    "Sulfur": {"M": 32.065, "radius": 0.88},
    "Chlorine": {"M": 35.453, "radius": 1.75},
    "CO2": {"M": 44.01, "radius": 1.65},
    "H2": {"M": 2.016, "radius": 1.445},
    "O2": {"M": 31.998, "radius": 1.73},
    "N2": {"M": 28.014, "radius": 1.82},
    "Cl2": {"M": 70.906, "radius": 1.6},
}

# Generate synthetic data
n_samples = 1000
data = []

for _ in range(n_samples):
    # Random values for the parameters (within realistic ranges)
    epsilon = np.random.uniform(0.3, 0.7)  # porosity
    k = np.random.uniform(1, 5)  # Kozeny constant
    sigma = np.random.uniform(50, 150)  # specific surface area
    Pm = np.random.uniform(1, 10)  # mean pressure
    mu = np.random.uniform(0.1, 0.3)  # viscosity of gas (Poise)
    delta_P = np.random.uniform(0.1, 2)  # pressure difference
    L = np.random.uniform(0.1, 1)  # pore length

    # Loop through each compound
    for compound, params in molecular_data.items():
        D = np.random.uniform(params["radius"] + 0.1, 5)  # pore diameter (must be larger than molecule)
        M = params["M"]  # Molecular weight

        # Calculate the Darcy constant (KD)
        KD = (epsilon**3) / (k * sigma**2 * (1 - epsilon)**2)

        # Calculate T (temperature) and Jg (gas flux)
        T = (Pm * D) / (R * epsilon)  # Simplified relation for temperature
        Jg = (KD / mu) * Pm * (M / (R * T)) * (delta_P / L)

        # Append the data for each compound
        data.append([compound, epsilon, k, sigma, Pm, mu, delta_P, L, D, M, T, Jg])

# Convert to DataFrame for analysis
df = pd.DataFrame(data, columns=[
    "Compound", "Porosity (ε)", "Kozeny constant (k)", "Specific Surface (σ)", 
    "Mean Pressure (Pm)", "Viscosity (μ)", "Pressure Difference (P1-P2)", 
    "Pore Length (L)", "Pore Diameter (D)", "Molecular Weight (M)", "Temperature (T)", "Gas Flux (Jg)"
])

# Perform linear regression for the prediction
X = df[["Porosity (ε)", "Kozeny constant (k)", "Specific Surface (σ)", 
        "Mean Pressure (Pm)", "Viscosity (μ)", "Pressure Difference (P1-P2)", 
        "Pore Length (L)", "Pore Diameter (D)", "Molecular Weight (M)"]]
y = df["Gas Flux (Jg)"]

# Initialize the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict Jg values
df["Predicted Gas Flux (Jg)"] = model.predict(X)

# Filter compounds where Jg is close to 0 for Carbon and CO2
target_compounds = ["Carbon", "CO2"]
df_filtered = df[df["Compound"].isin(target_compounds)]

# Average parameters for compounds excluding Carbon and CO2
df_non_target = df[~df["Compound"].isin(target_compounds)]
average_params = df_non_target.drop(columns=["Compound", "Gas Flux (Jg)", "Predicted Gas Flux (Jg)"]).mean()

# Impression Plots

# 1. **Bubble Plot for Multiple Parameters**
plt.figure(figsize=(12, 8))
for compound in df["Compound"].unique():
    subset = df[df["Compound"] == compound]
    plt.scatter(subset["Pore Diameter (D)"], subset["Gas Flux (Jg)"], 
                s=subset["Viscosity (μ)"] * 1000, label=compound, alpha=0.6)
plt.xlabel("Pore Diameter (D) [m]")
plt.ylabel("Gas Flux (Jg) [kg m^-2 s^-1]")
plt.title("Gas Flux vs Pore Diameter (Bubble size: Viscosity μ)")
plt.legend()
plt.show()

# 2. **Violin Plot for Distribution of Gas Flux (Jg)**
plt.figure(figsize=(12, 8))
sns.violinplot(x="Compound", y="Gas Flux (Jg)", data=df, scale="width", inner="quartile")
plt.xticks(rotation=45)
plt.xlabel("Compound")
plt.ylabel("Gas Flux (Jg) [kg m^-2 s^-1]")
plt.title("Distribution of Gas Flux for Different Compounds")
plt.show()

# 3. **3D Scatter Plot for Key Parameters**
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for compound in df["Compound"].unique():
    subset = df[df["Compound"] == compound]
    ax.scatter(subset["Porosity (ε)"], subset["Pore Diameter (D)"], subset["Gas Flux (Jg)"], label=compound, alpha=0.6)
ax.set_xlabel("Porosity (ε)")
ax.set_ylabel("Pore Diameter (D)")
ax.set_zlabel("Gas Flux (Jg)")
ax.set_title("3D Scatter: Porosity, Pore Diameter, and Gas Flux")
ax.legend()
plt.show()

# 4. **Histogram for Frequency of Gas Flux**
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x="Gas Flux (Jg)", hue="Compound", bins=20, kde=True, palette="tab10")
plt.xlabel("Gas Flux (Jg) [kg m^-2 s^-1]")
plt.ylabel("Frequency")
plt.title("Histogram of Gas Flux Across Compounds")
plt.show()

# 5. **Scatter Plot of Temperature vs Gas Flux**
plt.figure(figsize=(12, 8))
sns.scatterplot(x="Temperature (T)", y="Gas Flux (Jg)", hue="Compound", data=df, palette="viridis", alpha=0.6)
plt.xlabel("Temperature (T) [K]")
plt.ylabel("Gas Flux (Jg) [kg m^-2 s^-1]")
plt.title("Gas Flux vs Temperature for Different Compounds")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Output the average parameters for non-carbon and non-CO2 compounds
print("Average parameters for compounds excluding Carbon and CO2:")
print(average_params)

# Output the determined parameters for individual compounds
print("\nDetermined parameters for individual compounds:")
compound_groups = df.groupby("Compound")
for compound, group in compound_groups:
    print(f"\nCompound: {compound}")
    print(group.mean())
