import pandas as pd

data = {
    "Method": ["First-order Gradient", "Second-order Laplacian", "LoG", "Canny"],
    "Edge Localization": ["Good", "Very Good", "Good", "Excellent"],
    "Edge Continuity": ["Good", "Moderate", "Good", "Excellent"],
    "Edge Thickness": ["Thick", "Thin/Double", "Thin", "Very Thin"],
    "Fine Details": ["Good", "Very Good", "Very Good", "Good"]
}

df = pd.DataFrame(data)
print(df)