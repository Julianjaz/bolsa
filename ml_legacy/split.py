
import pandas as pd
# Cargar el DataFrame final
df = pd.read_parquet("datos_finales_con_features.parquet")

train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

train_df.to_parquet("train_data.parquet", index=False)
test_df.to_parquet("test_data.parquet", index=False)