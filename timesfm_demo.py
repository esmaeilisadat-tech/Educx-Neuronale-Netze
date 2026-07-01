import timesfm
import numpy as np

print("Loading TimesFM 2.5 model...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

print("Compiling model with ForecastConfig...")
model.compile(
    timesfm.ForecastConfig(
        max_context=128,
        max_horizon=12,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
)

print("Running forecast...")
# One simple dummy time series
inputs = [np.linspace(0, 1, 100)]
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=inputs
)

print(f"Point Forecast Shape: {point_forecast.shape}")
if quantile_forecast is not None:
    print(f"Quantile Forecast Shape: {quantile_forecast.shape}")

print(f"Point Forecast Values:\n{point_forecast}")
