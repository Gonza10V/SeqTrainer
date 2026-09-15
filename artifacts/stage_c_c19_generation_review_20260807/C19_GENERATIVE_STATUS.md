# C19 checkpoint generative status

- Decision: `SMOKE_FINITE_WITH_CALIBRATION_WARNING`
- Mode: `smoke`
- Checkpoint step: `26,250`
- Checkpoint SHA-256: `e08d96cab9808b45a8972e36f0f50fcf5754e012565c35f685b0d555f3da8b4f`
- All reported numeric values finite: `True`
- Prodigal: `completed`

## Generated-versus-held-out diagnostics

| Group | 6-mer JSD | aligned diversity ratio | GC absolute error | entropy absolute error | max homopolymer |
|---|---:|---:|---:|---:|---:|
| temperature_0.6 | 0.459920 | 0.4512 | 0.107747 | 0.026565 | 6.00 |
| temperature_0.8 | 0.441314 | 0.5031 | 0.088379 | 0.018573 | 6.00 |
| temperature_1 | 0.432238 | 0.4512 | 0.081380 | 0.009367 | 6.25 |

## Smoke calibration warnings

- temperature_0.6: aligned 6-mer diversity ratio 0.451 is outside the exploratory 0.80-1.20 review band
- temperature_0.6: GC absolute error 0.108 exceeds 0.05
- temperature_0.8: aligned 6-mer diversity ratio 0.503 is outside the exploratory 0.80-1.20 review band
- temperature_0.8: GC absolute error 0.088 exceeds 0.05
- temperature_1: aligned 6-mer diversity ratio 0.451 is outside the exploratory 0.80-1.20 review band
- temperature_1: GC absolute error 0.081 exceeds 0.05

## Next step

Set `RUN_FULL=True` and provide the c16 report to evaluate the frozen generation component.

## Interpretation limits

- A smoke result is operational evidence, not the frozen E25 generation gate.
- The generation component alone cannot authorize E100; held-out prediction and controlled-memory gates are also required.
- Generated ORFs and Prodigal calls do not establish biological function, expression, viability, fitness, or safety.
