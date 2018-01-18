from bootstrap_evaluation import *
from combined_evaluation import  *
from dropout_evaluation import *

if __name__ == "__main__":
    dropout_osband_sin_evaluation()
    dropout_osband_nonlinear_evaluation()

    combined_osband_sin_evaluation(epochs=20000)
    combined_osband_nonlinear_evaluation(epochs=10000)

    bootstrap_osband_sin_evaluation(n_samples=50, n_heads=5, epochs=12000)
    bootstrap_osband_nonlinear_evaluation(n_heads=10, epochs=8000)
