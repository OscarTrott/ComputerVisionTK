from ComputerVisionTools import Tools
from TaylorSeriesApproximation import TaylorSeriesApproximator

opticalFlow = False
taylorSeriesApproximation = True

if opticalFlow:
    runner = Tools()

    runner.start()

# ---------------------------
# Taylor series approximation
if taylorSeriesApproximation:
    runner = TaylorSeriesApproximator()

    runner.start()
