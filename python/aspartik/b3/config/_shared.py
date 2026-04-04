from typing import Literal

type CalculatorKind = Literal["cpu", "cuda", "metal"]
type SubstitutionModel = Literal["JC", "K80", "HKY", "GTR"]
type TreePrior = Literal["yule", "constant", "exponential"]
