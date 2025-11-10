--------------------------------------------------------------------------------
-- Inputs and outputs (same as example)

type Image = Tensor Real [28, 28]
type Label = Index 10

validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--------------------------------------------------------------------------------
-- Network (same)

@network
classifier : Image -> Tensor Real [10]

-- Stronger "advises": still argmax, PLUS cap all non-true classes by eta.
-- Minimal change to your original one-liner.
@parameter
eta : Real

advises : Image -> Label -> Bool
advises x i =
  forall j . j != i =>
    (classifier x ! i > classifier x ! j) and (classifier x ! j <= eta)

--------------------------------------------------------------------------------
-- Epsilon + ball (same idea)

@parameter
epsilon : Real

boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

--------------------------------------------------------------------------------
-- Robustness around a point (unchanged shape)

robustAround : Image -> Label -> Bool
robustAround image label = forall pertubation .
  let perturbedImage = image - pertubation in
  boundedByEpsilon pertubation and validImage perturbedImage =>
    advises perturbedImage label

--------------------------------------------------------------------------------
-- Dataset plumbing (same)

@parameter(infer=True)
n : Nat

@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)
