--------------------------------------------------------------------------------
-- Full specification of the ACAS XU networks

-- Taken from Appendix VI of "Reluplex: An Efficient SMT Solver for Verifying
-- Deep Neural Networks" at https://arxiv.org/pdf/1702.01135.pdf

-- Comments describing the properties are taken directly from the text.

--------------------------------------------------------------------------------
-- Utilities

-- The value of the constant `pi`.
pi = 3.141592

--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a tensor of 5 real numbers.

type Input = Tensor Real [5]

-- Next we add meaningful names for the indices.
-- The fact that all tensor types come annotated with their size means that it
-- is impossible to mess up indexing into vectors, e.g. if you changed
-- `distanceToIntruder = 0` to `distanceToIntruder = 5` the specification would
-- fail to type-check.

distanceToIntruder = 0   -- measured in metres
angleToIntruder    = 1   -- measured in radians
intruderHeading    = 2   -- measured in radians
speed              = 3   -- measured in metres/second
intruderSpeed      = 4   -- measured in meters/second

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are also a tensor of 5 reals. Each one representing the score
-- for the 5 available courses of action.

type Output = Tensor Real [5]

-- Again we define meaningful names for the indices into output vectors.

clearOfConflict = 0
weakLeft        = 1
weakRight       = 2
strongLeft      = 3
strongRight     = 4

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
acasXu : Input -> Output

--------------------------------------------------------------------------------
-- Normalisation

-- As is common in machine learning, the network operates over
-- normalised values, rather than values in the problem space
-- (e.g. using standard units like m/s).
-- This is an issue for us, as we would like to write our specification in
-- terms of the problem space values .
-- Therefore before applying the network, we first have to normalise
-- the values in the problem space.

-- For clarity, we therefore define a new type synonym
-- for unnormalised input vectors which are in the problem space.
type UnnormalisedInput = Tensor Real [5]

-- Next we define the minimum and maximum values that each input can take.
-- These correspond to the range of the inputs that the network is designed
-- to work over.
minimumInputValues : UnnormalisedInput
minimumInputValues = [0, -pi, -pi, 0, 0]

maximumInputValues : UnnormalisedInput
maximumInputValues = [60261.0, pi, pi, 1200.0, 1200.0]

-- We can therefore define a simple predicate saying whether a given input
-- vector is in the right range.
validInput : UnnormalisedInput -> Bool
validInput x = forall i .
  minimumInputValues ! i <= x ! i <= maximumInputValues ! i

-- Then the mean values that will be used to scale the inputs.
meanScalingValues : UnnormalisedInput
meanScalingValues = [19791.091, 0.0, 0.0, 650.0, 600.0]

-- We can now define the normalisation function that takes an input vector and
-- returns the unnormalised version.
normalise : UnnormalisedInput -> Input
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / (maximumInputValues ! i - minimumInputValues ! i)

-- Using this we can define a new function that first normalises the input
-- vector and then applies the neural network.
normAcasXu : UnnormalisedInput -> Output
normAcasXu x = acasXu (normalise x)

-- A constraint that says the network chooses output `i` when given the
-- input `x`. We must necessarily provide a finite index that is less than 5
-- (i.e. of type Index 5). The `a ! b` operator lookups index `b` in vector `a`.
advises : Index 5 -> UnnormalisedInput -> Bool
advises i x = forall j . i != j => normAcasXu x ! i < normAcasXu x ! j


--------------------------------------------------------------------------------
-- Property 3

-- If the intruder is directly ahead and is moving towards the
-- ownship, the score for COC will not be minimal.

-- Tested on: all networks except N_{1,7}, N_{1,8}, and N_{1,9}.

directlyAhead : UnnormalisedInput -> Bool
directlyAhead x =
  1500  <= x ! distanceToIntruder <= 1800 and
  -0.06 <= x ! angleToIntruder    <= 0.06

movingTowards : UnnormalisedInput -> Bool
movingTowards x =
  x ! intruderHeading >= 3.10  and
  x ! speed           >= 980   and
  x ! intruderSpeed   >= 960

@property
property3 : Bool
property3 = forall x .
  validInput x and directlyAhead x and movingTowards x =>
  not (advises clearOfConflict x)


--------------------------------------------------------------------------------
-- Property 4 (φ4)
-- If the intruder is directly ahead and is moving away from the ownship
-- but at a lower speed than the ownship, the score for COC will not be minimal.
-- Tested on: all networks except N_{1,7}, N_{1,8}, N_{1,9}.

movingAwayLowerSpeed : UnnormalisedInput -> Bool
movingAwayLowerSpeed x =
  x ! intruderHeading == 0.0 and
  x ! speed           >= 1000 and
  700 <= x ! intruderSpeed <= 800

@property
property4 : Bool
property4 = forall x .
  validInput x and directlyAhead x and movingAwayLowerSpeed x =>
  not (advises clearOfConflict x)


--------------------------------------------------------------------------------
-- Property 5 (φ5)
-- If the intruder is near and approaching from the left, the network advises
-- “strong right”.
-- Tested on: N_{1,1}.

nearApproachingFromLeft : UnnormalisedInput -> Bool
nearApproachingFromLeft x =
  250 <= x ! distanceToIntruder <= 400 and
  0.2 <= x ! angleToIntruder    <= 0.4 and
  -pi <= x ! intruderHeading    <= -pi + 0.005 and
  100 <= x ! speed              <= 400 and
  0   <= x ! intruderSpeed      <= 400

@property
property5 : Bool
property5 = forall x .
  validInput x and nearApproachingFromLeft x =>
  advises strongRight x


--------------------------------------------------------------------------------
-- Property 6 (φ6)
-- If the intruder is sufficiently far away, the network advises COC.
-- Tested on: N_{1,1}.

sufficientlyFar : UnnormalisedInput -> Bool
sufficientlyFar x =
  12000 <= x ! distanceToIntruder <= 62000 and
  ( (0.7 <= x ! angleToIntruder <= pi) or
    (-pi <= x ! angleToIntruder <= -0.7) ) and
  -pi <= x ! intruderHeading <= -pi + 0.005 and
  100 <= x ! speed <= 1200 and
  0   <= x ! intruderSpeed <= 1200

@property
property6 : Bool
property6 = forall x .
  validInput x and sufficientlyFar x =>
  advises clearOfConflict x


--------------------------------------------------------------------------------
-- Property 7 (φ7)
-- If vertical separation is large, the network will never advise a strong turn.
-- (Large vertical separation is encoded by selecting N_{1,9}.)
-- Tested on: N_{1,9}.

withinOperatingRange : UnnormalisedInput -> Bool
withinOperatingRange x =
  0 <= x ! distanceToIntruder <= 60760 and
  -pi <= x ! angleToIntruder  <= pi and
  -pi <= x ! intruderHeading  <= pi and
  100 <= x ! speed            <= 1200 and
  0   <= x ! intruderSpeed    <= 1200

@property
property7 : Bool
property7 = forall x .
  validInput x and withinOperatingRange x =>
  (not (advises strongRight x)) and (not (advises strongLeft x))


--------------------------------------------------------------------------------
-- Property 8 (φ8)
-- For a large vertical separation and a previous “weak left” advisory, the
-- network will either output COC or continue advising “weak left”.
-- (Previous advisory = weak left is encoded by selecting N_{2,9}.)
-- Tested on: N_{2,9}.

largeVertPrevWeakLeft : UnnormalisedInput -> Bool
largeVertPrevWeakLeft x =
  0 <= x ! distanceToIntruder <= 60760 and
  -pi <= x ! angleToIntruder <= (-0.75 * pi) and
  -0.1 <= x ! intruderHeading <= 0.1 and
  600 <= x ! speed <= 1200 and
  600 <= x ! intruderSpeed <= 1200

@property
property8 : Bool
property8 = forall x .
  validInput x and largeVertPrevWeakLeft x =>
  (advises clearOfConflict x) or (advises weakLeft x)


--------------------------------------------------------------------------------
-- Property 9 (φ9)
-- Even if the previous advisory was “weak right”, a nearby intruder should cause
-- a “strong left” advisory.
-- (Previous advisory = weak right is encoded by selecting N_{3,·}; tested on N_{3,3}.)

nearbyIntruder_weakRightToStrongLeft : UnnormalisedInput -> Bool
nearbyIntruder_weakRightToStrongLeft x =
  2000 <= x ! distanceToIntruder <= 7000 and
  -0.4 <= x ! angleToIntruder    <= -0.14 and
  -pi  <= x ! intruderHeading    <= -pi + 0.01 and
  100  <= x ! speed              <= 150 and
  0    <= x ! intruderSpeed      <= 150

@property
property9 : Bool
property9 = forall x .
  validInput x and nearbyIntruder_weakRightToStrongLeft x =>
  advises strongLeft x


--------------------------------------------------------------------------------
-- Property 10 (φ10)
-- For a far away intruder, the network advises COC.
-- (Tested on N_{4,5}.)

farAway_phi10 : UnnormalisedInput -> Bool
farAway_phi10 x =
  36000 <= x ! distanceToIntruder <= 60760 and
  0.7   <= x ! angleToIntruder    <= pi and
  -pi   <= x ! intruderHeading    <= -pi + 0.01 and
  900   <= x ! speed              <= 1200 and
  600   <= x ! intruderSpeed      <= 1200

@property
property10 : Bool
property10 = forall x .
  validInput x and farAway_phi10 x =>
  advises clearOfConflict x


--------------------------------------------------------------------------------
-- Helpers for φ1–φ2 (threshold / maximality)

-- When we need to talk about "maximal" score (worst action), mirror of `advises`.
maximises : Index 5 -> UnnormalisedInput -> Bool
maximises i x = forall j . i != j => normAcasXu x ! i > normAcasXu x ! j

-- Common input guard for “distant intruder, ownship much faster”.
distantSlowIntruder : UnnormalisedInput -> Bool
distantSlowIntruder x =
  x ! distanceToIntruder >= 55947.691 and
  x ! speed              >= 1145 and
  x ! intruderSpeed      <= 60


--------------------------------------------------------------------------------
-- Property 1 (φ1)
-- If the intruder is distant and significantly slower, the COC score is ≤ 1500.
-- (Tested on all 45 networks.)

@property
property1 : Bool
property1 = forall x .
  validInput x and distantSlowIntruder x =>
  normAcasXu x ! clearOfConflict <= 1500.0


--------------------------------------------------------------------------------
-- Property 2 (φ2)
-- Under the same distant/slow setting, the COC score is never the maximal score.
-- (Tested on N_{x,y} for all x ≥ 2 and all y.)

@property
property2 : Bool
property2 = forall x .
  validInput x and distantSlowIntruder x =>
  not (maximises clearOfConflict x)
