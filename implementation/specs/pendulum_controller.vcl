-- Problem-space tensors
type State2 = Tensor Real [2]      -- [theta, theta_dot]
theta2     = 0
theta_dot2 = 1

type State = Tensor Real [3]      -- [theta, theta_dot, error]
theta     = 0
theta_dot = 1
err       = 2

type Ctrl   = Tensor Real [1]      -- torque u
u_idx = 0


-- Ranges / constants
pi : Real
pi = 3.1416

-- Valid state space

type unnormalisedState = Tensor Real [3]

minimumInputValues2 : State2
minimumInputValues2 = [ -pi, -10.0 ]
maximumInputValues2 : State2
maximumInputValues2 = [  pi,  10.0 ]

minimumInputValues : unnormalisedState
minimumInputValues = [ -pi, -10.0, -pi]
maximumInputValues : unnormalisedState
maximumInputValues = [  pi,  10.0,  pi]

validState2 : State2 -> Bool
validState2 x = forall i . minimumInputValues2 ! i <= x ! i <= maximumInputValues2 ! i

validState : unnormalisedState -> Bool
validState x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i

abs : Real -> Real
abs r = if r >= 0 then r else -r


-- === Normalisation params (from norms.json) ===
@parameter
mu_theta     : Real
@parameter
mu_theta_dot : Real
@parameter
sd_theta     : Real
@parameter
sd_theta_dot : Real
@parameter
mu_err       : Real
@parameter
sd_err       : Real

meanScalingValues : unnormalisedState
meanScalingValues = [ mu_theta, mu_theta_dot, mu_err ]

stdDevScalingValues : unnormalisedState
stdDevScalingValues = [ sd_theta, sd_theta_dot, sd_err ]

normalise : unnormalisedState -> State
normalise x = foreach i . ( x ! i - meanScalingValues ! i ) / ( stdDevScalingValues ! i )

-- Build 3-input vector from 2D state + setpoint
@parameter
setpoint : Real    -- e.g. 0.0

mkInput3 : State2 -> unnormalisedState
mkInput3 x = [ x ! theta2, x ! theta_dot2, (x ! theta2) - setpoint ]

-- === Network + properties ===
u_max : Real
u_max = 10.0
u_min : Real
u_min = -10.0

@network
ctrlNet : State -> Ctrl

normCtrlNet : unnormalisedState -> Ctrl
normCtrlNet x = ctrlNet ( normalise x )

applyCtrl3 : State2 -> Ctrl
applyCtrl3 x = normCtrlNet ( mkInput3 x )

-- A) Saturation respected globally -> for all valid states, u_min <= u <= u_max
sat_lower : State2 -> Bool
sat_lower x = u_min <= applyCtrl3 x ! u_idx

sat_upper : State2 -> Bool
sat_upper x = applyCtrl3 x ! u_idx <= u_max

@property
sat_ok : Bool
sat_ok = forall x . validState2 x => sat_lower x and sat_upper x

-- @property
-- sat_ok_lower : Bool
-- sat_ok_lower = forall x . validState2 x => sat_lower x

-- @property
-- sat_ok_upper : Bool
-- sat_ok_upper = forall x . validState2 x => sat_upper x

margin : Real
margin = 2.0

sat_given_margin : Real -> State2 -> Bool
sat_given_margin i x = 
    (margin * u_min) <= applyCtrl3 x ! u_idx <= (u_max * margin)

@property
sat_ok_with_margin : Bool
sat_ok_with_margin = forall x .
  validState2 x =>
    sat_given_margin margin x

-- B) If the pendulum if at extreme angles, the torque is in the correct direction
-- If theta >= 3*pi/4, then u <= 0
-- and if theta <= -3*pi/4, then u >= 0
@property
correctDirectionTorque : Bool
correctDirectionTorque = forall x .
  (validState2 x and (x ! theta2) >= (3.0 * pi / 4.0) => applyCtrl3 x ! u_idx <= 0) and
    (validState2 x and (x ! theta2) <= (-3.0 * pi / 4.0) => applyCtrl3 x ! u_idx >= 0)

-- C) Small control value when theta, theta_dot near zero
u_quiet : Real
u_quiet = 0.1

epsilon_theta : Real
epsilon_theta = 0.0001

@property
quiet_near_upright : Bool
quiet_near_upright = forall x .
  validState2 x and (abs (x ! theta2) <= epsilon_theta) and (abs (x ! theta_dot2) <= epsilon_theta) =>
    abs (applyCtrl3 x ! u_idx) <= u_quiet

-- C) lipschitz robustness 
-- Robustness radius
epsilon : Real
epsilon = 0.00001

lipschitz_constant : Real
lipschitz_constant = 15.0

-- # epsilon-bounded difference between two states
boundedByEps2 : State2 -> Bool
boundedByEps2 d = forall i . -epsilon <= d ! i <= epsilon

-- @network
-- ctrlNet2 : State -> Ctrl

-- applyCtrlCopy3 : State2 -> Ctrl
-- applyCtrlCopy3 x = ctrlNet2 ( normalise ( mkInput3 x ) )

-- torque_output_diff : State2 -> State2 -> Real
-- torque_output_diff x1 x2 = abs ( (applyCtrl3 x1 ! u_idx) - (applyCtrlCopy3 x2 ! u_idx) )
-- @property
-- lipschitz_robustness : Bool
-- lipschitz_robustness = forall x1 x2 .
--   validState2 x1 and validState2 x2 and boundedByEps2 ( x1 - x2 ) =>
--     torque_output_diff x1 x2 <= lipschitz_constant

@parameter(infer=True)
n : Nat

-- Dataset of *problem-space* states: [theta, theta_dot] (no error term here)
@dataset
pend_states : Vector State2 n

-- Precomputed reference controls at those states (1-dim each)
@dataset
u_ref : Vector Ctrl n

-- Lipschitz-like dataset robustness that uses only ONE network application:
-- Compare u(x+δ) to an *external* reference u_ref(x) precomputed offline.

@property
robust_dataset : Vector Bool n
robust_dataset = foreach i .
  forall d .
    let x0        = pend_states ! i in
    let x1        = x0 + d in
    let dtheta    = (x1 ! theta2)     - (x0 ! theta2) in
    let dthetadot = (x1 ! theta_dot2) - (x0 ! theta_dot2) in
    boundedByEps2 d
      and validState2 x1
      and validState (mkInput3 x1)   -- ensure all 3 NN inputs are bounded
      =>
      abs( applyCtrl3 x1 ! u_idx - (u_ref ! i ! u_idx) ) <= lipschitz_constant *( abs dtheta + abs dthetadot )