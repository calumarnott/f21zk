-- Problem-space tensors
type State2 = Tensor Real [2]      -- [theta, theta_dot]
type State3 = Tensor Real [3]      -- [theta, theta_dot, error]  (if your net uses error as an explicit input)
type Ctrl   = Tensor Real [1]      -- torque u

-- Indices
theta     = 0
theta_dot = 1
err_idx   = 2
u_idx     = 0 -- output torque

-- === Values inferred from your config (default.yaml) ===
-- Ranges from verify.clip.{lo,hi} and physical constants
pi : Real
pi = 3.1416

validState : State2 -> Bool
validState x = -pi <= x ! theta <= pi and -10.0 <= x ! theta_dot <= 10.0

validState3 : State3 -> Bool
validState3 x = -pi <= x ! theta <= pi and -10.0 <= x ! theta_dot <= 10.0 and -pi <= x ! err_idx <= pi

-- Controller limits from controller.theta.{u_min,u_max}
abs : Real -> Real
abs r = if r >= 0 then r else -r

-- Small-signal neighbourhood (tunable; keep as parameters)
@parameter theta0 : Real   -- e.g. 0.10
@parameter omega0 : Real   -- e.g. 0.50

nearUpright2 : State2 -> Bool
nearUpright2 x = abs (x ! theta) <= theta0 and abs (x ! theta_dot) <= omega0

nearUpright3 : State3 -> Bool
nearUpright3 x = abs (x ! theta) <= theta0 and abs (x ! theta_dot) <= omega0 and abs (x ! err_idx) <= omega0

-- === Embedding gap: training-time normalisation (from norms.json) ===
@parameter mu_theta     : Real
@parameter mu_theta_dot : Real
@parameter sd_theta     : Real
@parameter sd_theta_dot : Real

normalise2 : State2 -> State2
normalise2 x = foreach i .
  let mu = if i == theta then mu_theta else mu_theta_dot in
  let sd = if i == theta then sd_theta else sd_theta_dot in
  (x ! i - mu) / sd

-- If your model has 3 inputs [theta, theta_dot, error], we also need mean/std for error.
@parameter mu_err : Real
@parameter sd_err : Real

mkInput3 : Real -> State2 -> State3
mkInput3 setpoint x = [ x ! theta, x ! theta_dot, (x ! theta) - setpoint ]

normalise3 : State3 -> State3
normalise3 x = foreach i .
  let mu = if i == theta then mu_theta else (if i == theta_dot then mu_theta_dot else mu_err) in
  let sd = if i == theta then sd_theta else (if i == theta_dot then sd_theta_dot else sd_err) in
  (x ! i - mu) / sd

-- Robustness radius from verify.eps in your config
@parameter epsilon : Real

boundedByEps2 : State2 -> Bool
boundedByEps2 d = forall i . -epsilon <= d ! i <= epsilon

boundedByEps3 : State3 -> Bool
boundedByEps3 d = forall i . -epsilon <= d ! i <= epsilon
