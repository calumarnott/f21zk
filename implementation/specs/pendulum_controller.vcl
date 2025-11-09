include "pendulum_common.vcl"

u_max_cfg : Real
u_max_cfg = 10.0

@parameter setpoint : Real   -- from config; default 0.0

@network
ctrlNet3 : State3 -> Ctrl

applyCtrl3 : State2 -> Ctrl
applyCtrl3 x = ctrlNet3 ( normalise3 ( mkInput3 setpoint x ) )

@property
sat_ok : Bool
sat_ok = forall x . validState x =>
  abs (applyCtrl3 x ! u_idx) <= u_max_cfg

@parameter u_quiet : Real
@property
quiet_near_upright : Bool
quiet_near_upright = forall x .
  validState x and nearUpright2 x =>
    abs (applyCtrl3 x ! u_idx) <= u_quiet

@property
quiet_robust : Bool
quiet_robust = forall x d .
  validState x and nearUpright2 x and boundedByEps2 d and validState (x + d) =>
    abs (applyCtrl3 (x + d) ! u_idx) <= u_quiet

@parameter tau_min : Real
@property
corrective_sign_right : Bool
corrective_sign_right = forall x .
  validState x and nearUpright2 x and (x ! theta) > 0 =>
    applyCtrl3 x ! u_idx <= -tau_min

@property
corrective_sign_left : Bool
corrective_sign_left = forall x .
  validState x and nearUpright2 x and (x ! theta) < 0 =>
    applyCtrl3 x ! u_idx >=  tau_min
