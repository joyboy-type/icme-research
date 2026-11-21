"""One frozen diagnostic routing rule; no label inputs or fitted thresholds."""
import math

def route(baseline, subject, eligibility, foreground_contribution, background_contribution, epsilon=1e-7):
 result={'winner':baseline,'triggered':False,'status':'unchanged','reason':'default_B1'}
 if baseline==subject:
  result['reason']='heads_agree';return result
 if eligibility=='ineligible_scope':
  result['reason']='instruction_or_scene_out_of_scope';return result
 if eligibility!='eligible':
  result.update(status='unable',reason=eligibility);return result
 if baseline not in ('A','B') or subject not in ('A','B'):
  result.update(status='unable',reason='nonbinary_cached_winner');return result
 fg,bg=foreground_contribution,background_contribution
 if not all(math.isfinite(x) for x in [fg,bg]) or min(abs(fg),abs(bg))<=epsilon:
  result.update(status='unable',reason='nonfinite_or_negligible_evidence');return result
 direction=1 if baseline=='A' else -1
 if direction*fg < -epsilon and direction*bg > epsilon and abs(bg)>abs(fg)+epsilon:
  result.update(winner=subject,triggered=True,status='switched',reason='verified_source_support_complement_opposition')
 else:
  result['reason']='opposition_not_verified'
 return result
