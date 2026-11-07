"""Temporal serve eligibility driven by FSM evidence, never by missing balls."""
import math


class ServeGuard:
    def __init__(self,min_restart_seconds=3.0,enforce_restart_pause=False):
        if not math.isfinite(min_restart_seconds) or min_restart_seconds<0:
            raise ValueError('Invalid restart pause')
        self.min_restart_seconds=float(min_restart_seconds)
        self.enforce_restart_pause=bool(enforce_restart_pause)
        self.reset()

    def reset(self):
        self.state='';self.active=False;self.last_end=None;self.time=None
        self.start_permitted=False;self.start_reason=None

    def update(self,time,state):
        if not math.isfinite(time) or (self.time is not None and time<self.time):
            raise ValueError('ServeGuard requires chronological video time; reset on seek')
        previous=self.state;self.time=float(time);self.state=state or ''
        if self.state=='rally_end' and previous!='rally_end':
            self.last_end=self.time;self.active=False;self.start_permitted=False
        if self.state=='rally_active':self.active=True;self.start_permitted=False
        if self.state=='serving' and previous!='serving':
            self.start_reason=('serve_during_active_rally' if self.active else
                'serve_restart_too_soon' if self.remaining>0 else None)
            self.start_permitted=self.start_reason is None

    @property
    def remaining(self):
        if self.last_end is None or self.time is None:return 0.0
        return max(0.0,self.min_restart_seconds-(self.time-self.last_end))

    def reason(self,label):
        if label!='serve':return None
        if self.active:return 'serve_during_active_rally'
        if self.enforce_restart_pause and self.remaining>0:return 'serve_restart_too_soon'
        if self.state=='serving':
            if self.enforce_restart_pause and self.start_reason:return self.start_reason
            return None
        return 'serve_start_not_confirmed'

    def decision(self,label):
        reason=self.reason(label)
        return dict(raw_label=label,label=None if reason else label,suppression_reason=reason,
            warning_reason=('serve_restart_pause_uncertain' if label=='serve' and reason is None
                and (self.remaining>0 or self.start_reason=='serve_restart_too_soon') else None),
            state=self.state,time=self.time,restart_wait_seconds=self.remaining,
            min_restart_seconds=self.min_restart_seconds)
