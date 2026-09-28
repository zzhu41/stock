"""Pure one-date adaptation of the fixed R2 WLS25/smooth4/prior-max model."""
import math
import numpy as np

from v12.reference import Policy,Portfolio
from v10_live.policy import Snapshot,_date,_history_prefix,_public_indicators,_channels
from .constants import ASSETS,CASH,BENCHMARK,CANDIDATE_ID,CANDIDATE_HASH,STRATEGY_ID
from .profile import load_profile as verified_profile

LEGACY_START = "2014-01-02"


def _raw25(closes):
    seg=closes[-25:];xs=list(range(25));weights=[i+1 for i in xs];total=sum(weights)
    mx=sum(weights[i]*xs[i] for i in xs)/total
    my=sum(weights[i]*seg[i] for i in xs)/total
    xy=sum(weights[i]*(xs[i]-mx)*(seg[i]-my) for i in xs)
    xx=sum(weights[i]*(xs[i]-mx)**2 for i in xs)
    rets=[closes[i]/closes[i-1]-1. for i in range(len(closes)-20,len(closes))]
    mean=sum(rets)/len(rets)
    sigma=(sum((r-mean)**2 for r in rets)/len(rets))**.5
    slope=(xy/xx)/my if xx>0 and my>0 else 0.
    return slope*250/sigma if sigma>0 else 0.


def _legacy_vol(closes,window):
    rets=[closes[i]/closes[i-1]-1. for i in range(len(closes)-window,len(closes))]
    mean=sum(rets)/len(rets)
    return (sum((r-mean)**2 for r in rets)/len(rets))**.5


def _conv_vol(closes,window):
    prices=np.asarray(closes[-window-1:],dtype=np.float64)
    returns=prices[1:]/prices[:-1]-1.
    weights=np.ones(window)/window
    mean=np.convolve(returns,weights,mode="valid")[0]
    square=np.convolve(returns*returns,weights,mode="valid")[0]
    return float(np.sqrt(max(0.,square-mean*mean)))


def latest_features(rows,signal_date,code=None):
    """Only trailing observations enter; n counts real own quotes, not dates."""
    n=len(rows)
    out=dict(close=float("nan"),valid=0.,score=0.,bars=n,ret1=float("nan"),mom5=float("nan"),
        mom20=float("nan"),mom60=float("nan"),vol20=float("nan"),ma180=float("nan"),
        ma250=float("nan"),volume_ratio=float("nan"),prior_vol20=float("nan"),prior_vol60=float("nan"))
    if not rows or rows[-1][0]!=signal_date:return out
    out["close"]=rows[-1][1]
    if n<270:return out
    # All windows fit inside this tail, but maturity uses the full own count.
    closes=[float(r[1]) for r in rows[-270:]]
    raw=[_raw25(closes[:-lag] if lag else closes) for lag in (3,2,1,0)]
    score=float(np.convolve(np.asarray(raw),np.ones(4)/4,mode="valid")[0])
    previous=closes[:-1]
    # Frozen VOL20 was overwritten with the legacy list/sum calculation only
    # on mature risk rows from the study start. Preserve that exact boundary.
    prior20=(_legacy_vol(previous,20) if code!=CASH and n-1>=270 and rows[-2][0]>=LEGACY_START
             else _conv_vol(previous,20))
    prior60=_conv_vol(previous,60)
    ma180=float(np.convolve(np.asarray(closes[-180:]),np.ones(180)/180,mode="valid")[0])
    average_volume=sum(r[2] for r in rows[-21:-1])/20.
    out.update(valid=float(code!=CASH),score=score if code!=CASH else 0.,
        ret1=closes[-1]/closes[-2]-1.,mom5=closes[-1]/closes[-6]-1.,
        mom20=closes[-1]/closes[-21]-1.,mom60=closes[-1]/closes[-61]-1.,
        vol20=max(prior20,prior60),prior_vol20=prior20,prior_vol60=prior60,
        ma180=closes[-1]/ma180-1.,ma250=closes[-1]/(sum(closes[-250:])/250)-1.,
        volume_ratio=rows[-1][2]/average_volume if average_volume>0 else 0.)
    return out


def snapshot(histories,calendar,signal_date):
    _date(signal_date);dates=[_date(d) for d in calendar]
    if dates!=sorted(set(dates)) or signal_date not in dates:raise ValueError("V12-R2 requires an observed benchmark date")
    dates=[d for d in dates if d<=signal_date]
    if set(ASSETS)-set(histories):raise ValueError("V12-R2 missing original-universe TR histories")
    indicators,prior_dates={},{}
    for code in ASSETS:
        rows=_history_prefix(histories[code],signal_date,code)
        indicators[code]=latest_features(rows,signal_date,code)
        prior_dates[code]=rows[-2][0] if len(rows)>1 and rows[-1][0]==signal_date else None
    if not math.isfinite(indicators[BENCHMARK]["close"]):raise ValueError("V12-R2 benchmark has no same-day quote")
    data=Snapshot(indicators,len(dates)-1,False)
    data.prior_quote_dates=prior_dates
    return data,dates


def decide(histories,calendar,signal_date,state=None):
    profile=verified_profile();config=profile["config"]
    state=dict(state or {})
    data,dates=snapshot(histories,calendar,signal_date);day=len(dates)-1
    held=state.get("holding") or None;entry=state.get("entry_date")
    if held is not None and held not in ASSETS:raise ValueError("V12-R2 unknown independent holding")
    if state.get("last_date") and state["last_date"]>signal_date:raise ValueError("V12-R2 backdated signal")
    if held and not entry:raise ValueError("V12-R2 requires actual entry date")
    if entry and (_date(entry)>signal_date or entry not in dates):raise ValueError("V12-R2 entry date not in benchmark calendar")
    trigger,locked_code=state.get("crash_trigger_date"),state.get("crash_code")
    if bool(trigger)!=bool(locked_code):raise ValueError("V12-R2 crash date/code must be stored together")
    lock_until=-1
    if trigger:
        if locked_code!=held or trigger!=entry or _date(trigger) not in dates:
            raise ValueError("V12-R2 crash lock does not match actual entry")
        lock_until=dates.index(trigger)+config["crash_lock"]
    age=day-dates.index(entry) if entry else 0;active=day<lock_until
    account=Portfolio(holding=held,age=age,lock_until=lock_until)
    model=Policy(data,config)
    can_sell=held is None or math.isfinite(data.price(day,held))
    intended,panic,requested_crash=model.intent(day,account,can_sell)
    can_buy=intended is not None and math.isfinite(data.price(day,intended))
    executable=bool(can_sell and can_buy);target=intended if executable else held
    crash=bool(requested_crash and executable and target!=held)
    next_trigger,next_code=(trigger,locked_code) if active else (None,None)
    if crash:next_trigger,next_code=signal_date,target
    channels=_channels(data,day,intended,config) if requested_crash else []
    risk=data.value(day,held,"vol20") if held else float("nan")
    threshold=min(.10,max(.02,config["panic"]*risk)) if math.isfinite(risk) else None
    if not executable:reason="当前持仓或目标缺当日报价，本次没有有效成交建议"
    elif active:reason="抄底锁仓：实际入场%s，第5个后续交易日恢复决策"%trigger
    elif crash:reason="抄底(%s)，实际虚拟入场后保留5日锁仓"%"∪".join(channels)
    else:
        prefix="波动急跌退出（阈值%.2f%%）；"%(100*threshold) if panic and threshold is not None else ""
        reason=prefix+("继续持有" if target==held else "首次建仓" if held is None else "调仓")+"；WLS25四报价平滑 / MA180 / 健康轮动最短2日"
    return dict(strategy_id=STRATEGY_ID,candidate_id=CANDIDATE_ID,candidate_hash=CANDIDATE_HASH,
        signal_date=signal_date,target=target,previous_holding=held,intended_target=intended,
        reason=reason,panic=bool(panic),crash=crash,crash_requested=bool(requested_crash),
        executable=executable,crash_trigger_date=next_trigger,crash_code=next_code,
        lock_active=active or crash,lock_remaining_sessions=5 if crash else max(0,lock_until-day),
        selected_by="explicit_user_designation",research_primary=None,research_qualified=False,
        order_submission=False,diagnostics=dict(bull=bool(model.regime.current),ranking=data.ranked(day),
            indicators=_public_indicators(data),prior_quote_dates=data.prior_quote_dates,
            risk_context="prior_max20_60",risk_multiple=1.4,panic_threshold=threshold,
            actual_holding_age=age,healthy_min_hold=2,previous_holding=held,
            qvix_in_use=False,crash_channels=channels,cold_start=held is None,
            feature_clock="Four own-quote WLS25 mean; risk sigma from strictly previous own quote, no second lag"))
