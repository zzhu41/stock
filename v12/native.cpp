// V12 isolated fork: only locked panic exit is new. No fast-math/future prices.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "schema.hpp"

extern "C" int simulate(const double* features, const double* scores, const int* orders,
                        const int* fear, int T, int A, const double* configs, int N,
                        int lo, int hi, double fee, int workers,
                        double* returns_out, int* holdings_out, double* summaries) {
    if (lo < 0 || hi >= T || lo > hi || fee < 0 || fee >= .5 || A > 63) return -1;
    const int D = hi-lo+1;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int n=0;n<N;n++) {
        const double* p=configs+n*NP;
        const int score_id=(int)p[P_SCORE];
        const uint64_t stock=(uint64_t)p[P_STOCK_MASK], glob=(uint64_t)p[P_GLOBAL_MASK];
        const uint64_t trade=stock|glob|(uint64_t(1)<<GOLD), safe=glob|(uint64_t(1)<<GOLD);
        auto in=[](uint64_t mask,int a){ return a>=0 && bool(mask & (uint64_t(1)<<a)); };
        auto f=[&](int t,int a,int field){ return features[(t*A+a)*NF+field]; };
        auto s=[&](int t,int a){ return scores[(score_id*T+t)*A+a]; };
        auto valid=[&](int t,int a){ return a>=0 && a!=CASH && f(t,a,F_VALID)>.5 && std::isfinite(f(t,a,F_CLOSE)); };
        int holding=-1, age=0, lock_until=-1, pending=-1, confirmations=0;
        bool bull_state=true, initialized=false, pending_bull=true;
        int regime_count=0, switches=0, entries=0, blocked=0, missing=0, crashes=0;
        std::vector<int> cooldown(A,-1);
        double nav=1., mark=0., peak=1., maxdd=0., held_peak=0., sum=0., sumsq=0.;
        for (int t=lo;t<=hi;t++) {
            const double before=nav;
            double current=holding>=0?f(t,holding,F_CLOSE):std::numeric_limits<double>::quiet_NaN();
            bool can_sell=holding<0 || std::isfinite(current);
            if (!can_sell) missing++;
            if (t>lo) {
                if (holding>=0 && can_sell) { nav*=current/mark; mark=current; }
                age++;
            }
            int q=t-(int)p[P_LAG], target=holding;
            bool panic=false, crash_trigger=false, confirming=false, locked_panic_exit=false;
            if (q>=0) {
                const int* order=orders+(score_id*T+q)*A;
                const int mom=(int)p[P_MOM], exit_mom=(int)p[P_EXIT_MOM], fast_mom=(int)p[P_FAST_MOM];
                int regime=(int)p[P_REGIME];
                bool bull=true;
                double distance=valid(q,BENCH)?f(q,BENCH,(int)p[P_MA]):1.;
                if (!std::isfinite(distance)) distance=1.;
                if (regime!=2 && regime!=3) {
                    if (regime==4) {
                        int eligible=0,votes=0;
                        for (int a=0;a<A;a++) if(in(stock,a) && valid(q,a) && std::isfinite(f(q,a,(int)p[P_MA]))) {
                            eligible++; if(f(q,a,(int)p[P_MA])>0) votes++;
                        }
                        bull=eligible?double(votes)/eligible>=p[P_BREADTH_THRESHOLD]:distance>0;
                    } else {
                        double threshold=initialized?(bull_state?-p[P_REGIME_HYST]:p[P_REGIME_HYST]):0.;
                        bull=distance>threshold;
                        if(regime==5) bull=bull && f(q,BENCH,F_MOM20)>0;
                    }
                    if(!initialized) {bull_state=bull;initialized=true;}
                    else if(bull!=bull_state && p[P_REGIME_CONFIRM]>1) {
                        if(regime_count && pending_bull==bull) regime_count++;
                        else {pending_bull=bull;regime_count=1;}
                        if(regime_count>=(int)p[P_REGIME_CONFIRM]) {bull_state=bull;regime_count=0;}
                        bull=bull_state;
                    } else {bull_state=bull;regime_count=0;}
                }
                bool hv=valid(q,holding) && holding!=CASH;
                if(hv) {
                    // For delayed signals only the already observed close updates the stop.
                    held_peak=std::max(held_peak,f(q,holding,F_CLOSE));
                    double threshold=p[P_PANIC];
                    if((int)p[P_PANIC_MODE]==1) threshold=std::min(.10,std::max(.02,threshold*f(q,holding,F_VOL20)));
                    panic=p[P_PANIC]>0 && f(q,holding,F_RET1)<=-threshold;
                    if(p[P_TRAIL_STOP]>0 && held_peak>0 && f(q,holding,F_CLOSE)/held_peak-1<=-p[P_TRAIL_STOP]) panic=true;
                }
                uint64_t comp=regime==3?trade:((bull || regime==1)?stock|glob:safe);
                double floor=bull?p[P_BULL_ENTRY]:std::max(p[P_BULL_ENTRY],p[P_BEAR_ENTRY]);
                target=-1;
                for(int k=0;k<A;k++) {
                    int a=order[k];
                    if(!in(comp,a) || !valid(q,a) || (panic&&a==holding) || t<cooldown[a]) continue;
                    double m=f(q,a,mom);
                    if(m<=floor || (m>p[P_OVERHEAT] && f(q,a,fast_mom)<=0)) continue;
                    target=a;break;
                }
                if(target<0 && bull && valid(q,GOLD) && f(q,GOLD,mom)>0 && !(panic&&holding==GOLD) && t>=cooldown[GOLD]) target=GOLD;
                if(target<0 && (int)p[P_FALLBACK_MODE]!=1) {
                    double best_vol=1e100;
                    for(int k=0;k<A;k++) {
                        int a=order[k];
                        if(!in(safe,a) || !valid(q,a) || (panic&&a==holding) || t<cooldown[a]) continue;
                        if((int)p[P_FALLBACK_MODE]==0) {target=a;break;}
                        if(f(q,a,F_VOL20)<best_vol){target=a;best_vol=f(q,a,F_VOL20);}
                    }
                    if(target>=0 && f(q,target,mom)<p[P_FALLBACK_FLOOR]) target=-1;
                }
                if(target<0) target=CASH;
                if(!panic && holding>=0 && target!=holding && hv) {
                    bool forced=(!bull && regime!=1 && in(stock,holding)) ||
                        f(q,holding,exit_mom)<=p[P_EXIT_FLOOR] ||
                        (f(q,holding,mom)>p[P_OVERHEAT] && f(q,holding,fast_mom)<=0);
                    if(!forced) {
                        bool keep=age<(int)p[P_MIN_HOLD];
                        double buffer=p[P_BUFFER];
                        if(valid(q,target)) {
                            if(target==GOLD && p[P_GOLD_BUFFER]>=0) buffer=p[P_GOLD_BUFFER];
                            else if(in(glob,target) && p[P_GLOBAL_BUFFER]>=0) buffer=p[P_GLOBAL_BUFFER];
                        }
                        if(p[P_TREND_BUFFER]>0 && f(q,holding,mom)>.10) buffer=std::max(buffer,p[P_TREND_BUFFER]);
                        switch((int)p[P_BUFFER_MODE]) {
                            case 0: keep=keep || (valid(q,target)?f(q,target,mom):0.)-f(q,holding,mom)<buffer;break;
                            case 1: keep=keep || (valid(q,target)?s(q,target):0.)-s(q,holding)<buffer;break;
                            case 2: keep=keep || (valid(q,target)?s(q,target):0.)-s(q,holding)<buffer*std::max(std::abs(s(q,holding)),1e-9);break;
                            case 3: {
                                int rank=0;
                                for(int k=0;k<A;k++) if(in(comp,order[k])&&valid(q,order[k])) {
                                    rank++;if(order[k]==holding){keep=keep||rank<=(int)p[P_RANK_KEEP];break;}
                                }
                            }
                        }
                        if(keep) target=holding;
                        else if((int)p[P_SWITCH_CONFIRM]>1) {
                            confirming=true;
                            if(pending==target) confirmations++; else {pending=target;confirmations=1;}
                            if(confirmations<(int)p[P_SWITCH_CONFIRM]) target=holding;
                        }
                    }
                }
                if(t<lock_until) {
                    // A lagged shock on the entry date predates the actual
                    // close fill. Only a later signal observation can exit.
                    locked_panic_exit=p[P_LOCKED_PANIC_EXIT]>.5 && panic &&
                        holding>=0 && holding!=CASH && age>(int)p[P_LAG];
                    target=locked_panic_exit?CASH:holding;
                    confirming=false;
                }
                else if((int)p[P_CRASH_MASK] && can_sell) {
                    bool guard=true;
                    const int g=(int)p[P_CRASH_GUARD];
                    const bool risk_held=holding>=0 && holding!=CASH;
                    if(risk_held && g==4) guard=false;
                    else if(risk_held && g!=0 && !hv) guard=false;
                    else if(hv) {
                        if(g==1) guard=f(q,holding,F_MOM20)<=0;
                        else if(g==3) guard=f(q,holding,F_RET1)<=-.04 || f(q,holding,F_MOM5)<=-.08;
                    }
                    if(guard) {
                        int candidate=-1;double best=1e100;
                        for(int k=0;k<A;k++) {
                            int a=order[k];
                            if(!in(trade,a)||a==holding||!valid(q,a)||(p[P_CRASH_STOCK_ONLY]>.5&&!in(stock,a))||t<cooldown[a]) continue;
                            if((int)p[P_CRASH_GUARD]==2&&hv&&s(q,a)<=s(q,holding)) continue;
                            int mask=(int)p[P_CRASH_MASK];double m5=f(q,a,F_MOM5),dist=f(q,a,F_MA250);
                            bool deep=(mask&1)&&m5<=p[P_DEEP_MOM]&&dist<-p[P_DEEP_BELOW];
                            bool qvix=(mask&2)&&fear[q]&&m5<=p[P_RELAXED_MOM]&&dist<-.20;
                            bool volume=(mask&4)&&f(q,a,F_VOLUME_RATIO)>=p[P_VOLUME_RATIO]&&m5<=p[P_RELAXED_MOM]&&dist<-p[P_VOLUME_BELOW];
                            if(!(deep||qvix||volume)) continue;
                            if((int)p[P_CRASH_PICK]==0){candidate=a;break;}
                            double criterion=(int)p[P_CRASH_PICK]==1?m5:f(q,a,F_VOL20);
                            if(criterion<best){best=criterion;candidate=a;}
                        }
                        if(candidate>=0) {
                            target=candidate;crash_trigger=true;confirming=false;
                        }
                    }
                }
            }
            if(!confirming){pending=-1;confirmations=0;}
            if(target>=0 && target!=holding) {
                double buy=f(t,target,F_CLOSE);
                if(!can_sell||!std::isfinite(buy)){blocked++;target=holding;}
                else {
                    if(panic && holding>=0 && p[P_PANIC_COOLDOWN]>0) cooldown[holding]=t+(int)p[P_PANIC_COOLDOWN]+1;
                    if(t>lo){nav*=1-2*fee;switches++;}
                    entries++;holding=target;mark=buy;age=0;held_peak=buy;
                    pending=-1;confirmations=0;
                    if(crash_trigger){lock_until=t+(int)p[P_CRASH_LOCK];crashes++;}
                    // A rejected order must retain the old lock. This branch
                    // is reached only after the actual sell and buy succeed.
                    if(locked_panic_exit) lock_until=-1;
                }
            } else if(pending==holding || (target==holding && (int)p[P_SWITCH_CONFIRM]<=1)) {pending=-1;confirmations=0;}
            peak=std::max(peak,nav);maxdd=std::min(maxdd,nav/peak-1.);
            double ret=nav/before-1.;sum+=ret;sumsq+=ret*ret;
            if(returns_out)returns_out[n*D+(t-lo)]=ret;
            if(holdings_out)holdings_out[n*D+(t-lo)]=holding;
        }
        double* z=summaries+n*10;
        z[0]=nav;z[1]=std::pow(nav,244./D)-1;z[2]=maxdd;z[3]=switches;z[4]=entries;
        z[5]=blocked;z[6]=missing;z[7]=crashes;z[8]=sum;z[9]=sumsq;
    }
    return 0;
}
