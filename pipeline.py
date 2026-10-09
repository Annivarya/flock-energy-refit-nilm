"""Auditable REFIT cleaning, washing-machine EDA and offline NILM experiment."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports'
DATA = ROOT / 'data'
SEED = 42
# Exact washing-machine labels only; washer-dryers are not interchangeable targets.
WM = {1:[5], 2:[2], 3:[6], 4:[4,5], 5:[3], 6:[2], 7:[5], 8:[4], 9:[3],
      10:[5], 11:[3], 13:[3], 15:[3], 16:[5], 17:[4], 18:[5], 19:[2], 20:[4], 21:[3]}
TRAIN_HOMES = [1, 2, 5, 6, 9]
TEST_HOME = 7
CONTEXT = 30


def dump(obj, name):
    (REPORT / name).write_text(json.dumps(obj, indent=2, default=str))


def fill_short_gaps(series, limit=2):
    """Interpolate only complete, bounded NA runs of at most limit minutes."""
    missing = series.isna()
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform('sum')
    interpolated = series.interpolate(method='time', limit_area='inside')
    return series.where(~(missing & (lengths <= limit)), interpolated)


def minute_frame(df, cols, *, clean_raw=False, fill=True):
    """UTC epoch is canonical. Stable last-record wins for exact duplicate epochs."""
    df = df.copy()
    epoch = pd.to_numeric(df.Unix, errors='coerce')
    df.index = pd.to_datetime(epoch, unit='s', utc=True, errors='coerce')
    df = df.loc[df.index.notna()].sort_index(kind='stable')
    df = df.loc[~df.index.duplicated(keep='last')]
    values = df[cols].apply(pd.to_numeric, errors='coerce')
    if clean_raw:
        for col in cols:
            cap = 30000 if col == 'Aggregate' else 5000
            values[col] = values[col].mask((values[col] < 0) | (values[col] > cap))
    minute = values.resample('1min').mean()
    count = values.resample('1min').count()
    # Require >=3 readings (approximately half the nominal sample count).
    minute = minute.where(count >= 3)
    if clean_raw and fill:
        minute = minute.apply(fill_short_gaps)
    return minute, count


def savefig(name):
    plt.tight_layout()
    plt.savefig(REPORT / f'{name}.png', dpi=150)
    plt.close()


def raw_cleaning():
    cols = ['Aggregate'] + [f'Appliance{i}' for i in range(1,10)]
    plotcols = ['Aggregate', 'Appliance5', 'Appliance6']
    files = sorted((DATA / 'raw').glob('*House1_*csv'))
    if len(files) != 2:
        raise FileNotFoundError('Expected RAW_House1_Part1.csv and Part2.csv')
    frames = [pd.read_csv(f) for f in files]
    source_stats = []
    for f, d in zip(files, frames):
        t = pd.to_datetime(d.Time, errors='coerce')
        epoch = pd.to_datetime(d.Unix, unit='s', utc=True, errors='coerce')
        delta = d.Unix.diff()
        source_stats.append({'file':f.name, 'rows':len(d), 'bad_time':int(t.isna().sum()),
            'bad_epoch':int(epoch.isna().sum()), 'backward_steps':int((delta < 0).sum()),
            'duplicate_epochs':int(d.Unix.duplicated().sum()),
            'median_interval_seconds':float(delta[delta>0].median()),
            'intervals_not_8s':int((delta.dropna()!=8).sum()),
            'time_minus_epoch_seconds':t.sub(epoch.dt.tz_localize(None)).dt.total_seconds().value_counts().head().to_dict()})
    df = pd.concat(frames, ignore_index=True)
    del frames
    invalid = {}
    for col in ['Aggregate'] + [f'Appliance{i}' for i in range(1,10)]:
        s = df[col]
        invalid[col] = {'missing':int(s.isna().sum()), 'zeros':int((s==0).sum()),
            'negative':int((s<0).sum()), 'above_cap':int((s>(30000 if col=='Aggregate' else 5000)).sum()),
            'max_w':float(s.max())}
    before, _ = minute_frame(df, cols)
    after, counts = minute_frame(df, cols, clean_raw=True)
    nofill, _ = minute_frame(df, cols, clean_raw=True, fill=False)
    stats = {'files':source_stats, 'combined_duplicate_epochs':int(df.Unix.duplicated().sum()),
        'source_channel_quality':invalid, 'start_utc':str(after.index.min()), 'end_utc':str(after.index.max()),
        'grid_minutes':len(after), 'coverage_rule':'at least 3 valid samples/minute',
        'before_missing_minutes':before.isna().sum().to_dict(),
        'after_missing_minutes':after.isna().sum().to_dict(),
        'filled_minutes':(nofill.isna() & after.notna()).sum().to_dict(),
        'max_samples_in_minute':counts.max().to_dict()}
    gaps=[]
    for col in cols:
        missing=after[col].isna()
        group=missing.ne(missing.shift()).cumsum()
        for _,run in after.loc[missing].groupby(group[missing]):
            gaps.append({'channel':col,'start_utc':str(run.index[0]),'end_utc':str(run.index[-1]),'minutes':len(run)})
    pd.DataFrame(gaps).to_csv(REPORT/'raw_outages.csv',index=False)
    stats['outage_runs']=len(gaps)
    stats['longest_remaining_gap_minutes']=max(g['minutes'] for g in gaps)
    dump(stats, 'raw_quality.json')
    after.to_pickle(DATA / 'raw_house1_minute.pkl')
    # Show real high-load segment and the worst gap separately.
    valid = before.Aggregate.dropna()
    peak = before.Appliance5.dropna().idxmax().floor('D')
    missing = before.Aggregate.isna()
    groups = missing.ne(missing.shift()).cumsum()
    sizes = missing.groupby(groups).sum()
    longest = sizes.idxmax()
    gap_center = before.index[groups == longest][0]
    for name, start, end in [('raw_before_after', peak, peak+pd.Timedelta(days=1)),
        ('raw_gap_detail', gap_center-pd.Timedelta(minutes=15), gap_center+pd.Timedelta(minutes=180))]:
        fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
        for ax,col,label in zip(axes,plotcols,['Whole home','Washing machine','Dishwasher']):
            ax.plot(before.loc[start:end].index,before.loc[start:end,col],label='Before',alpha=.6,lw=1)
            ax.plot(after.loc[start:end].index,after.loc[start:end,col],label='After',alpha=.8,lw=1)
            ax.set_ylabel(f'{label}\nW'); ax.legend(loc='upper right'); ax.set_xlim(start,end)
        axes[-1].set_xlabel('Raw epoch time, UTC (raw DST offset is not corrected)')
        savefig(name)
    return stats


def cycles(series, threshold=20, bridge=10):
    """Merge active minutes across <=10 observed off minutes, never across NaN."""
    active = series.notna() & (series > threshold)
    positions = np.flatnonzero(active.to_numpy())
    records = []
    if not len(positions):
        return pd.DataFrame(columns=['start','end','duration_min','energy_kwh','peak_w','active_min'])
    missing_prefix = np.r_[0, np.cumsum(series.isna().to_numpy())]
    previous, following = positions[:-1], positions[1:]
    missing_between = missing_prefix[following+1] - missing_prefix[previous]
    breaks = np.flatnonzero((following-previous-1 > bridge) | (missing_between > 0))+1
    for pos in np.split(positions,breaks):
        a,b = pos[0],pos[-1]
        duration = b-a+1
        energy = series.iloc[a:b+1].sum()/60000
        # Candidate operating events, not validated physical wash cycles.
        if duration < 10 or energy < .02:
            continue
        if a == 0 or b == len(series)-1 or pd.isna(series.iloc[a-1]) or pd.isna(series.iloc[b+1]):
            continue  # boundary-censored events
        records.append({'start':series.index[a],'end':series.index[b], 'duration_min':duration,
            'energy_kwh':energy,'peak_w':series.iloc[a:b+1].max(), 'active_min':len(pos)})
    return pd.DataFrame(records,columns=['start','end','duration_min','energy_kwh','peak_w','active_min'])


def read_clean_house(home):
    path = DATA / 'cleaned' / f'CLEAN_House{home}.csv'
    cols = ['Aggregate'] + [f'Appliance{i}' for i in WM[home]]
    df = pd.read_csv(path,usecols=['Unix', 'Issues']+cols)
    # Cleaned epochs have official DST correction; Time is intentionally not reparsed.
    minute,count = minute_frame(df,cols)
    issues = pd.Series(df.Issues.to_numpy(),index=pd.to_datetime(df.Unix,unit='s',utc=True))
    issue_min = issues.sort_index().resample('1min').max().reindex(minute.index)
    quality = {'house':home, 'rows':len(df), 'duplicates':int(df.Unix.duplicated().sum()),
        'start_utc':str(minute.index.min()), 'end_utc':str(minute.index.max()),
        'grid_minutes':len(minute),'issue_minutes':int((issue_min>0).sum())}
    # Reject inconsistent labels; do not use Issues as an input feature.
    for col in cols:
        cap = 30000 if col == 'Aggregate' else 4000
        minute[col] = minute[col].mask((minute[col]<0) | (minute[col]>cap))
    minute['Issues'] = issue_min
    return minute.astype('float32'),quality


def eda():
    summaries, events_all, quality_all, sensitivity = [],[],[],[]
    cache = DATA / 'minute'
    cache.mkdir(exist_ok=True)
    for home, channels in WM.items():
        print(f'EDA house {home}',flush=True)
        f,q = read_clean_house(home)
        f.to_pickle(cache / f'house{home}.pkl')
        quality_all.append(q)
        for channel in channels:
            col=f'Appliance{channel}'
            s = f[col]
            events = cycles(s)
            events['house']=home; events['channel']=channel
            events_all.append(events)
            row = {'house':home,'channel':channel, 'start_utc':str(s.index.min()),'end_utc':str(s.index.max()),
                'grid_minutes':len(s),'valid_minutes':int(s.notna().sum()),'missing_or_excluded_minutes':int(s.isna().sum()),
                'zero_pct_valid':float(s.eq(0).sum()/max(1,s.notna().sum())*100),
                'off_pct_valid':float(s.le(20).sum()/max(1,s.notna().sum())*100),'events':len(events),
                'observed_energy_kwh':float(s.sum()/60000),
                'issues_flagged_valid_minutes':int(((f.Issues>0)&s.notna()).sum()),
                'target_exceeds_aggregate_minutes':int((s>f.Aggregate).sum()),
                'median_cycle_min':float(events.duration_min.median()),
                'median_cycle_kwh':float(events.energy_kwh.median()),
                'median_peak_w':float(events.peak_w.median())}
            summaries.append(row)
            for threshold,bridge in [(10,10),(20,5),(20,10),(20,15),(50,10)]:
                c=cycles(s,threshold,bridge)
                sensitivity.append({'house':home,'channel':channel,'threshold_w':threshold,'bridge_min':bridge,
                    'events':len(c),'median_duration_min':float(c.duration_min.median())})
    summary = pd.DataFrame(summaries)
    events = pd.concat([e for e in events_all if len(e)],ignore_index=True)
    summary.to_csv(REPORT/'availability.csv',index=False)
    events.to_csv(REPORT/'cycles.csv',index=False)
    pd.DataFrame(quality_all).to_csv(REPORT/'clean_quality.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(REPORT/'cycle_sensitivity.csv',index=False)
    h1 = pd.read_pickle(cache/'house1.pkl')
    s=h1.Appliance5
    ev = cycles(s)
    candidates=ev.loc[(ev.duration_min.between(45,180)) & (ev.energy_kwh.between(.3,2))]
    representative=(candidates if len(candidates) else ev).iloc[0]
    day=representative.start.floor('D')
    trace=h1.loc[day:day+pd.Timedelta(days=1)].iloc[:-1]
    plt.figure(figsize=(12,4));plt.plot(trace.index,trace.Aggregate,label='Aggregate',lw=1)
    plt.plot(trace.index,trace.Appliance5,label='Washing machine',lw=1);plt.legend();plt.ylabel('W');plt.xlabel('UTC');savefig('representative_day')
    detail=s.loc[representative.start-pd.Timedelta(minutes=10):representative.end+pd.Timedelta(minutes=10)]
    plt.figure(figsize=(10,4));plt.plot(detail.index,detail);plt.axhline(20,color='gray',ls='--',label='20 W active threshold')
    plt.legend();plt.ylabel('W');plt.xlabel('UTC');savefig('cycle_trace')
    local=pd.to_datetime(events.start,utc=True).dt.tz_convert('Europe/London')
    events['hour']=local.dt.hour;events['weekday']=local.dt.dayofweek
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    events.groupby('hour').size().reindex(range(24),fill_value=0).plot.bar(ax=axes[0],color='#247a89')
    events.groupby('weekday').size().reindex(range(7),fill_value=0).plot.bar(ax=axes[1],color='#247a89')
    axes[0].set_xlabel('UK local start hour');axes[1].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],rotation=0)
    for ax in axes:ax.set_ylabel('Detected events (unequal exposure)')
    savefig('usage_patterns')
    # Exposure-normalized energy in each local hour/day, per observed minute.
    buckets=[]
    for home,chs in WM.items():
        f=pd.read_pickle(cache/f'house{home}.pkl');local=f.index.tz_convert('Europe/London')
        for ch in chs:
            s=f[f'Appliance{ch}']
            tmp=pd.DataFrame({'w':s.to_numpy(),'hour':local.hour,'weekday':local.dayofweek})
            for dim in ['hour','weekday']:
                g=tmp.groupby(dim).w.agg(['sum','count'])
                for key,r in g.iterrows():buckets.append({'house':home,'channel':ch,'dimension':dim,'bucket':key,'energy_kwh':r['sum']/60000,'observed_minutes':r['count']})
    buckets=pd.DataFrame(buckets);buckets.to_csv(REPORT/'time_buckets.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,dim in zip(axes,['hour','weekday']):
        g=buckets[buckets.dimension==dim].groupby('bucket')[['energy_kwh','observed_minutes']].sum()
        (g.energy_kwh/g.observed_minutes*60).plot.bar(ax=ax,color='#247a89')
        ax.set_ylabel('kWh per observed hour');ax.set_xlabel('UK local '+dim)
    savefig('usage_energy_normalized')
    selected=[1,2,5,6,7,9]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,col,label in zip(axes,['energy_kwh','duration_min'],['Cycle energy (kWh)','Duration (minutes)']):
        ax.boxplot([events.loc[events.house==h,col].to_numpy() for h in selected],tick_labels=selected,showfliers=False)
        ax.set_xlabel('House');ax.set_ylabel(label)
    savefig('cycle_comparison')
    return summary,events


def features(aggregate):
    """Offline sequence-to-point features; max future lookahead is 30 min."""
    x=pd.DataFrame(index=aggregate.index)
    for lag in [-30,-15,-10,-5,-2,-1,0,1,2,5,10,15,30]:
        x[f'lag_{lag}']=aggregate.shift(lag)
    for size in [5,15,31,61]:
        roll=aggregate.rolling(size,center=True,min_periods=size)
        for statistic in ['mean','std','min','max']:
            x[f'{statistic}_{size}']=getattr(roll,statistic)()
    x['delta']=aggregate.diff()
    # No household identity, calendar, issue flags or appliance-derived features.
    good=aggregate.notna().rolling(61,center=True,min_periods=61).sum().eq(61)
    return x.astype('float32').where(good, np.nan)


def split_mask(index, split):
    periods={'train':('2014-03-01','2014-07-01'), 'validation':('2014-07-01','2014-08-01'), 'test':('2014-08-01','2014-09-01')}
    a,b=[pd.Timestamp(t,tz='UTC') for t in periods[split]]
    # Every feature window belongs entirely to its split, with a 60-minute embargo.
    return (index>=a+pd.Timedelta(minutes=CONTEXT)) & (index<b-pd.Timedelta(minutes=CONTEXT))


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    active=y>20
    pr,rc,f1,_=precision_recall_fscore_support(active,p>20,average='binary',zero_division=0)
    true_energy=y.sum()/60000;pred_energy=p.sum()/60000
    return {'n':len(y),'active_pct':float(active.mean()*100),'mae_w':float(np.abs(y-p).mean()),
        'rmse_w':float(np.sqrt(np.mean((y-p)**2))), 'active_mae_w':float(np.abs(y[active]-p[active]).mean()) if active.any() else None,
        'off_mae_w':float(np.abs(y[~active]-p[~active]).mean()) if (~active).any() else None,
        'precision':float(pr),'recall':float(rc),'f1':float(f1),
        'average_precision':float(average_precision_score(active,p)) if active.any() else None,
        'observed_true_kwh':float(true_energy),'observed_pred_kwh':float(pred_energy),
        'relative_energy_error':float(abs(pred_energy-true_energy)/true_energy) if true_energy else None}


def model_experiment():
    sets={s:[] for s in ['train','validation','test','unseen']}
    coverage=[]
    for home in TRAIN_HOMES+[TEST_HOME]:
        f=pd.read_pickle(DATA/'minute'/f'house{home}.pkl').loc['2014-03-01':'2014-08-31 23:59']
        x=features(f.Aggregate)
        y=f[f'Appliance{WM[home][0]}']
        valid=x.notna().all(axis=1)&y.notna()&(f.Issues==0)&(y<=f.Aggregate)
        for split in (['unseen'] if home==TEST_HOME else ['train','validation','test']):
            period=split_mask(x.index,'test' if split=='unseen' else split)
            mask=valid & period
            coverage.append({'house':home,'split':split,'embargoed_grid_minutes':int(period.sum()),
                'eligible_minutes':int(mask.sum()),'eligible_pct':float(mask.sum()/max(1,period.sum())*100),
                'missing_context_minutes':int((period & ~x.notna().all(axis=1)).sum()),
                'missing_target_minutes':int((period & y.isna()).sum()),
                'issue_minutes':int((period & (f.Issues>0)).sum()),
                'target_above_aggregate_minutes':int((period & (y>f.Aggregate)).sum())})
            part=x.loc[mask].copy();part['target']=y.loc[mask];part['house']=home
            sets[split].append(part)
    pd.DataFrame(coverage).to_csv(REPORT/'split_coverage.csv',index=False)
    sets={s:pd.concat(v) for s,v in sets.items()}
    train=sets['train'];val=sets['validation']
    names=[c for c in train.columns if c not in ['target','house']]
    # Deterministic uniform training subsample preserves natural on/off prevalence.
    train=train.sample(n=min(180000,len(train)),random_state=SEED).sort_index()
    x=train[names];y=train.target
    candidates=[];models=[]
    for loss in ['squared_error','absolute_error']:
        print('Training',loss,len(train),flush=True)
        model=HistGradientBoostingRegressor(loss=loss,max_iter=140,max_leaf_nodes=31,
            learning_rate=.08,min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=SEED)
        model.fit(x,y)
        p=np.clip(model.predict(val[names]),0,val.lag_0.to_numpy())
        m=metrics(val.target,p)
        candidates.append({'loss':loss,**m});models.append(model)
    # Predeclared energy-aware validation objective, never optimized against test.
    scores=[m['mae_w']+50*m['relative_energy_error'] for m in candidates]
    winner=int(np.argmin(scores));model=models[winner]
    dump({'candidates':candidates,'selection_objective':'validation MAE + 50 * relative energy error',
          'selected_loss':candidates[winner]['loss'],'training_rows_full':len(sets['train']),
          'training_rows_used':len(train),'features':names},'model_selection.json')
    joblib.dump({'model':model,'features':names},REPORT/'model.joblib')
    rows=[];predictions=[];daily=[]
    mean=float(y.mean())
    for split in ['validation','test','unseen']:
        d=sets[split];truth=d.target.to_numpy();aggregate=d.lag_0.to_numpy()
        outputs={'zero':np.zeros(len(d)), 'training_mean':np.minimum(mean,aggregate),
                 'sequence_gbdt':np.clip(model.predict(d[names]),0,aggregate)}
        for name,p in outputs.items():
            rows.append({'split':split,'house':'pooled','model':name,**metrics(truth,p)})
            for home in d.house.unique():
                mask=d.house.to_numpy()==home
                rows.append({'split':split,'house':int(home),'model':name,**metrics(truth[mask],p[mask])})
        pred=pd.DataFrame({'house':d.house.to_numpy(),'actual_w':truth,'predicted_w':outputs['sequence_gbdt'],
            'aggregate_w':aggregate,'split':split},index=d.index)
        predictions.append(pred)
        for home,g in pred.groupby('house'):
            sums=g.resample('1D').agg({'actual_w':'sum','predicted_w':'sum','aggregate_w':'count'})
            for date,r in sums.iterrows():
                daily.append({'split':split,'house':home,'day_utc':str(date),'evaluated_minutes':r.aggregate_w,
                    'actual_kwh':r.actual_w/60000,'predicted_kwh':r.predicted_w/60000})
    result=pd.DataFrame(rows);result.to_csv(REPORT/'metrics.csv',index=False)
    pred=pd.concat(predictions);pred.to_csv(REPORT/'predictions.csv.gz',index_label='timestamp_utc',compression='gzip')
    days=pd.DataFrame(daily);days.to_csv(REPORT/'daily_energy.csv',index=False)
    # Block-bootstrap day-level energy error intervals, keeping correlated minutes together.
    intervals=[];rng=np.random.default_rng(SEED)
    for split,g in days.groupby('split'):
        byday=g.groupby('day_utc')[['actual_kwh','predicted_kwh']].sum()
        estimates=[]
        for _ in range(500):
            a=byday.iloc[rng.integers(0,len(byday),len(byday))].sum()
            if a.actual_kwh>0:estimates.append(abs(a.predicted_kwh-a.actual_kwh)/a.actual_kwh)
        intervals.append({'split':split,'relative_energy_error_ci_95':np.quantile(estimates,[.025,.975]).tolist(),
            'note':'day block bootstrap, conditional on these households; not population CI'})
    dump(intervals,'uncertainty.json')
    # Transparent example-selection: highest ground-truth energy day in H1 test, plus unseen H7.
    for split,home,name in [('test',1,'prediction_seen'),('unseen',7,'prediction_unseen')]:
        g=pred[(pred.split==split)&(pred.house==home)].sort_index()
        day=g.actual_w.resample('1D').sum().idxmax()
        trace=g.loc[day:day+pd.Timedelta(days=1)].reindex(pd.date_range(day,periods=1440,freq='min'))
        plt.figure(figsize=(12,4));plt.plot(trace.index,trace.actual_w,label='Ground truth',lw=1)
        plt.plot(trace.index,trace.predicted_w,label='Prediction',lw=1);plt.legend();plt.ylabel('W');plt.xlabel('UTC');savefig(name)
    resolution=[]
    for (split,home),g in pred.groupby(['split','house']):
        for minutes in [1,15,30]:
            means=g[['actual_w','predicted_w']].resample(f'{minutes}min').mean()
            count=g.actual_w.resample(f'{minutes}min').count()
            means=means.loc[count==minutes]
            if len(means):
                m=metrics(means.actual_w,means.predicted_w)
                # metrics integrates 1-min points; rescale only absolute energies.
                m['observed_true_kwh']*=minutes;m['observed_pred_kwh']*=minutes
                resolution.append({'split':split,'house':home,'resolution_min':minutes,**m})
    pd.DataFrame(resolution).to_csv(REPORT/'resolution_sensitivity.csv',index=False)
    failures=days[days.split.isin(['test','unseen'])].copy()
    failures['abs_energy_error_kwh']=(failures.actual_kwh-failures.predicted_kwh).abs()
    failures.sort_values('abs_energy_error_kwh',ascending=False).head(15).to_csv(REPORT/'worst_days.csv',index=False)
    plt.figure(figsize=(6,5));plt.scatter(days.actual_kwh,days.predicted_kwh,s=10,alpha=.6)
    lim=max(days.actual_kwh.max(),days.predicted_kwh.max());plt.plot([0,lim],[0,lim],color='gray',ls='--')
    plt.xlabel('Actual observed energy/day (kWh)');plt.ylabel('Predicted observed energy/day (kWh)');savefig('daily_energy_scatter')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','raw','eda','model'],default='all')
    args=parser.parse_args();REPORT.mkdir(exist_ok=True)
    start=time.time()
    if args.stage in ['all','raw']:raw_cleaning()
    if args.stage in ['all','eda']:eda()
    if args.stage in ['all','model']:model_experiment()
    dump({'seed':SEED,'python':platform.python_version(),'elapsed_seconds':time.time()-start,'stage':args.stage},'run.json')

if __name__=='__main__':
    main()
