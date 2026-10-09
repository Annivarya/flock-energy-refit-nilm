"""Behavioral tests for gap preservation, event boundaries and leakage controls."""
import unittest
import numpy as np
import pandas as pd
from pipeline import fill_short_gaps,minute_frame,cycles,features,split_mask,metrics

class PipelineTests(unittest.TestCase):
    def test_long_gap_is_not_partially_filled(self):
        idx=pd.date_range('2020-01-01',periods=9,freq='min',tz='UTC')
        s=pd.Series([1,np.nan,3,np.nan,np.nan,np.nan,7,np.nan,9],index=idx)
        result=fill_short_gaps(s)
        self.assertEqual(result.iloc[1],2)
        self.assertTrue(result.iloc[3:6].isna().all())
        self.assertEqual(result.iloc[7],8)

    def test_stable_last_duplicate_and_minute_coverage(self):
        d=pd.DataFrame({'Unix':[0,10,20,20,60],'Aggregate':[3,3,9,6,100]})
        minute,count=minute_frame(d,['Aggregate'])
        self.assertEqual(minute.iloc[0,0],4)
        self.assertEqual(count.iloc[0,0],3)
        self.assertTrue(pd.isna(minute.iloc[1,0]))

    def test_cycles_never_bridge_missing_labels(self):
        idx=pd.date_range('2020-01-01',periods=40,freq='min',tz='UTC')
        s=pd.Series([0]+[500]*12+[0]*2+[np.nan]+[0]*2+[500]*12+[0]*10,index=idx)
        detected=cycles(s)
        self.assertEqual(len(detected),2)
        self.assertTrue((detected.duration_min==12).all())

    def test_feature_support_and_split_embargo(self):
        idx=pd.date_range('2014-06-30',periods=3000,freq='min',tz='UTC')
        s=pd.Series(np.arange(len(idx)),index=idx,dtype=float)
        x=features(s)
        self.assertEqual(x.loc[idx[100],'lag_-30'],s.iloc[130])
        train=split_mask(idx,'train');val=split_mask(idx,'validation')
        self.assertLess(idx[train].max()+pd.Timedelta(minutes=30),pd.Timestamp('2014-07-01',tz='UTC'))
        self.assertGreaterEqual(idx[val].min()-pd.Timedelta(minutes=30),pd.Timestamp('2014-07-01',tz='UTC'))
        s.iloc[150]=np.nan;x=features(s)
        self.assertTrue(x.iloc[120:181].isna().all(axis=1).all())

    def test_energy_conversion_and_zero_baseline(self):
        m=metrics(np.full(60,1000),np.zeros(60))
        self.assertEqual(m['observed_true_kwh'],1)
        self.assertEqual(m['relative_energy_error'],1)
        self.assertEqual(m['recall'],0)

if __name__=='__main__':unittest.main()
