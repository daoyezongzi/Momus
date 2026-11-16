"""Regression tests for demonstrated measurement errors and paired inference."""
import unittest

import numpy as np
import pandas as pd

from audit import bootstrap_effects, corrected_benchmark, investment_metrics, metrics


class AuditRegressionTests(unittest.TestCase):
    def test_tencent_field_order_known_day(self):
        # Raw API row: 2024-01-02,3426.270,3386.350,3426.270,3386.350,...
        # Former decoder gave field 1 the name close.
        old=pd.DataFrame({"close":[3426.27],"open":[3386.35]})
        fixed=corrected_benchmark(old)
        self.assertEqual(fixed.close.iloc[0],3386.35)
        self.assertEqual(fixed.open.iloc[0],3426.27)
        self.assertEqual(old.close.iloc[0],3426.27)

    def test_first_entry_fee_is_included(self):
        result,r,rb=investment_metrics([99.,99.],[100.,100.],100.,100.)
        self.assertAlmostEqual(result["total_return_net"],-.01)
        self.assertAlmostEqual(result["mdd"],-.01)
        self.assertAlmostEqual(r[0],-.01)
        self.assertEqual(result["valid_days"],2)
        old,_,_=investment_metrics([99.,99.],[100.,100.])
        self.assertEqual(old["total_return_net"],0)

    def test_path_axis_and_rank_ties(self):
        truth=np.repeat(np.arange(10.)[:,None],4,axis=1)
        pred=truth[::-1].copy()
        result=metrics(pred,truth)
        self.assertAlmostEqual(result["price_ic"],-1)
        self.assertAlmostEqual(result["price_rankic"],-1)
        tied=pred.copy();tied[0]=tied[1]
        expected=pd.Series(tied[:,0]).rank().corr(pd.Series(truth[:,0]).rank())
        self.assertAlmostEqual(metrics(tied,truth)["price_rankic"],expected)
        affine=metrics(tied*np.arange(1,5)+10,truth*np.arange(1,5)+10)
        self.assertAlmostEqual(affine["price_rankic"],expected)

    def test_paired_cluster_resampling_preserves_constant_effect(self):
        b=np.ones((12,4))*2;a=b-.2
        for kind in ("stock","time","crossed"):
            q=bootstrap_effects(a,b,np.repeat(np.arange(3),4),np.tile(np.arange(4),3),reps=500,kind=kind)
            np.testing.assert_allclose(q[:,:4],-.2)
            np.testing.assert_allclose(q[:,4],10.)


if __name__=="__main__":unittest.main()
