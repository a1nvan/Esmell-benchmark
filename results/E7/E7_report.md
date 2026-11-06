# E7 — End-to-end composition and quantification

## Scientific question
Can sensor-only inference recover nominal presence and concentration on future readings?

- Train 2338; grouped validation 71; independent within130 test 152;
  future146 test 916. Common mixture recipes: 39.
- Complete benchmark: True. Auxiliary CNN loss multiplier: 0.5.
- This is RETROSPECTIVE: 146 has already been examined in E5/E6.
- Small air/pure support on 28 September; only one mixture day in 146.
- No independently verified delivered concentrations. No causal identification of aging.

## Both future tests (no test-based model selection)
```
                    model  seed     split  e2e_macro8_nmae   macro8  exact_mix  pooled_mae_ppb  fn_fraction  fp_fraction  tp_fraction
                AlwaysAir    42 within130         0.282822 0.125000   0.000000     1775.260965     1.000000     0.000000     0.000000
                AlwaysAir    42 future146         0.259077 0.125000   0.000000      559.674672     1.000000     0.000000     0.000000
                TrainMode    42 within130         0.307079 0.125000   0.000000     1528.199561     0.655985     0.139110     0.204905
                TrainMode    42 future146         0.303469 0.125000   0.000000     1086.901747     0.297815     0.555262     0.146923
                AlwaysAll    42 within130         0.364467 0.125000   0.504000     1278.537281     0.000000     0.418428     0.581572
                AlwaysAll    42 future146         0.375729 0.125000   0.500000     1799.875546     0.000000     0.801042     0.198958
       ExtraTrees3+Median    42 within130         0.281674 0.132937   0.032000     1714.410088     0.941847     0.017199     0.040953
       ExtraTrees3+Median    42 future146         0.229947 0.183191   0.010204      608.358443     0.793691     0.140154     0.066155
ExtraTrees3+ExtraTreesReg    42 within130         0.281896 0.132937   0.032000     1726.143660     0.935445     0.020246     0.044309
ExtraTrees3+ExtraTreesReg    42 future146         0.251421 0.183191   0.010204      646.015690     0.747426     0.174662     0.077913
            ROCKET+Median    42 within130         0.289346 0.159352   0.048000     1674.958333     0.881651     0.049997     0.068352
            ROCKET+Median    42 future146         0.260014 0.251690   0.163265      821.932678     0.366890     0.458863     0.174247
     ROCKET+ExtraTreesReg    42 within130         0.312605 0.159352   0.048000     1697.145233     0.870125     0.043891     0.085984
     ROCKET+ExtraTreesReg    42 future146         0.316158 0.251690   0.163265      957.065871     0.315087     0.478138     0.206775
 CNN_presence_only+Median    42 within130         0.283561 0.147181   0.056000     1670.333333     0.910602     0.031412     0.057986
 CNN_presence_only+Median    42 future146         0.255971 0.209633   0.071429      763.160480     0.519095     0.372879     0.108026
 CNN_presence_only+Median    43 within130         0.282896 0.157648   0.056000     1643.890351     0.909950     0.022362     0.067688
 CNN_presence_only+Median    43 future146         0.271945 0.164794   0.010204      766.515284     0.550870     0.353078     0.096053
 CNN_presence_only+Median    44 within130         0.277570 0.155664   0.048000     1663.914474     0.935005     0.019800     0.045196
 CNN_presence_only+Median    44 future146         0.244191 0.232038   0.091837      743.128821     0.578556     0.332951     0.088493
         CNN_joint+Median    42 within130         0.289488 0.127566   0.056000     1672.980263     0.895318     0.043247     0.061436
         CNN_joint+Median    42 future146         0.259492 0.223012   0.102041      781.162300     0.515045     0.374596     0.110359
        CNN_joint+RegHead    42 within130         0.282079 0.127566   0.056000     1686.263887     0.888265     0.023832     0.087903
        CNN_joint+RegHead    42 future146         0.254697 0.223012   0.102041      687.640953     0.585093     0.278274     0.136633
         CNN_joint+Median    43 within130         0.276995 0.177299   0.096000     1650.096491     0.920887     0.012482     0.066631
         CNN_joint+Median    43 future146         0.240313 0.250134   0.132653      727.095706     0.559102     0.340774     0.100124
        CNN_joint+RegHead    43 within130         0.273536 0.177299   0.096000     1667.605012     0.911219     0.008007     0.080774
        CNN_joint+RegHead    43 future146         0.257030 0.250134   0.132653      699.866116     0.580854     0.297714     0.121432
         CNN_joint+Median    44 within130         0.283260 0.174147   0.072000     1656.958333     0.896558     0.037161     0.066281
         CNN_joint+Median    44 future146         0.249900 0.231516   0.142857      789.200873     0.469699     0.396438     0.133863
        CNN_joint+RegHead    44 within130         0.284326 0.174147   0.072000     1702.959961     0.872339     0.030231     0.097429
        CNN_joint+RegHead    44 future146         0.271081 0.231516   0.142857      775.133199     0.478224     0.359452     0.162324
```

## Matched ablations (negative nMAE difference favors first method)
```
                    contrast  seed     split   n  delta_nmae_sample_mean  delta_nmae_mixture_mean  delta_macro8_nmae
    Auxiliary loss: detector    42 within130 152                0.002206             2.683077e-03           0.005927
    Auxiliary loss: detector    42 future146 916                0.011672            -1.878598e-02           0.003521
    Auxiliary loss: detector    43 within130 152               -0.002193            -1.332268e-18          -0.005901
    Auxiliary loss: detector    43 future146 916               -0.005203            -8.173731e-02          -0.031631
    Auxiliary loss: detector    44 within130 152                0.001117            -1.312821e-03           0.005690
    Auxiliary loss: detector    44 future146 916                0.019464            -2.895343e-02           0.005709
        Joint head vs median    42 within130 152                0.000741             3.297866e-03          -0.007409
        Joint head vs median    42 future146 916               -0.009228            -3.576551e-03          -0.004795
        Joint head vs median    43 within130 152               -0.000095             4.303919e-04          -0.003459
        Joint head vs median    43 future146 916                0.012505             1.213428e-02           0.016717
        Joint head vs median    44 within130 152                0.007504             1.047685e-02           0.001066
        Joint head vs median    44 future146 916                0.011394             2.141661e-02           0.021182
Cascaded regressor vs median    42 within130 152               -0.000511            -5.627403e-04           0.000222
Cascaded regressor vs median    42 future146 916                0.026923             7.249069e-03           0.021475
  ROCKET regressor vs median    42 within130 152                0.022178             2.164307e-02           0.023259
  ROCKET regressor vs median    42 future146 916                0.047041             4.323199e-02           0.056144
```

## Exact-recipe-controlled evaluation
```
                    model  seed     split  recipes  mean_recipe_nmae  mean_recipe_exact  records
                AlwaysAir    42 future146       39          0.493669           0.000000       85
                AlwaysAir    42 within130       39          0.493669           0.000000      105
                AlwaysAll    42 future146       39          0.297153           0.487179       85
                AlwaysAll    42 within130       39          0.297153           0.487179      105
         CNN_joint+Median    42 future146       39          0.382911           0.108547       85
         CNN_joint+Median    42 within130       39          0.467354           0.061966      105
         CNN_joint+Median    43 future146       39          0.355114           0.153846       85
         CNN_joint+Median    43 within130       39          0.463798           0.098291      105
         CNN_joint+Median    44 future146       39          0.343613           0.152137       85
         CNN_joint+Median    44 within130       39          0.466291           0.074786      105
        CNN_joint+RegHead    42 future146       39          0.398106           0.108547       85
        CNN_joint+RegHead    42 within130       39          0.469141           0.061966      105
        CNN_joint+RegHead    43 future146       39          0.378882           0.153846       85
        CNN_joint+RegHead    43 within130       39          0.461157           0.098291      105
        CNN_joint+RegHead    44 future146       39          0.377679           0.152137       85
        CNN_joint+RegHead    44 within130       39          0.471092           0.074786      105
 CNN_presence_only+Median    42 future146       39          0.395468           0.094872       85
 CNN_presence_only+Median    42 within130       39          0.465202           0.070513      105
 CNN_presence_only+Median    43 future146       39          0.431407           0.025641       85
 CNN_presence_only+Median    43 within130       39          0.460938           0.051282      105
 CNN_presence_only+Median    44 future146       39          0.379281           0.111111       85
 CNN_presence_only+Median    44 within130       39          0.463421           0.042735      105
ExtraTrees3+ExtraTreesReg    42 future146       39          0.408710           0.008547       85
ExtraTrees3+ExtraTreesReg    42 within130       39          0.467004           0.047009      105
       ExtraTrees3+Median    42 future146       39          0.393614           0.008547       85
       ExtraTrees3+Median    42 within130       39          0.471254           0.047009      105
     ROCKET+ExtraTreesReg    42 future146       39          0.406236           0.197863       85
     ROCKET+ExtraTreesReg    42 within130       39          0.476189           0.049145      105
            ROCKET+Median    42 future146       39          0.357096           0.197863       85
            ROCKET+Median    42 within130       39          0.454039           0.049145      105
                TrainMode    42 future146       39          0.433840           0.000000       85
                TrainMode    42 within130       39          0.433840           0.000000      105
```

## Interpretation protocol
1. Compare CNN_joint+Median vs CNN_presence_only+Median for auxiliary training effect on detection.
2. Compare CNN_joint+RegHead vs CNN_joint+Median for numeric benefit at fixed detector.
3. Compare ExtraTrees3+ExtraTreesReg vs ExtraTrees3+Median for cascaded regression benefit.
4. Compare within130 vs future146 descriptively, not as isolated physical drift.
5. Inspect per_gas.csv, per_composition.csv, errors FP/FN/TP, exact_recipe_matched.csv.
6. Never interpret three CNN seeds as independent sensor/session replications.
