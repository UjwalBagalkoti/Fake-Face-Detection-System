from training.metrics import calibrate_threshold,binary_metrics
def test_metrics_and_threshold():
    y=[0,0,1,1]; p=[.05,.2,.8,.95]; m=binary_metrics(y,p,calibrate_threshold(y,p)); assert m["accuracy"]==1.0 and m["roc_auc"]==1.0
