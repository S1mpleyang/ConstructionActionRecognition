# Confuse Matrix

The data in the file (STR-Transformer.txt) should be like this:
```
Label:0 --> Prob:0   # true label --> predict label
Label:0 --> Prob:3
...
```

Modify the path to your file (STR-Transformer.txt) in draw_cm.py and run it


# ROC Curve

Generate two file 

(1) STR_Transformer_label.pth 

-size [540,7]

-one-hot code of true label

e.g. [[1., 0., 0.,  0, 0., 0., 0.]]

(2)STR_Transformer_tensor.pth

-size [540,7]

-scores generated by model

e.g. [[8.497159 , -3.47669744, -2.80618858, -7.54730511, -7.54730511, -0.45979768, -3.15947509]]

Modify the path to your file in roc_plot.py and run it

# Loss Curve

Generate a file train_STR_Transformer.log

-epoch loss

-1 0.17163007370288969

-2 0.02682878160196694

Modify the path to your file in loss_curve.py and run it

