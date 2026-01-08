mkdir -p ../pth

python3 RINAMI_foldability_prediction_train_and_test.py ../pth/foldability_predictor_0.pth
python3 RINAMI_regression_train_and_test.py ../pth/dG_predictor_0.pth ../pth/foldability_predictor_0.pth

for i in {1..10}
do
  echo "training:${i}step"
  python3 RINAMI_foldability_prediction_train_and_test_with_small_lr.py ../pth/foldability_predictor_${i}.pth ../pth/dG_predictor_$((i - 1)).pth
  python3 RINAMI_regression_train_and_test.py ../pth/dG_predictor_${i}.pth ../pth/foldability_predictor_${i}.pth
done
