from DataPreprocessor import DataPreprocessor 
from TrainTestPipeline import TrainTestPipeline
import pandas as pd
import os 

# if not os.path.exists('MicrotracMinMax'):
#     os.mkdir('MicrotracMinMax')


data_folder = './MicrotracDataFilesPT'
flow_values_excel = './TrueFlowValues_.xlsx'
dp = DataPreprocessor(data_folder, flow_values_excel, root_folder_ = '.')


success = dp.prepare_df(preproc_type = 'yeo-johnson')
assert success

# # Pearson-correlation, full dataset with augmented features
x_filt, columns = dp.get_feature_selection_x(method='pearson', threshold = 0.8, \
                                            heldout_cols = ['Density'])


y_regr = dp.get_regression_y()
all_samples = dp.get_samples()

pipeline = TrainTestPipeline(x_data = x_filt, y_data = y_regr, all_samples = all_samples, 
                             model_name = 'RandomForestRegressor', heldout_samples = 'random', num_heldout = 4)


tr_test_ = pipeline.do_train_test(cv = False)


pipeline2 = TrainTestPipeline(x_data = x_filt, y_data = y_regr, all_samples = all_samples, 
                             model_name = 'DecisionTreeRegressor', heldout_samples = 'random', num_heldout = 4)

pipeline2.exhaustive_train(outfile_name = '../MicrotracMinMax/ExhaustiveTrainTest_RandomForest_Augmented_filtered.csv', cv=False)
# # MIC correlation, full dataset with augmented features 
# x_filt_mic, columns_mic = dp.get_feature_selection_x(method='mic', threshold = 0.8, \
#                                             heldout_cols = ['Density'])

# # Pearson correlation with a threshold of 0.8
# dp.visualize_correlation(output_file = '../MicrotracMinMax/PearsonCorrelation_for_Augmented_Data.pdf', )
# # MIC-correlation with a threshold of 0.8
# dp.visualize_correlation(output_file = '../MicrotracMinMax/MICCorrelation_for_Augmented_Data.pdf', \
#                         method = 'mic', threshold = 0.8)
