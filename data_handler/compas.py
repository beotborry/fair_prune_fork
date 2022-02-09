import pandas as pd
from data_handler.AIF360.compas_dataset import CompasDataset
from data_handler.tabular_dataset import TabularDataset


class CompasDataset_torch(TabularDataset):
    """Adult dataset."""

    def __init__(self, root, split='train',sen_attr='race', group_mode=-1, influence_scores=None):

        # dataset = load_preproc_data_compas()
        dataset = CompasDataset(root_dir=root)
        if sen_attr == 'sex':
            sen_attr_idx = 0
        elif sen_attr == 'race':
            sen_attr_idx = 2
        else:
            raise Exception('Not allowed group')

        self.num_groups = 2
        self.num_classes = 2

        super(CompasDataset_torch, self).__init__(root=root, dataset=dataset, sen_attr_idx=sen_attr_idx,
                                                  split=split, group_mode=group_mode, influence_scores=influence_scores)


# def load_preproc_data_compas(protected_attributes=None):
#     def custom_preprocessing(df):
#         """The custom pre-processing function is adapted from
#             https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
#         """

#         df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
#                  'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
#                  'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

#         # Indices of data samples to keep
#         ix = df['days_b_screening_arrest'] <= 30
#         ix = (df['days_b_screening_arrest'] >= -30) & ix
#         ix = (df['is_recid'] != -1) & ix
#         ix = (df['c_charge_degree'] != "O") & ix
#         ix = (df['score_text'] != 'N/A') & ix
#         df = df.loc[ix,:]
#         df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
#                                 pd.to_datetime(df['c_jail_in'])).apply(
#                                                         lambda x: x.days)

#         # Restrict races to African-American and Caucasian
#         dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

#         # Restrict the features to use
#         dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
#                 'two_year_recid','length_of_stay']].copy()

#         # Quantize priors count between 0, 1-3, and >3
#         def quantizePrior(x):
#             if x <=0:
#                 return '0'
#             elif 1<=x<=3:
#                 return '1 to 3'
#             else:
#                 return 'More than 3'

#         # Quantize length of stay
#         def quantizeLOS(x):
#             if x<= 7:
#                 return '<week'
#             if 8<x<=93:
#                 return '<3months'
#             else:
#                 return '>3 months'

#         # Quantize length of stay
#         def adjustAge(x):
#             if x == '25 - 45':
#                 return '25 to 45'
#             else:
#                 return x

#         # Quantize score_text to MediumHigh
#         def quantizeScore(x):
#             if (x == 'High')| (x == 'Medium'):
#                 return 'MediumHigh'
#             else:
#                 return x

#         def group_race(x):
#             if x == "Caucasian":
#                 return 1.0
#             else:
#                 return 0.0

#         dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
#         dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
#         dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
#         dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

#         # Recode sex and race
#         dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
#         dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

#         features = ['two_year_recid',
#                     'sex', 'race',
#                     'age_cat', 'priors_count', 'c_charge_degree']

#         # Pass vallue to df
#         df = dfcutQ[features]

#         return df

#     XD_features = ['age_cat', 'c_charge_degree', 'priors_count', 'sex', 'race']
#     D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
#     Y_features = ['two_year_recid']
#     X_features = list(set(XD_features)-set(D_features))
#     categorical_features = ['age_cat', 'priors_count', 'c_charge_degree']

#     # privileged classes
#     all_privileged_classes = {"sex": [1.0],
#                               "race": [1.0]}

#     # protected attribute maps
#     all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
#                                     "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}

#     return CompasDataset(
#         label_name=Y_features[0],
#         favorable_classes=[0],
#         protected_attribute_names=D_features,
#         privileged_classes=[all_privileged_classes[x] for x in D_features],
#         instance_weights_name=None,
#         categorical_features=categorical_features,
#         features_to_keep=X_features + Y_features + D_features,
#         na_values=[],
#         metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
#                   'protected_attribute_maps': [all_protected_attribute_maps[x]
#                                                for x in D_features]},
#         custom_preprocessing=custom_preprocessing)
