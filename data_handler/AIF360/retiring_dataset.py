import os

import pandas as pd

from data_handler.AIF360.standard_dataset import StandardDataset

#
# default_mappings = {
#     'label_maps': [{1.0: True, 0.0: False}],
#     'protected_attribute_maps': [{1.0: 1, 0.0: 2}]
# }

class RetiringDataset(StandardDataset):
    """Adult Census Income Dataset.
    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, root_dir='./data/retiring_adult/',
                 label_name='PINCP',
                 favorable_classes=[True],
                 protected_attribute_names=['RAC1P'],
                 privileged_classes=[[1]],
                 instance_weights_name=None,
                 categorical_features=['COW','SCHL','MAR','OCCP','POBP','RELP'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=['NaN'], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.
        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:
            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True
            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:
            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']
            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:
            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... categorical_features=['workclass', 'education', 'marital-status',
            ... 'occupation', 'relationship', 'native-country', 'race'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})
            Note that we are now adding `race` as a `categorical_features`.
            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        # train_path = os.path.join(root_dir, 'adult.data')
        # test_path = os.path.join(root_dir, 'adult.test')

        # as given by adult.names
        column_names = [
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            'SEX',
            'RAC1P', 'PINCP']
        try:
            df = pd.read_csv(root_dir + '/retiring_adult.csv', header=0, names=column_names, skipinitialspace=False, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            from folktables import ACSDataSource, ACSIncome
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
            df = data_source.get_data(states=None, download=True)

            df = df[df['AGEP'] > 16]
            df = df[df['PINCP'] > 100]
            df = df[df['WKHP'] > 0]
            df = df[df['PWGTP'] >= 1]
            df = df[df['RAC1P'] <= 2]

            target_transform = lambda x: x > 50000
            df['PINCP'] = target_transform(df['PINCP'])

            group_transform = lambda x: x == 1
            df['RAC1P'] = group_transform(df['RAC1P'])
        
            sex_binarizer = lambda x: x == 1
            df['SEX'] = sex_binarizer(df['SEX'])

            # WKHP, AGEP Normalize
            df['AGEP'] = (df['AGEP'] - df['AGEP'].mean()) / df['AGEP'].std()
            df['WKHP'] = (df['WKHP'] - df['WKHP'].mean()) / df['WKHP'].std()

            df.to_csv("./data/retiring_adult/retiring_adult.csv", sep=',', index=False, columns=column_names)

            import sys
            sys.exit(1)

        super(RetiringDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
