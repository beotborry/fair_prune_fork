import pandas as pd
from data_handler.AIF360.credit_dataset import CreditDataset
#from data_handler.tabular_classes.credit_dataset import CreditDataset
from data_handler.tabular_dataset import TabularDataset


class CreditDataset_torch(TabularDataset):
    """Adult dataset."""

    def __init__(self, root, split='train', sen_attr='sex', group_mode=-1, influence_scores=None):

        dataset = CreditDataset(root_dir=root)
        if sen_attr == 'sex':
            sen_attr_idx = 1
        else:
            raise Exception('Not allowed group')

        self.num_groups = 2
        self.num_classes = 2

        super(CreditDataset_torch, self).__init__(root=root, dataset=dataset, sen_attr_idx=sen_attr_idx, 
                                                  split=split, group_mode=group_mode, influence_scores=influence_scores)
