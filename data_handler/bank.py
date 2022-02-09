from data_handler.AIF360.bank_dataset import BankDataset
from data_handler.tabular_dataset import TabularDataset


class BankDataset_torch(TabularDataset):
    """Bank dataset."""

    def __init__(self, root, split='train', sen_attr='age',  group_mode=-1, influence_scores = None):
        dataset = BankDataset(root_dir=root)
        if sen_attr == 'age':
            sen_attr_idx = 0
        else:
            raise Exception('Not allowed group')

        self.num_groups = 2
        self.num_classes = 2

        super(BankDataset_torch, self).__init__(root=root, dataset=dataset, sen_attr_idx=sen_attr_idx, 
                                                split=split, group_mode=group_mode, influence_scores=influence_scores)