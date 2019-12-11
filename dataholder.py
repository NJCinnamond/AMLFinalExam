class DataHolder:

    def __init__(self, train_tags, test_tags, train_res, test_res, 
                        train_restint, test_resint, train_descript, test_descript):
        self.train_tags = train_tags
        self.test_tags = test_tags
        self.train_res = train_res
        self.test_res = test_res
        self.train_resint = train_restint
        self.test_resint = test_resint
        self.train_descript = train_descript
        self.test_descript = test_descript
    
    def __len__(self):
        return len(self.tags)
