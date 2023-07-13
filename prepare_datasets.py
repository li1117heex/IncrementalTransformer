from datasets import load_dataset

def load_split(name,namesave):
    dataset = load_dataset(name)
    if 'test' not in dataset.keys():
        datasetsplit = dataset['train'].train_test_split(test_size=0.1)
        datasetsplit['validation'] = datasetsplit['test']
        datasetsplit['test'] = dataset['validation']
        dataset=datasetsplit
    dataset.save_to_disk(namesave)

load_split('squad','squadsplit')
load_split('squadv2','squadv2split')
load_split('race','race')
load_split('boolq','boolqsplit')
load_split('quac','quacqsplit')