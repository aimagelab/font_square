# $Font^2$

Il dataset viene scaricato dinamicamente dentro la cartella `blocks` 
```python
dataset = Font2('path/to/dataset', store_on_disk=True, auto_download=True)
loader = DataLoader(dataset, batch_size=32, num_workers=1, collate_fn=dataset.collate_fn, shuffle=False)
```

in questo modo il loader scaricherà i dati un automatico `auto_download=True` e li salverà `store_on_disk=True` dentro la cartella `blocks` allo stesso livello di `fonts.json` e `words.json`
Il dataset é possibile scaricarlo manualmente dalle releases come anche i checkpoints delle resnet-18 e vgg 
IMPORTANTISSIMO `shuffle=False` altrimenti non scarica i blocchi sequenzialmente e ci mette 10 anni
I blocchi contenuti nella release sono già shuffled e con le augmentation applicate (vedi paper) quindi basta leggerli sequenzialmente
