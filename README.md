# $Font^2$

Il dataset viene scaricato dinamicamente dentro la cartella `blocks` 
```python
dataset = Font2('path/to/dataset', store_on_disk=True, auto_download=True)
```
in questo modo il loader scaricherà i dati un automatico `auto_download=True` e li salverà `store_on_disk=True` dentro la cartella `blocks` allo stesso livello di `fonts.json` e `words.json`
Il dataset é possibile scaricarlo manualmente dalle releases come anche i checkpoints delle resnet-18 e vgg 
