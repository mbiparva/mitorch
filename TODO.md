- Rework auxillary head with pure PyTorch.
- Add loss function in netwrapper.loss.
- Register loss function with decorator.
- Rename `auxillary_head.py` to `self_supervised_models.py`.
- Use and test already developed CrossEntropyLoss first.
- Loss function is cross-entropy loss.
- loss input is 'b', 'd' and output is float. 
- auxiallry head model that gets registered (with name SliceOrderingNet):
    input: nn.module
    in create: net = nn.module()
    head = auxillary_head(net)
    retrun auxillary_head(net(model_input))
