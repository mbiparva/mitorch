batch, channel, depth, height, width -> b d d (using pooling or anything)
Auxillary head: input is a volume
                output is batch, depth, depth (which would be softmax of first depth)
                loss is multi-label cross-entropy

create a file in models directory for Auxillary head (using nn.module, choose pooling method)
