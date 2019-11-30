# Languagemodels

## `rnn.cpp`

make each array expression "implicit" by default, and then create a tensor
type that instantiates it.

Currently writing a DSL for array based derivatives. Small, extensible API
that's meant to generate close to the metal code. Will generate code
using [`facebookresearch/TensorComprehensions`](https://github.com/facebookresearch/TensorComprehensions)

I keep whining about how PyTorch is slow and TensorFlow makes me want to kill
myself, so I might as well be the change I want to see in the world
with regards to libraries for machine learning...

## `lstm.py`

jointly train the language model with the word embedding. Use `--train` to
train, `--test` to test.

Example of perfect embeddings on the sentence `the quick brown fox jumps over the lazy dog`:
- `!` for predict.
- `~` for similarity.

```
╰─$ ./lstm.py repl 
>! the quick brown                                                                                                                                                                                          
['fox']
>! quick brown fox                                                                                                                                                                                          
['jumps']
>! brown fox jumps                                                                                                                                                                                          
['over']
>! over the lazy                                                                                                                                                                                            
['dog']
>~ dog                                                                                                                                                                                                      
                 dog 1.00 
                lazy 0.45 
               jumps 0.41 
               brown 0.10 
                over -0.04 
                 fox -0.08 
               quick -0.16 
                 the -0.98 
>! the                                                                                                                                                                                                      
['the']
>~ the                                                                                                                                                                                                      
                 the 1.00 
                over 0.20 
                 fox 0.11 
               quick 0.08 
               brown -0.01 
               jumps -0.40 
                lazy -0.52 
                 dog -0.98 
```
