# Languagemodels

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
