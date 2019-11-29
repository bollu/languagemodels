#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <map>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

int last_word_index = 0;
set<string> vocab;
map<string, int> word2ix;
map<int, string> ix2word;

// end of sentence is denoted by -1
int num_sentences = 0;
static const long MAX_SENTENCES = 1e5;
static const long MAX_SENTENCE_LEN = 1e2;
int ss[MAX_SENTENCES][MAX_SENTENCE_LEN];
vector<vector<int> > sentences;

// only 1D arrays
struct Arr {
    float *data = nullptr;
    float *der = nullptr;
    std::string name = "undef";
    int len = 0;

    Arr() = default;
    Arr (int len, string name) : 
        len(len), data(new float[len]), der(new float[len]), name(name) {
            // these variables are differentiable.
            for(int i = 0; i < len; ++i) der[i] = 1.0;
        };

    float & operator [](int ix) {
        assert(ix <= len);
        assert(ix >= 0);
        return data[ix];
    }
    
    void print_data() {
        cout <<name <<  "[";
        for(int i = 0; i < len; ++i) {
            cout << data[i] << (i < len - 1 ? " " : "");
        }
        cout << "]\n";
    }
};

enum class ExprType {
    Add, Dot, Matmul, Negate, Div, Tanh, Sigmoid, Arr, Undef
};

struct Expr {
    ExprType ty = ExprType::Undef;
    Arr val;
    Expr *args[10] = { nullptr };
    int npred = 0;
    int nargs = 0;
    Expr *pred[10] = { nullptr };
    // gradients for all predecessors
    Arr *grad[10] = { nullptr };

    void addarg(Expr *e) {
        assert(nargs < 10);
        args[nargs++] = e;
        assert(e->npred < 10);
        e->pred[e->npred++] = this;
    }

    Expr() = default;

    static Expr *arr(Arr a) {
        Expr *e = new Expr;
        e->val = a;
        e->ty = ExprType::Arr;
        return e;
    }

    Expr *add(Expr *other) {
        Expr *e = new Expr;
        e->ty = ExprType::Add;
        e->addarg(this); 
        e->addarg(other);
        assert(this->len() == other->len());
        e->val = Arr(this->len(), e->get_name());
        return e;
    }

    Expr *matmul(Expr *other) {
        Expr *e = new Expr;
        e->ty = ExprType::Matmul;
        e->addarg(this); 
        e->addarg(other);
    }

    Expr *dot(Expr *other) {
        Expr *e = new Expr;
        e->ty = ExprType::Dot;
        e->addarg(this); 
        e->addarg(other);

        assert(this->len() == other->len());
        e->val = Arr(1, e->get_name());
        return e;
    }
    
    void force() {
        switch(ty) {
            case ExprType::Arr: return;
            case ExprType::Add:
                    args[0]->force();
                    args[1]->force();
                for(int i = 0; i < args[0]->len(); ++i) {
                    val[i] = args[0]->at(i) + args[1]->at(i);
                }
                return;
            case ExprType::Dot:
                    args[0]->force();
                    args[1]->force();
                    val[0] = 0;
                    for(int i = 0; i < args[0]->len(); ++i) {
                        val[0] += args[0]->at(i) * args[1]->at(i);
                    }
                    return;

            default: assert(false && "unhandled"); 
        }
    }

    float &operator[] (int ix) {
        return val[ix];
    }

    float &at(int ix) { return val[ix]; }

    int len() {
        switch(ty) {
            case ExprType::Arr: return val.len;
            case ExprType::Add: return args[0]->len();
            default:
                assert (false && "unhandled");
        }
    }


    string get_name() {
        switch(ty) {
            case ExprType::Arr: return val.name;
            case ExprType::Add: 
                return "(+ " + args[0]->get_name() + " " + 
                        args[1]->get_name() + ")";

            case ExprType::Dot: 
                return "(dot " + args[0]->get_name() + " " + 
                        args[1]->get_name() + ")";
            default:
                assert(false && "unrechable");

        }
    }
    
};

void use_expr() {
    const int N  = 3;
    Arr arra = Arr(N, "a");
    Arr arrb = Arr(N, "b");
    for(int i = 0; i < N; ++i) arra[i] = i;
    for(int i = 0; i < N; ++i) arrb[i] = i*2;
    Expr *a = Expr::arr(arra);
    Expr *b = Expr::arr(arrb);

    Expr *add = a->add(b);
    Expr *dot = b->dot(add);
    cout << dot->get_name();

    // force this thunk.
    cout << "\n";
    dot->force();

    arra.print_data();
    cout << "\n";
    arrb.print_data();
    cout << "\n";
    add->val.print_data();

    cout << "\n";
    dot->val.print_data();

}

void add_word_to_vocab(string w) {
    if (vocab.find(w) != vocab.end()) { return; }
    word2ix[w] = last_word_index;
    ix2word[last_word_index] = w;
    last_word_index++;
}

bool is_char_word_break(char c) {
    return c == ' ' || c== '\t' || c == '\n' || c == '.';
}

// consume whitespace and get the next word, or empty string if run out.
string parse_word(FILE *f) {
    char c;
    do {
        c = fgetc(f);
    } while(c != EOF && is_char_word_break(c));

    if (c == EOF) { return ""; }
    ungetc(c, f);

    string w;
    while(1) {
        c = fgetc(f); 
        if (is_char_word_break(c)) break;
        w += c;
    } 

    return w;
}

bool consume_till_sentence_end_or_word(FILE *f) {
    while(1) {
        char c = fgetc(f);
        if (c == '.') { return true; }
        if (c == ' ' || c == '\t' || c == '\n')  { continue; }
        ungetc(c, f);
        return false;

    }
}

vector<int> parse_sentence(FILE *f) {
    vector<int> sentence;
    while(1) {
        // returns true if '.' was consumed
        if (consume_till_sentence_end_or_word(f)) { return sentence; }
        string w = parse_word(f);
        if (w == "") { return sentence; }
        auto it = word2ix.find(w);
        assert(it != word2ix.end());
        sentence.push_back(it->second);
    }
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: <input-path>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");

    // construct vocab
    while(1) {
        string w = parse_word(f);
        cout << "found word: " << w << "\n";
        if (feof(f)) { break; }
        add_word_to_vocab(w);
    }
    rewind(f);
    while(1) {
        vector<int> sentence = parse_sentence(f);
        sentences.push_back(sentence);
        if (feof(f)) { break; }
    }
    fclose(f);

    for(auto it: word2ix) {
        cout << "|" << it.first << "|->" << it.second << "\n";
    }

    cout << "SENTENCES:\n";
    for(auto s: sentences) {
        for (auto w : s) {
            cout << w << " ";
        }
        cout << "\n";
    }

    use_expr();

}
