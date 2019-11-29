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


struct Shape {
    static const int MAXDIM = 10;
    int ndim = 0;
    int vals[Shape::MAXDIM];

    int nelem() {
        int n = 1;
        for(int i = 0; i < ndim; ++i) n *= vals[i];
        return n;
    }

    static Shape unify(Shape sh1, Shape sh2) {
        assert(sh1.ndim == sh2.ndim);
        return sh1;
    }

    static Shape oned(int n) {
        Shape sh;
        sh.ndim = 1;
        sh.vals[0] = n;
        return sh;
    }

    string to_str() {
        string s =  "sh[";
        for(int i = 0; i < ndim; ++i) {
            s += vals[i] + (i < ndim - 1 ? " " : "");
        }
        s += "]";
        return s;
    }
};

// only 1D arrays
struct Arr {
    Shape sh;
    float *data = nullptr;
    std::string name = "undef";

    Arr() = default;
    Arr (Shape sh, string name) : 
        sh(sh), data(new float[sh.nelem()]), name(name) {
        };

    Arr (int n, string name) : 
        sh(Shape::oned(n)), data(new float[n]), name(name) {
        };

    float & operator [](int ix) {
        cout << name << "[" << ix << "]\n";
        assert(ix < sh.nelem());
        assert(ix >= 0);
        return data[ix];
    }
    
    void print_data() {
        cout <<name <<  "[";
        for(int i = 0; i < sh.nelem(); ++i) {
            cout << data[i] << (i < sh.nelem() - 1 ? " " : "");
        }
        cout << "]\n";
    }
};

enum class ExprType {
    Add, 
    Dot, 
    Matmul, 
    PointwiseMul,
    Negate, 
    Div, 
    Tanh, 
    Sigmoid, 
    Arr, 
    Undef, 
    AllOnes, 
    AllZeros
};

struct Expr {
    ExprType ty = ExprType::Undef;
    Arr val;
    // if it's a virtual node such as AllZeros, AllOnes, this will be its length. eg. AllZeros, AllOnes
    Shape virtual_sh;

    Expr *args[10] = { nullptr };
    int npred = 0;
    int nargs = 0;
    Expr *pred[10] = { nullptr };

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
        e->val = Arr(Shape::unify(this->len(), other->len()), e->to_str());
        return e;
    }

    static Expr *pointwisemul(Expr *l,  Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::PointwiseMul;
        e->addarg(l);
        e->addarg(r);
        return e;
    }

    Expr *matmul(Expr *other) {
        Expr *e = new Expr;
        e->ty = ExprType::Matmul;
        e->addarg(this); 
        e->addarg(other);
        return e;
    }

    Expr *dot(Expr *other) {
        Expr *e = new Expr;
        e->ty = ExprType::Dot;
        e->addarg(this); 
        e->addarg(other);

        // assert(this->len() == other->len());
        e->val = Arr(1, e->to_str());
        return e;
    }

    Expr *allones(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllOnes;
        e->virtual_sh = sh;
        return e;
    }


    Expr *allzeros(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllZeros;
        e->virtual_sh = sh;
        return e;
    }

    // return the expression for the gradient with the other array
    Expr *grad(string name) {
        switch(ty) {
            case ExprType::Arr:  {
                return val.name == name ? 
                    Expr::allones(len()) : Expr::allzeros(len());
             }
            case ExprType::Add:
                return args[0]->grad(name)->add(args[1]->grad(name));
            case ExprType::Dot:
                return Expr::pointwisemul(args[0]->grad(name), args[1])->add(Expr::pointwisemul(args[0], args[1]->grad(name)));
            default: assert(false && "unimplemented");
        }
    }
    
    void force() {
        switch(ty) {
            case ExprType::Arr: return;
            case ExprType::AllOnes: return;
            case ExprType::Add:
                    args[0]->force();
                    args[1]->force();
                for(int i = 0; i < args[0]->len().nelem(); ++i) {
                    val[i] = args[0]->at(i) + args[1]->at(i);
                }
                return;
            case ExprType::Dot:
                    args[0]->force();
                    args[1]->force();
                    val[0] = 0;
                    assert(args[0]->len().ndim == 1);
                    assert(args[1]->len().ndim == 1);

                    for(int i = 0; i < args[0]->len().nelem(); ++i) {
                        val[0] += args[0]->at(i) * args[1]->at(i);
                    }
                    return;
            case ExprType::PointwiseMul:
                    args[0]->force();
                    args[1]->force();
                    for(int i = 0; i < args[0]->len().nelem(); ++i) {
                            val[i] = args[0]->at(i) * args[1]->at(i);
                    }

            default: assert(false && "unhandled"); 
        }
    }

    float at(int ix) { 
        switch(ty) {
            case ExprType::AllOnes:
                return 1;
            case ExprType::AllZeros:
                return 0;
            default:
                return val[ix];
        }
    }

    Shape len() {
        switch(ty) {
            case ExprType::Arr: return val.sh;
            case ExprType::AllOnes: return virtual_sh;
            case ExprType::AllZeros: return virtual_sh;
            case ExprType::Add: return Shape::unify(args[0]->len(), args[1]->len());
            case ExprType::PointwiseMul: return Shape::unify(args[0]->len(), args[1]->len());
            default:
                assert (false && "unhandled");
        }
    }


    string to_str() {
        switch(ty) {
            case ExprType::Arr: return val.name;
            case ExprType::AllOnes: return "11..1";
            case ExprType::AllZeros: return "00..0";
            case ExprType::Add: 
                return "(+ " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Dot: 
                return "(dot " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::PointwiseMul: 
                return "(.* " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";
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
    cout << dot->to_str();

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

    Expr *dotder = dot->grad("b");
    cout << "grad of dot wrt b:";
    cout << ": " << dotder->to_str();

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
