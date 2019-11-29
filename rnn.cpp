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
#include <math.h>

using namespace std;
using Dim = int;
using DimSize = int;

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
    DimSize vals[Shape::MAXDIM];

    int nelem() {
        int n = 1;
        for(int i = 0; i < ndim; ++i) n *= vals[i];
        return n;
    }

    int operator [](int dim) {
        assert(dim >= 0);
        assert(dim < ndim);
        return vals[dim];
    }

    bool operator == (const Shape &other) const {
        if (ndim != other.ndim) return false;
        for(int i = 0; i < ndim; ++i) {
            if(vals[i] != other.vals[i]) return false;
        }
        return true;
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

    static Shape twod(int n, int m) {
        Shape sh;
        sh.ndim = 2;
        sh.vals[0] = n;
        sh.vals[1] = m;
        return sh;
    }

    string to_str() {
        string s =  "sh[";
        for(int i = 0; i < ndim; ++i) {
            s += to_string(vals[i]) + (i < ndim - 1 ? " " : "");
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
    Arr(const Arr &other) = default;
    Arr (Shape sh, string name) : 
        sh(sh), data(new float[sh.nelem()]), name(name) {
        };

    Arr (int n, string name) : 
        sh(Shape::oned(n)), data(new float[n]), name(name) {};
    Arr (int n, int m, string name) :
        sh(Shape::twod(n, m)), data(new float[n*m]), name(name) {};

    float &operator [](int ix) {
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
    Sub,
    Dot, 
    MatMatMul, 
    MatVecMul,
    Replicate,
    PointwiseMul,
    Tanh,
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
    Expr(const Expr &other) = default;

    static Expr *arr(Arr a) {
        Expr *e = new Expr;
        e->val = a;
        e->ty = ExprType::Arr;
        return e;
    }

    static Expr* add(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::Add;
        e->addarg(l); 
        e->addarg(r);
        e->val = Arr(Shape::unify(l->sh(), r->sh()), e->to_str());
        return e;
    }

    static Expr* sub(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::Sub;
        e->addarg(l); 
        e->addarg(r);
        e->val = Arr(Shape::unify(l->sh(), r->sh()), e->to_str());
        return e;
    }

    static Expr *pointwisemul(Expr *l,  Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::PointwiseMul;
        e->addarg(l);
        e->addarg(r);
        return e;
    }

    static Expr *matmatmul(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::MatMatMul;
        e->addarg(l);
        e->addarg(r);
        return e;

    }

    static Expr *matvecmul(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::MatVecMul;
        e->addarg(l);
        assert(l->sh().ndim == 2);
        assert(r->sh().ndim == 1);
        e->addarg(r);
        return e;

    }

    static Expr *replicate(Expr *inner, Shape replicatesh) {
        Expr *e = new Expr;
        e->ty = ExprType::Replicate;
        e->addarg(inner);
        // the new shape is that of the shape we want to replicate to
        e->virtual_sh = replicatesh;
        return e;
    }


    static Expr *dot(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::Dot;
        e->addarg(l); 
        e->addarg(r);

        // assert(this->sh() == other->sh());
        e->val = Arr(1, e->to_str());
        return e;
    }

    static Expr *tanh(Expr *inner) {
        Expr *e = new Expr;
        e->ty = ExprType::Tanh;
        e->addarg(inner);
        return e;
    }

    static Expr *allones(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllOnes;
        e->virtual_sh = sh;
        return e;
    }


    static Expr *allzeros(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllZeros;
        e->virtual_sh = sh;
        return e;
    }

    // return the expression for the gradient with the other array
    Expr *grad(Arr dx) {
        switch(ty) {
            case ExprType::Arr:  {
                return val.name == dx.name ? 
                    Expr::allones(sh()) : Expr::allzeros(sh());
             }
            case ExprType::Add:
                return Expr::add(args[0]->grad(dx), args[1]->grad(dx));
            case ExprType::Dot:
                return Expr::add(Expr::pointwisemul(args[0]->grad(dx), args[1]),
                        Expr::pointwisemul(args[0], args[1]->grad(dx)));
            // (1 - tanh^2 X) .* X'
            case ExprType::Tanh: {
                Expr *dtan = Expr::sub(Expr::allones(sh()), Expr::pointwisemul(new Expr(*this), new Expr(*this)));
                // derivative of the inner computation
                Expr *dinner = args[0]->grad(dx);
                return Expr::pointwisemul(dtan, dinner);
             }
            case ExprType::MatMatMul: {
                    assert(false && "need to implement replicate");
                   return Expr::add(Expr::matmatmul(args[0]->grad(dx), args[1]),
                           Expr::matmatmul(args[0], args[1]->grad(dx)));

               }
            case ExprType::MatVecMul: {
                   // TODO!! What is the justification for the "fit shape"??
                   // (d/dN(M x) = M [dx/dN] + [dM/dy]
                   return Expr::add(
                           Expr::replicate(Expr::matvecmul(args[0]->grad(dx), args[1]), dx.sh),
                           Expr::replicate(Expr::matvecmul(args[0], args[1]->grad(dx)), dx.sh)) ;
               }
            default: assert(false && "unimplemented");
        }
    }
    
    void force() {
        switch(ty) {
            case ExprType::Arr: return;
            case ExprType::AllOnes: return;
            case ExprType::Add: {
                    args[0]->force();
                    args[1]->force();
                for(int i = 0; i < args[0]->sh().nelem(); ++i) {
                    val[i] = args[0]->at(i) + args[1]->at(i);
                }
                return;
            }
            case ExprType::Sub: {
                    args[0]->force();
                    args[1]->force();
                for(int i = 0; i < args[0]->sh().nelem(); ++i) {
                    val[i] = args[0]->at(i) - args[1]->at(i);
                }
                return;
            }
            case ExprType::Dot:
                    args[0]->force();
                    args[1]->force();
                    val[0] = 0;
                    assert(args[0]->sh().ndim == 1);
                    assert(args[1]->sh().ndim == 1);

                    for(int i = 0; i < args[0]->sh().nelem(); ++i) {
                        val[0] += args[0]->at(i) * args[1]->at(i);
                    }
                    return;
            case ExprType::PointwiseMul:
                    args[0]->force();
                    args[1]->force();
                    for(int i = 0; i < args[0]->sh().nelem(); ++i) {
                            val[i] = args[0]->at(i) * args[1]->at(i);
                    }
                    return;

            case ExprType::Tanh: 
                    args[0]->force();
                    for(int i = 0; i < args[0]->sh().nelem(); ++i) {
                        val[i] = tanhf(args[0]->at(i));
                    }


            case ExprType::MatMatMul: {
                    args[0]->force();
                    args[1]->force();

                    int M = args[0]->sh()[0];
                    int N = args[0]->sh()[1];
                    assert(N == args[1]->sh()[0]);
                    int O = args[1]->sh()[1];

                    for(int i = 0; i < M; ++i) {
                        for(int j = 0; j < O; ++j) {
                            val[i*M+j] = 0;
                            for(int k = 0; k < N; ++k ) {
                                val[i*M+j] += args[0]->at(i*N+k) * args[1]->at(k*M+j);
                            }

                        }
                    }
                    return;
           }

            case ExprType::MatVecMul: {
                    args[0]->force();
                    args[1]->force();

                    int M = args[0]->sh()[0];
                    int N = args[0]->sh()[1];
                    assert(N == args[1]->sh()[0]);

                    for(int i = 0; i < M; ++i) {
                        val[i] = 0;
                        for(int j = 0; j < N; ++j) {
                                val[j] += args[0]->at(i*N+j) * args[1]->at(j);
                            }

                    }
                    return;
           }

            default: cerr << to_str(); assert(false && "unhandled"); 
        }
    }

    float at(int ix) { 
        switch(ty) {
            case ExprType::AllOnes:
                return 1;
            case ExprType::AllZeros:
                return 0;
            case ExprType::Replicate:
                // find the value of the index wrt the replication
                // a[i, j, k] -> a[i]
                return args[0]->at(ix % args[0]->sh().nelem());

            default:
                return val[ix];
        }
    }

    Shape sh() {
        switch(ty) {
            case ExprType::Arr: return val.sh;
            case ExprType::AllOnes: return virtual_sh;
            case ExprType::AllZeros: return virtual_sh;
            case ExprType::Tanh: return args[0]->sh();
            case ExprType::Replicate: return virtual_sh;
            case ExprType::Add: return Shape::unify(args[0]->sh(), args[1]->sh());
            case ExprType::Sub: return Shape::unify(args[0]->sh(), args[1]->sh());
            case ExprType::PointwiseMul: return Shape::unify(args[0]->sh(), args[1]->sh());
            case ExprType::MatMatMul: return args[0]->sh();
            case ExprType::MatVecMul: return args[1]->sh();
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

            case ExprType::Sub: 
                return "(- " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Dot: 
                return "(dot " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Tanh:
                return "(tanh " + args[0]->to_str() + ")";

            case ExprType::PointwiseMul: 
                return "(.* " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::MatMatMul: 
                return "(@mm " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::MatVecMul: 
                return "(@mv " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Replicate: 
                return "(replicate " + virtual_sh.to_str() + " " + 
                    args[0]->to_str()+ ")";
            default:
                assert(false && "unrechable");

        }
    }
    
};

bool isexprconstant(const Expr *e) {
    switch(e->ty) {
        case ExprType::AllOnes:
        case ExprType::AllZeros:
            return true;
        default: return false;
    }
}

// move all constants to the left. If both params are constants, then fold
Expr *constantfold(Expr *e) {
    switch(e->ty) {
        case ExprType::PointwiseMul:
            if (e->args[0]->ty == ExprType::AllZeros || 
                    e->args[1]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh());
            }
            if (e->args[0]->ty == ExprType::AllOnes) {
                return constantfold(e->args[1]);
            }
            if (e->args[1]->ty == ExprType::AllOnes) {
                return constantfold(e->args[0]);
            }
            return Expr::pointwisemul(constantfold(e->args[0]), constantfold(e->args[1]));

        case ExprType::MatVecMul:
            if (e->args[0]->ty == ExprType::AllZeros || 
                    e->args[1]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh());
            }
            if (e->args[0]->ty == ExprType::AllOnes) {
                return constantfold(e->args[1]);
            }
            if (e->args[1]->ty == ExprType::AllOnes) {
                return constantfold(e->args[0]);
            }
            return Expr::matvecmul(constantfold(e->args[0]), constantfold(e->args[1]));



        case ExprType::Add:
            if(e->args[0]->ty == ExprType::AllZeros && 
                    e->args[1]->ty == ExprType::AllOnes) {
                return Expr::allones(e->sh());
            }
            if(e->args[0]->ty == ExprType::AllZeros) {
                return constantfold(e->args[1]);
            }
            if(e->args[1]->ty == ExprType::AllZeros) {
                return constantfold(e->args[0]);
            }
            return Expr::add(constantfold(e->args[0]), constantfold(e->args[1]));

        case ExprType::Replicate:
            if (e->args[0]->sh() == e->virtual_sh) {
                return constantfold(e->args[0]);
            } 
            else if (e->args[0]->ty == ExprType::AllOnes) {
                return Expr::allones(e->virtual_sh);
            }
            else if (e->args[0]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->virtual_sh);
            }
            else {
                return Expr::replicate(constantfold(e->args[0]), e->virtual_sh);
            }


        default: return e;
    }
}

void use_expr() { 
    {
    const int N = 3;
    Arr arra = Arr(N, "a");
    Arr arrb = Arr(N, "b");
    for(int i = 0; i < N; ++i) arra[i] = i;
    for(int i = 0; i < N; ++i) arrb[i] = i*2;
    Expr *a = Expr::arr(arra);
    Expr *b = Expr::arr(arrb);

    Expr *add = Expr::add(a, b);
    Expr *dot = Expr::dot(add, b);
    cout << dot->to_str();

    // force this thunk.
    dot->force();

    cout << "arra:\n"; arra.print_data();
    cout << "\narrb:"; arrb.print_data();
    cout << "\nadd:"; add->val.print_data();

    cout << "\ndot: "; cout << dot->to_str();
    cout << "\ndot: "; dot->val.print_data();

    Expr *dotder = dot->grad(arrb);
    cout << "grad of dot wrt b:";
    cout << ": " << dotder->to_str();

    for(int i = 0; i < 3; ++i) {
        dotder = constantfold(dotder);
    }

    cout << " | simpl " << dotder->to_str();
    }

    {

        const int M = 3;
        const int N = 3;
        Arr arrm= Arr(M, N, "m");
        Arr arrv = Arr(N, "v");
        for(int i = 0; i < M; ++i) 
            for(int j = 0; j < N; ++j)
                arrm[i*M+j] = i == j ? 1 : 0;
        for(int i = 0; i < N; ++i) arrv[i] = i;

        Expr *m = Expr::arr(arrm);
        Expr *v = Expr::arr(arrv);

        Expr *tanhv = Expr::tanh(v);
        Expr *dot = Expr::matvecmul(m, tanhv);
        cout << "\ndot:" << dot->to_str();

        Expr *dot_grad_m = dot->grad(arrm);
        for(int i = 0; i < 6; ++i) {
            cout << "\n" << i << "|dot->grad[m]:" << dot_grad_m->to_str();
            dot_grad_m = constantfold(dot_grad_m);
        }

        cout << "\n\n";
        Expr *dot_grad_v = dot->grad(arrv);
        for(int i = 0; i < 6; ++i) {
        cout << "\n" << i << "| dot->grad[v]:" << dot_grad_v->to_str();
            dot_grad_v = constantfold(dot_grad_v);
        }
        // cout << "\ndot der wrt a:" << dot->grad("b");
    }

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
