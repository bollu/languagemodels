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
    int ndim = -42;
    DimSize vals[Shape::MAXDIM] = {-42};;

    int nelem() {
        int n = 1;
        for(int i = 0; i < ndim; ++i) n *= vals[i];
        return n;
    }

    int operator [](int dim) const {
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

    bool operator != (const Shape &other) const {
        return !(*this == other);
    }

    // lex comparison
    bool operator < (const Shape &other) const {
        if (ndim < other.ndim) return true;
        if (ndim > other.ndim) return false;

        assert(ndim == other.ndim);

        for(int i = 0; i < ndim; ++i) {
            if (vals[i] < other.vals[i]) return true;
            if (vals[i] > other.vals[i]) return false;
            assert(vals[i] == other.vals[i]);
        }

        assert(*this == other);
        return false;

    }

    static Shape unify(Shape sh1, Shape sh2) {
        Shape smaller, larger;
        if (sh1.ndim < sh2.ndim) { smaller = sh1; larger = sh2; }
        else { smaller = sh2; larger = sh1; }

        for(int i = 0; i < smaller.ndim; ++i) {
            assert(smaller[i] == larger[i]);
        }
        return larger;
    }   

    static Shape zerod() {
        Shape sh;
        sh.ndim = 0;
        return sh;
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

    string to_str() const {
        string s =  "sh[";
        for(int i = 0; i < ndim; ++i) {
            s += to_string(vals[i]) + (i < ndim - 1 ? " " : "");
        }
        s += "]";
        return s;
    }

    // remove an outermost (leftmost) dimension
    Shape removeOutermost() const {
        Shape sh;
        assert(ndim > 0);
        sh.ndim = ndim - 1;
        for(int i = 0; i < ndim - 1; ++i) {
            sh.vals[i] = vals[i+1];
        }
        return sh;
    }

    // append an outermost dimension
    Shape addOutermost(int size) const {
        Shape sh = *this;
        sh.ndim++;
        for(int i = 1; i < ndim+1; ++i) {
            sh.vals[i] = vals[i-1];
        }
        sh.vals[0] = size;
        return sh;
    }
};

// only 1D frrays
/*
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

    bool operator < (const Arr &other) const {
        // lex compare, first on name, then on shape
        return name < other.name || 
            ((name == other.name) && (sh < other.sh));
    }

    bool operator == (const Arr &other) const {
        return name == other.name && (sh == other.sh);
    }

    bool operator != (const Arr &other) const {
        return !(*this == other);
    }
};
*/

enum class ExprType {
    Sum,
    Mul,
    Undef,
    Arr,
    Index,
    Slice,
    Contract
};

struct Index {
    string name;
    Index(string name) : name(name) {};

    bool operator == (const Index &other) const { return name == other.name; }
    bool operator < (const Index &other) const { return name < other.name; }

    string to_str() const { return name; };
};

struct Expr {
    // gives type of expression
    ExprType ty = ExprType::Undef;
    Expr(ExprType ty): ty(ty) {};
    virtual string to_str() const = 0;
    // virtual Expr *diff() = 0;

    Expr *ix(Index ix);
    Expr *contract(Index ix);
    // return free indeces
    virtual set<Index> free() const = 0;

    Expr *diff(string name) const;

    string detailed_to_str() const {
        string s = to_str();
        s += "[";
        for(Index i : free()) s += i.name + ",";
        s += "]";
        return s;
    }
};

struct Slice : public Expr {
    Expr *inner;
    Index ix;

    Slice(Expr *inner, Index ix): Expr(ExprType::Index), inner(inner), ix(ix)
    {};

    string to_str() const  {
        return "(! " + inner->to_str() + " " + ix.to_str() + ")";
    }

    set<Index> free() const {
        set<Index> s = inner->free();
        s.insert(ix);
        return s;
    }
};

Expr *Expr::ix(Index ix) {
    return new Slice(this, ix);
}

struct Arr : public Expr {
    string name;
    Arr(string name) : Expr(ExprType::Arr), name(name) {};
    string to_str() const { return name; };

    set<Index> free() const {
        return set<Index>();
    }
};

struct Mul : public Expr {
    Expr *l, *r;

    Mul (Expr *l, Expr *r) : Expr(ExprType::Mul), l(l), r(r) {};

    string to_str() const {
        return "(* " + l->to_str() + " " + r->to_str() + ")";
    }

    set<Index> free() const {
        set<Index> lf = l->free();
        set<Index> rf = l->free();
        rf.insert(lf.begin(), lf.end());
        return rf;
    }
};

struct Add : public Expr {
    Expr *l, *r;

    Add (Expr *l, Expr *r) : Expr(ExprType::Mul), l(l), r(r) {};

    string to_str() const {
        return "(+ " + l->to_str() + " " + r->to_str() + ")";
    }

    set<Index> free() const {
        set<Index> lf = l->free();
        set<Index> rf = l->free();
        rf.insert(lf.begin(), lf.end());
        return rf;
    }
};

struct Contract : public Expr {
    Index ix;
    Expr *inner;
    Contract (Index ix, Expr *inner) : Expr(ExprType::Contract), ix(ix),
        inner(inner) {};

    string to_str() const {
        return "(>< " + ix.to_str() + " " + inner->to_str() +  ")";
    }

    set<Index> free() const {
        set<Index> infree = inner->free();
        auto it = infree.find(ix);
        if (it != infree.end()) infree.erase(it);
        return infree;
    }
};


Expr *Expr::contract(Index ix) {
    return new Contract(ix, this);
}

struct Stmt {
    string lhs;
    Expr *rhs;
};


void add_word_to_vocab(string w) {
    if (vocab.find(w) != vocab.end()) { return; }
    word2ix[w] = last_word_index;
    ix2word[last_word_index] = w;
    last_word_index++;
}

bool is_char_word_break(char c) {
    return c == ' ' || c== '\t' || c == '\n' || c == '.';
}

// disable LSAN. Only keep ASAN. I leak far too much memory ;) 
extern "C" int __lsan_is_turned_off() { return 1; }

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


void test_expr_dot() {
    Expr *a = new Arr("a");
    Expr *b = new Arr("b");
    Index i("i");
    Expr *mul = new Mul(a->ix(i), b->ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "***dot***\n";
    cout << dot->detailed_to_str() << "\n";
}

void test_expr_matvec() {
    Expr *a = new Arr("a");
    Expr *b = new Arr("b");
    Index i("i"), j("j");
    Expr *mul = new Mul(a->ix(i)->ix(j), b->ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "***mul***\n";
    cout << matvec->detailed_to_str() << "\n";
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

    // use_expr();
    test_expr_dot();
    test_expr_matvec();
    // test_expr_tanh_matvec();
    // test_expr_rnn_batched();
}
