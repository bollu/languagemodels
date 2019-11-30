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

    string to_str() {
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
};

enum class ExprType {
    Add,
    Sub,
    Dot, 
    MatMatMul, 
    MatVecMul,
    Replicate,
    PointwiseMul,
    // constant aray
    Constant,
    Tanh,
    Arr, 
    Undef, 
    AllOnes, 
    AllZeros,
    // select a sub array from a given array. Used for embedding indexing.
    Index,
    // define a batch dimensions
    Batch,
    // unbatch a batch dimension to specify final computation of gradient update
    Unbatch,
    // let bindings. 
    Let, 
};

struct Expr {
    ExprType ty = ExprType::Undef;
    Arr val;
    // if it's a virtual node such as AllZeros, AllOnes, this will be its
    // length.
    Shape virtual_sh;

    Expr *args[10] = { nullptr };
    int npred = 0;
    int nargs = 0;
    Expr *pred[10] = { nullptr };

    // constant float for Constant
    float constval;

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
        if (l->ty == ExprType::AllZeros) return r;
        if (r->ty == ExprType::AllZeros) return l;

        Shape shunified = Shape::unify(l->sh(), r->sh());
        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

        Expr *e = new Expr;
        e->ty = ExprType::Add;
        e->addarg(l); 
        e->addarg(r);
        e->val = Arr(shunified, e->to_str());
        return e;
    }

    static Expr* sub(Expr *l, Expr *r) {
        Shape shunified = Shape::unify(l->sh(), r->sh());
        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

        Expr *e = new Expr;
        e->ty = ExprType::Sub;
        e->addarg(l); 
        e->addarg(r);
        e->val = Arr(Shape::unify(l->sh(), r->sh()), e->to_str());
        return e;
    }

    static Expr *pointwisemul(Expr *l,  Expr *r) {
        Shape shunified = Shape::unify(l->sh(), r->sh());
        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

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
        assert(l->sh().ndim == 2);
        assert(r->sh().ndim == 2);
        assert(l->sh()[1] == r->sh()[0]);
        e->val = Arr(Shape::twod(l->sh()[0], r->sh()[1]), e->to_str());
        return e;
    }

    static Expr *matvecmul(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::MatVecMul;
        e->addarg(l);
        e->addarg(r);

        assert(l->sh().ndim == 2);
        assert(r->sh().ndim == 1);
        assert(l->sh()[1] == r->sh()[0]);
        e->val = Arr(r->sh(), e->to_str());
        return e;

    }

    static Expr *replicate(Expr *inner, Shape replicatesh) {
        // constant fold directly in the replicate()
        if (inner->sh() == replicatesh) return inner;

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
        e->val = Arr(inner->sh(), e->to_str());
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

    static Expr *index(Expr *arr, Expr *index) {
        // indexing array is 1D. Contains offsets into array.
        assert(index->sh().ndim == 1);

        Expr *e = new Expr;
        e->addarg(arr);
        e->addarg(index);
        e->ty = ExprType::Index;


        // array shape: 3 3 [1 0 0; 0 1 0; 0 0 1]
        // index set: 1 [1 2]
        // out[index[0]] = [1 0 0]
        // out[index[1]] = [0 1 0]
        // size of out: inner * len(index)
        e->virtual_sh =
            arr->sh().removeOutermost().addOutermost(index->sh().vals[0]);
        e->val = Arr(e->virtual_sh, e->to_str());
        return e;
        
    }

    static Expr *batch(Expr *arr) {
        Expr *e = new Expr;
        e->addarg(arr);
        e->ty = ExprType::Batch;
        e->val = Arr(arr->sh(), e->to_str());
        return e;
    }

    static Expr *unbatch(Expr *arr, int batchsize) {
        Expr *e = new Expr;
        e->addarg(arr);
        e->ty = ExprType::Unbatch;
        // TODO: check that this matches the inner Batch() sizes.
        e->virtual_sh = arr->sh().addOutermost(batchsize); 
        e->val = Arr(e->virtual_sh, e->to_str());
        return e;
    }

    static Expr *constant(float c) {
        Expr *e = new Expr;
        e->ty = ExprType::Constant;
        e->constval = c;
        e->virtual_sh = Shape::zerod();
        e->val = Arr(e->virtual_sh, e->to_str());
        return e;
    }

    static Expr *let(Arr lhs, Expr *rhs, Expr *in) {
        Expr *e = new Expr;
        e->ty = ExprType::Let;
        e->val = lhs;
        e->args[0] = rhs;
        e->args[1] = in;
        assert(e->val.sh == e->args[0]->sh());
        return e;
    }  

    Expr *grad(Arr dx) {
        return grad_(dx, dx.sh, {{dx.name, Expr::allones(dx.sh)}});
    }


    // return the expression for the gradient with the other array
    Expr *grad_(Arr dx, Shape outsh, map<string, Expr *> dermap) {
        switch(ty) {
            case ExprType::Let: {
                Arr darr = Arr(val.sh, "d" + val.name);
                Expr *darrval =  args[0]->grad_(dx, val.sh, dermap);

                // construct the derivative of dx wrt to the knowledge that the
                // derivative of the let is darr.
                dermap[darr.name] = darrval;
                Expr *inner = args[1]->grad_(dx, outsh, dermap);

                return Expr::let(darr, darrval, inner);
            }
            case ExprType::Arr:  {
                // find this array in the derivative map.
                auto it = dermap.find(this->val.name);                
                 // if it's in the map, return the value
                if (it != dermap.end()) { return new Expr(*it->second); }
                else { return Expr::allzeros(sh()); }

             }
            case ExprType::Add:
                return Expr::add(args[0]->grad_(dx, outsh, dermap), args[1]->grad_(dx, outsh, dermap));
            case ExprType::Sub:
                return Expr::sub(args[0]->grad_(dx, outsh, dermap), args[1]->grad_(dx, outsh, dermap));
            case ExprType::Dot:
                return Expr::add(Expr::pointwisemul(args[0]->grad_(dx, outsh, dermap), args[1]),
                        Expr::pointwisemul(args[0], args[1]->grad_(dx, outsh, dermap)));
            // (1 - tanh^2 X) .* X'
            case ExprType::Tanh: {
                Expr *dtan = Expr::sub(Expr::allones(sh()), Expr::pointwisemul(new Expr(*this), new Expr(*this)));
                // derivative of the inner computation
                Expr *dinner = args[0]->grad_(dx, args[0]->sh(), dermap);
                return Expr::pointwisemul(dtan, dinner);
             }
            case ExprType::MatMatMul: {
                    assert(false && "need to implement replicate");
                   return Expr::add(Expr::matmatmul(args[0]->grad_(dx, args[0]->sh(), dermap), args[1]),
                           Expr::matmatmul(args[0], args[1]->grad_(dx, args[1]->sh(), dermap)));

               }
            case ExprType::MatVecMul: {
                   // TODO!! What is the justification for the "fit shape"??
                   // (d/dN(M x) = M [dx/dN] + [dM/dy]
                   return Expr::add(
                           Expr::replicate(Expr::matvecmul(args[0]->grad_(dx, args[0]->sh(), dermap), args[1]), outsh),
                           Expr::replicate(Expr::matvecmul(args[0], args[1]->grad_(dx, args[1]->sh(), dermap)), outsh));
               }
            case ExprType::Batch: {
                return Expr::batch(args[0]->grad_(dx, args[0]->sh(), dermap));
            }

            default:
                cerr << "\nUnimplemented gradient:\n|" << to_str() << "|\n";
                assert(false && "unimplemented");
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
                    return;


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

            case ExprType::Index: {
                args[0]->force();
                args[1]->force();
                // args[0] == array.
                // args[1] == index set
                for(int i = 0; i < args[1]->sh().nelem(); ++i) {
                    const int ix = args[1]->val[i];                    
                    const int stride = args[0]->sh().removeOutermost().nelem();
                    // array shape: 3 3 [1 0 0; 0 1 0; 0 0 1]
                    // index set: 1 [1 2]
                    // out[index[0]] = [1 0 0]
                    // out[index[1]] = [0 1 0]
                    // size of out: inner * len(index)
                    // hopefully GCC optimises this into a memcpy(...)
                    for(int j = 0; j < stride; ++i) {
                        val[stride*i + j] = args[0]->val[stride*ix+j];
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
            case ExprType::Index: return virtual_sh;
            case ExprType::Tanh: return args[0]->sh();
            case ExprType::Replicate: return virtual_sh;
            case ExprType::Batch: return args[0]->sh().removeOutermost();
            case ExprType::Unbatch: return virtual_sh;
            case ExprType::Constant: return Shape::zerod(); 
            case ExprType::Add: 
                return Shape::unify(args[0]->sh(), args[1]->sh());
            case ExprType::Sub: 
                return Shape::unify(args[0]->sh(), args[1]->sh());
            case ExprType::PointwiseMul:
                return Shape::unify(args[0]->sh(), args[1]->sh());
            // TODO: find shape of contraction.
            case ExprType::MatMatMul:
                return Shape::twod(args[0]->sh()[0], args[1]->sh()[1]);
            case ExprType::MatVecMul:
                return Shape::oned(args[0]->sh()[0]);

            default:
                assert (false && "unimplemented sh()");
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

            case ExprType::Index: 
                return "(!! " + args[0]->to_str() + " " +
                            args[1]->to_str() + ")";
            case ExprType::Batch:
                return "(batch " + args[0]->to_str() + ")";
            case ExprType::Unbatch:
                return "(unbatch " + to_string(virtual_sh[0]) + " " + 
                    args[0]->to_str() + ")";
            case ExprType::Constant:
                return "(constant " + to_string(constval) + ")";
            case ExprType::Let:
                return "(let " + val.name + " := " + args[0]->to_str() +  
                    " in " + args[1]->to_str() + ")";
            default:
                assert(false && "unimplemented to_str()");

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

        case ExprType::Sub:
            if(e->args[1]->ty == ExprType::AllZeros) {
                return constantfold(e->args[0]);
            }
            return Expr::sub(constantfold(e->args[0]), constantfold(e->args[1]));

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

        case ExprType::Batch:
            if (e->args[0]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh());
            }
            else if (e->args[0]->ty == ExprType::AllOnes) {
                return Expr::allones(e->sh());
            }
            return Expr::batch(constantfold(e->args[0]));

        default: return e;
    }
}

void use_expr() { 
    {
        cout << "Let bindings\n\n";

        const int N = 3;
        Arr arra = Arr(N, "a");
        Arr asq = Arr(N, "asq");

        Expr *afour = Expr::add(Expr::arr(asq), Expr::arr(asq));
        afour = Expr::let(asq, Expr::add(Expr::arr(arra), Expr::arr(arra)), afour);
        cout << "a^4: " << afour->to_str() << "\n\n";
        cout << "a^4->grad[a]: " << afour->grad(arra)->to_str() << "\n\n";
    }
    {
    const int N = 3;
    Arr arra = Arr(N, "a");
    Arr arrb = Arr(N, "b");
    for(int i = 0; i < N; ++i) arra[i] = i;
    for(int i = 0; i < N; ++i) arrb[i] = i*2;
    Expr *a = Expr::arr(arra);
    Expr *b = Expr::arr(arrb);

    Expr *add = Expr::add(a, b);
    Expr *dot = Expr::dot(a, b);
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

        Expr *v2 = Expr::matvecmul(m, v);
        Expr *tanhv2 = Expr::tanh(v2);
        cout << "\ndot:" << tanhv2->to_str();
        tanhv2->force();

        cout << "\narrm:"; arrm.print_data();
        cout << "arrv:"; arrv.print_data();
        cout << "v2:"; v2->val.print_data();
        cout << "tanhv2:"; tanhv2->val.print_data();

        Expr *out_grad_m = tanhv2->grad(arrm);
        for(int i = 0; i < 6; ++i) {
            cout << "\n" << i << "|out->grad[m]:" << out_grad_m->to_str();
            out_grad_m = constantfold(out_grad_m);
        }

        cout << "\n\n";
        Expr *out_grad_v = tanhv2->grad(arrv);
        for(int i = 0; i < 6; ++i) {
        cout << "\n" << i << "| out->grad[v]:" << out_grad_v->to_str();
            out_grad_v = constantfold(out_grad_v);
        }
        // cout << "\ndot der wrt a:" << dot->grad("b");
    } 

    {
        cout << "\n\n\nRNN Computation\n\n\n";
        // joint modelling of words and corpus
        static const int windowsize = 3;
        static const int embedsize = 4;
        static const int hiddensize = 10;
        vector<int> sentence;
        vector<Arr> embeds;
        Expr *inputs[windowsize];
        Expr *hiddens[windowsize+1];

        Arr H2H = Arr(hiddensize, hiddensize, "H2H");
        Arr I2H = Arr(hiddensize, embedsize, "I2H");
        Arr H2HBias = Arr(hiddensize, "H2HBias");

        Arr H2O = Arr(embedsize, hiddensize, "H2O");
        Arr H2OBias = Arr(embedsize, "H2OBias");

        for(int i = 0; i < (int)vocab.size(); ++i) {
            embeds[i] = Arr(embedsize, ix2word[i]);
        }

        for(int i = 0; i < windowsize; ++i) {
            inputs[i] = Expr::arr(Arr(embedsize, "i" + std::to_string(i)));
        }

        hiddens[0] = Expr::arr(Arr(hiddensize, "hinit"));
        for(int i = 1; i <= windowsize; ++i) {
            hiddens[i] = Expr::tanh(Expr::add(Expr::matvecmul(Expr::arr(I2H), inputs[i-1]),
                        Expr::matvecmul(Expr::arr(H2H), hiddens[i-1])));
        }

        Expr *out = Expr::tanh(Expr::add(Expr::matvecmul(Expr::arr(H2O), hiddens[windowsize]), Expr::arr(H2OBias)));
        cout << "RNN output: " << out->to_str() << "\n\n";


    }

    
    {
        cout << "\n\n\nRNN Computation with batching\n\n\n";
        // joint modelling of words and corpus
        static const int batchsize = 2;
        static const int windowsize = 1;
        static const int embedsize = 4;
        static const int hiddensize = 10;
        vector<int> sentence;
        vector<Arr> embeds;

        
        Arr H2H = Arr(hiddensize, hiddensize, "H2H");
        Arr I2H = Arr(hiddensize, embedsize, "I2H");
        Arr H2HBias = Arr(hiddensize, "H2HBias");

        Arr H2O = Arr(embedsize, hiddensize, "H2O");
        Arr H2OBias = Arr(embedsize, "H2OBias");

        for(int i = 0; i < (int)vocab.size(); ++i) {
            embeds[i] = Arr(embedsize, ix2word[i]);
        }

        // inputs: batchsize x embedsize
        Expr *inputs[windowsize];
        for(int i = 0; i < windowsize; ++i) {
            inputs[i] = Expr::arr(Arr(batchsize, embedsize, "i_batched" + std::to_string(i)));
        }

        // outputs array, batchsize x embedsize
        Arr outputs = Arr(batchsize, embedsize, "output_batched");

        Expr *hiddens[windowsize+1];

        // create the compute kernel
        hiddens[0] = Expr::arr(Arr(hiddensize, "hinit"));
        for(int i = 1; i <= windowsize; ++i) {
            hiddens[i] = Expr::tanh(Expr::add(Expr::matvecmul(Expr::arr(I2H), Expr::batch(inputs[i-1])),
                        Expr::matvecmul(Expr::arr(H2H), hiddens[i-1])));
        }

        // should be batchsize x embedsize, but our kernel pretends it is embedsize
        Expr *predict = Expr::tanh(Expr::add(Expr::matvecmul(Expr::arr(H2O), hiddens[windowsize]), Expr::arr(H2OBias)));

        cout << "RNN prediction:\n" << predict->to_str();

        // full loss
        Expr *loss = Expr::sub(Expr::batch(Expr::arr(outputs)), predict);
        cout << "\n\nRRN loss:\n" << loss->to_str();

        // find derivative of loss wrt H2H, H2O, I2H, H2OBias
        Expr *H2Hgrad = loss->grad(H2H);
        for(int i = 0; i < 6; ++i) {
            cout << "\n" << i << "| out->grad[H2H]:" << H2Hgrad->to_str();
                H2Hgrad = constantfold(H2Hgrad);
        }
        cout << "\n\n";

        
        Expr *H2Ograd = loss->grad(H2O);
        cout << "\n\nloss->grad[H2O]: " << H2Ograd->to_str();
        
        Expr *I2Hgrad = loss->grad(I2H);
        cout << "\n\nloss->grad[I2H]: " << I2Hgrad->to_str();
        
        Expr *H2OBiasgrad = loss->grad(H2OBias);
        cout << "\n\nloss->grad[H2OBias]: " << H2OBiasgrad->to_str();
        
        
        // scaled gradient
        const float learningrate = 1e-2;
        Expr *H2hgradscaled = Expr::pointwisemul(Expr::constant(learningrate), H2Hgrad);
        cout << "\n\nlearningrate * loss->grad[H2H]:\n" << H2hgradscaled->to_str();


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
