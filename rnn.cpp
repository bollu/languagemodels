#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include "codegenc.h"
#include "codegenmlir.h"

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
vector<vector<int>> sentences;

void add_word_to_vocab(string w) {
    if (vocab.find(w) != vocab.end()) {
        return;
    }
    word2ix[w] = last_word_index;
    ix2word[last_word_index] = w;
    last_word_index++;
}

bool is_char_word_break(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '.';
}

// disable LSAN. Only keep ASAN. I leak far too much memory ;)
extern "C" int __lsan_is_turned_off() { return 1; }

// consume whitespace and get the next word, or empty string if run out.
string parse_word(FILE *f) {
    char c;
    do {
        c = fgetc(f);
    } while (c != EOF && is_char_word_break(c));

    if (c == EOF) {
        return "";
    }
    ungetc(c, f);

    string w;
    while (1) {
        c = fgetc(f);
        if (is_char_word_break(c)) break;
        w += c;
    }

    return w;
}

bool consume_till_sentence_end_or_word(FILE *f) {
    while (1) {
        char c = fgetc(f);
        if (c == '.') {
            return true;
        }
        if (c == ' ' || c == '\t' || c == '\n') {
            continue;
        }
        ungetc(c, f);
        return false;
    }
}

vector<int> parse_sentence(FILE *f) {
    vector<int> sentence;
    while (1) {
        // returns true if '.' was consumed
        if (consume_till_sentence_end_or_word(f)) {
            return sentence;
        }
        string w = parse_word(f);
        if (w == "") {
            return sentence;
        }
        auto it = word2ix.find(w);
        assert(it != word2ix.end());
        sentence.push_back(it->second);
    }
}

void test_expr_dot() {
    int NDIMS = 3;
    Arr a("a", NDIMS);
    Arr b("b", NDIMS);
    Arr i("i");
    Expr *mul = new Mul(a.ix(i), b.ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "******dot******\n";
    cout << dot->to_str_with_shape() << "\n";
    Arr k("k");
    Expr *grad = dot->grad("a", {k});
    cout << "## dot->grad[a k]: ##\n\t" << grad->to_str_with_shape() << "\n";
    cout << "## dot->grad[a k](normalized): ##\n\t"
         << grad->normalize()->to_str_with_shape() << "\n";
}

void test_expr_matvec() {
    const int NOUT = 3;
    const int NIN = 3;
    Arr m("m", NOUT, NIN);
    Arr v("v", NIN);
    Arr i("i"), j("j");
    Expr *mul = new Mul(m.ix(i, j), v.ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "******Mat @ vec******\n";
    cout << matvec->to_str_with_shape() << "\n";

    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *gradm = matvec->grad("m", {Arr("d0"), Arr("d1")});
    cout << "## mul->grad[a d0 d1]: ##\n\t" << gradm->to_str() << "\n";
    cout << "## mul->grad[a d0 d1](normalized): ##\n\t"
         << gradm->normalize()->to_str_with_shape() << "\n";

    Expr *gradv = matvec->grad("v", {Arr("d0")});
    cout << "## mul->grad[v d0]: ##\n\t" << gradv->to_str_with_shape() << "\n";
    cout << "## mul->grad[v d0](normalized): ##\n\t"
         << gradv->normalize()->to_str_with_shape() << "\n";
}


void test_expr_tanh() {
    const int NDIM = 3;
    Arr a("a", NDIM, NDIM);
    Arr i("i");
    Arr j("j");
    Expr *aixed = a.ix(i, j);
    Expr *matvec = new Tanh(aixed);

    cout << "*****tanh on matrix*****\n";
    cout << matvec->to_str_with_shape() << "\n";

    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *grada = matvec->grad("a", {Arr("k"), Arr("l")});
    cout << "## mul->grad[a k l]: ##\n\t" << grada->to_str_with_shape() << "\n";
    cout << "## mul->grad[a k l](normalized): ##\n\t"
         << grada->normalize()->to_str_with_shape() << "\n";
}

// construct Dot using program
void test_program_dot() {
    Program p;
    IRBuilder builder(p);

    static const int ARRSIZE = 10;
    Arr a("a", ARRSIZE);
    Arr b("b", ARRSIZE);
    Arr grada("grada", ARRSIZE);
    Arr gradb("gradb", ARRSIZE);
    Arr dot("dot");
    Arr i("i");
    Arr k("k");
    builder.copy(dot.ix(), new Contract(i, new Mul(a.ix(i), b.ix(i))));
    builder.copy(grada.ix(k), new Mul(new ConstantFloat(1e-2),
                                      p[dot].rhs->grad("a", {k})->normalize()));
    builder.copy(gradb.ix(k), new Mul(new ConstantFloat(1e-2),
                                      p[dot].rhs->grad("b", {k})->normalize()));

    // Incr is weird, since we should probably be the ones doing the
    // indexing...
    builder.incr(a.ix(k), grada.ix(k));
    builder.incr(b.ix(k), gradb.ix(k));

    cout << "*****program dot*****\n";
    cout << p.to_str();
}

void test_program_dot_batched() {
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr focuses("focuses", BATCHSIZE, EMBEDSIZE);
    Arr ctxes("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr total_loss("total_loss");

    Arr bi("bi");
    Arr i("i");

    Program p;

    IRBuilder builder(p);
    Forall *forbi = builder.insertFor({bi});

    builder.setInsertPoint(forbi);
    Expr *dots = new Contract(i, new Mul(focuses.ix(bi, i), ctxes.ix(bi, i)));
    Expr *losses = new Sub(new ConstantInt(1), dots);

    builder.copy(total_loss.ix(), new Contract(bi, losses));

    Expr *lr = new ConstantFloat(1e-2);

    Arr dbi("dbi"), dk("dk");
    builder.incr(
        focuses.ix(dbi, dk),
        new Mul(lr,
                p[total_loss].rhs->grad("focuses", {dbi, dk})->normalize()));
    builder.incr(
        ctxes.ix(dbi, dk),
        new Mul(lr, p[total_loss].rhs->grad("ctxes", {dbi, dk})->normalize()));

    cout << "*****program batched dot*****\n";
    cout << p.to_str() << "\n";
}

void test_program_dot_indicrect() {
    cout << "*****program indirect dot*****\n";
    static const int VOCABSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr embeds("embeds", VOCABSIZE, EMBEDSIZE);
    Arr focusix("focusix");
    Arr ctxix("ctxix");
    Arr focus("focus", EMBEDSIZE);
    Arr ctx("ctx", EMBEDSIZE);
    Arr dot("dot");
    Arr loss("loss");

    Arr i("i"), k("k");
    Program p;

    IRBuilder builder(p);

    builder.reference(focus.ix(i), embeds.ix(focusix, i));
    builder.reference(ctx.ix(i), embeds.ix(ctxix, i));

    // (1 - (focus[i] * ctx[i]).contract(i))^2
    builder.copy(dot.ix(), new Contract(i, new Mul(focus.ix(i), ctx.ix(i))));
    builder.copy(loss.ix(), new Sub(new ConstantInt(1), p[dot].rhs));

    Expr *lr = new ConstantFloat(1e-2);

    builder.incr(focus.ix(k),
                 new Mul(lr, p[loss].rhs->grad("focus", {k})->normalize()));
    builder.incr(ctx.ix(k),
                 new Mul(lr, p[loss].rhs->grad("ctx", {k})->normalize()));
    cout << p.to_str();
}

Expr *matvecmul(Arr m, Arr v, Arr ix) {
    Arr c("c");
    return new Contract(c, new Mul(m.ix(ix, c), v.ix(c)));
}

void test_program_dot_batched_indirect() {
    cout << "*****program batched, indirect addressed dot (final code that we "
            "need for word embeddings)*****\n";
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr focuses("focuses", BATCHSIZE, EMBEDSIZE);
    Arr ctxes("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr focus("focus");
    Arr ctx("ctx");
    Arr dot("dot");
    Arr loss("loss");

    Arr bi("bi");
    Arr i("i");
    // Arr *k = new Arr("k");
    Program p;

    IRBuilder builder(p);
    // builder.setInsertPoint(builder.insertFor(bi));
    Forall *forbi = builder.insertFor({bi, i});
    builder.setInsertPoint(forbi);

    builder.reference(focus.ix(), focuses.ix(bi, i));
    builder.reference(ctx.ix(), ctxes.ix(bi, i));

    builder.incr(dot.ix(), new Mul(focus.ix(), ctx.ix()));

    builder.setInsertPoint(p);
    forbi = builder.insertFor({bi, i});
    builder.setInsertPoint(forbi);
    builder.copy(loss.ix(), new Sub(new ConstantInt(1), p[dot].rhs));

    Expr *lr = new ConstantFloat(1e-2);

    Arr focusl("focusl");
    Arr ctxl("ctxl");
    builder.reference(focusl.ix(), focuses.ix(bi, i));
    builder.reference(ctxl.ix(), ctxes.ix(bi, i));
    builder.incr(focusl.ix(),
                 new Mul(lr, p[loss].rhs->grad("focus", {})->normalize()));
    builder.incr(ctxl.ix(),
                 new Mul(lr, p[loss].rhs->grad("ctx", {})->normalize()));
    cout << p.to_str();

    //cout << "*****Codegened word embddings code:*****\n";
    //CodegenC cc;
    //cout << cc.program(p) << "\n";


    cout << "*****Codegened word embddings code (C):*****\n";
    CodegenMLIR cmlir;
    //cout << cc.program(p) << "\n";

}

Expr *l2(Arr v, Arr w) {
    Arr c("c");
    Expr *s = new Sub(v.ix(c), w.ix(c));
    return new Contract(c, new Mul(s, s));
}

// compute dy/dx

// get all paths from a source node to a target node, _backwards_.
// That is, expressions:
// x1 := x0 + 1
// x2 := x1 + 1
// x3 := x2 + 3
// with the call
// getPathsToArr(p, x3, x0, {x3}) will give
// - x3 x2 x1 x0
set<std::vector<Arr>> getPathsToArr(Program &p, Arr cur, Arr target,
                                    set<vector<Arr>> pathsToCur) {
    if (cur == target) {
        return pathsToCur;
    }

    ArrayGatherVisitor agv;

    // this array does not exist in the program, so just return.
    if (!p.is_array_assigned(cur)) {
        return {};
    }
    Expr *rhs = p[cur].rhs;

    // cout << "rhs for " << cur->name << " : " << (rhs ? rhs->to_str() :
    // "NULL") << "\n";
    if (!rhs) {
        return {};
    }

    // look in the RHS of this array
    agv.visitExpr(rhs);

    // cout << "arrays for |" << cur->name << "| := |"  << rhs->to_str() << "|";
    // for(Arr *a : agv.arrays) cout << a->name << " ";
    // cout << "|\n";

    set<vector<Arr>> ps;
    for (Arr child : agv.arrays) {
        set<vector<Arr>> pathsToChild;
        for (vector<Arr> path : pathsToCur) {
            path.push_back(child);
            pathsToChild.insert(path);
        }
        set<vector<Arr>> pathsToTarget =
            getPathsToArr(p, child, target, pathsToChild);
        ps.insert(pathsToTarget.begin(), pathsToTarget.end());
    }
    return ps;
}

// if array is [y intermediate x], compute:
// (>< i0 i1 i2  (* dy[outixs]/dintermediate[i0, i1, i2]  dindermediate[i0, i1,
// i2]/dx[inixs]))
Expr *codegenDerivativeChain(Program &p, vector<Arr> arrs, vector<Arr> inixs,
                             vector<Arr> outixs) {
    const int n = arrs.size();
    assert(arrs.size() >= 2);
    map<Arr, vector<Arr>> arr2ix;
    map<Arr, Expr *> arr2expr;
    arr2ix[arrs[0]] = outixs;
    arr2ix[arrs[n - 1]] = inixs;

    arr2expr[arrs[0]] = p[arrs[0]].rhs;

    for (int i = 1; i < n - 1; ++i) {
        struct Arr a = arrs[i];

        vector<Arr> aixs;

        for (int i = 0; i < a.sh.ndim; ++i) {
            aixs.push_back(Arr(a.name + "_" + to_string(i)));
        }
        arr2ix[arrs[i]] = aixs;

        Index *ix = p[arrs[i]].lhs;
        Expr *e = p[arrs[i]].rhs;
        // reindex
        for (int i = 0; i < a.sh.ndim; ++i) {
            e = e->subst(ix->ixs[i], aixs[i]);
        }

        arr2expr[arrs[i]] = e;
    }

    IRBuilder builder(p);
    Expr *chain = nullptr;
    vector<Arr> contractIxs;
    for (int i = 0; i < n - 1; ++i) {
        struct Arr cur = arrs[i];
        struct Arr next = arrs[i + 1];

        vector<Arr> ixcur = arr2ix[cur];
        vector<Arr> ixnext = arr2ix[next];
        vector<Arr> allixs;
        allixs.insert(allixs.end(), ixcur.begin(), ixcur.end());
        allixs.insert(allixs.end(), ixnext.begin(), ixnext.end());

        Expr *curval = arr2expr[cur];
        assert(curval);

        Expr *grad = curval->grad(next.name, ixnext)->normalize();
        Arr dcur_dnext("d" + cur.name + "_" + "d" + next.name,
                       cur.sh.appendInside(next.sh));
        builder.copy(dcur_dnext.ix(allixs), grad);

        if (!chain) {
            chain = dcur_dnext.ix(allixs);
        } else {
            cout << "prev chain: " << chain->to_str() << "\n";

            chain = new Mul(chain, dcur_dnext.ix(allixs));
            for (const Arr c : ixcur) {
                cout << "\tcontract " << chain->to_str() << "\n";
                chain = new Contract(c, chain);
            }
        }
        contractIxs = ixnext;
    }

    return chain;
}

// compute dy/dx, through any chain of expressions needed.
// TODO: keep a map of dy/dx -> array holding this value

Expr *takeIndirectDerivatives(Program &p, Arr y, Arr x, vector<Arr> inixs,
                              vector<Arr> outixs) {
    IRBuilder builder(p);
    set<vector<Arr>> yTox = getPathsToArr(p, y, x, {{y}});

    Expr *allsum = new ConstantInt(0);
    for (vector<Arr> path : yTox) {
        cout << "\n- path: ";
        for (Arr a : path) {
            cout << a.to_str() << " ";
        }
        allsum =
            new Add(allsum, codegenDerivativeChain(p, path, inixs, outixs));
    }
    return allsum;
}

// map an array to all uses of the array
map<Arr, set<Assign>> getUses(Program p) {
    map<Arr, set<Assign>> a2uses;
    for (Assign assign : p.assigns()) {
        ArrayGatherVisitor agv;
        agv.visitExpr(assign.rhs);
        for (Arr used : agv.arrays) {
            // map the used array on the RHS to the user on the LHS
            a2uses[used].insert(assign);
        }
    }

    return a2uses;
}

// map an array to all arrays used to compute this array
map<Arr, set<Arr>> getDependences(Program p) {
    map<Arr, set<Arr>> a2deps;
    for (Assign assign : p.assigns()) {
        ArrayGatherVisitor agv;
        agv.visitExpr(assign.rhs);
        for (Arr in : agv.arrays) {
            a2deps[assign.lhs->arr].insert(in);
        }
    }

    return a2deps;
}

// compute ds/dz for all variables reachable from $z$. Returns a mapping
// from each array to expressions
map<Arr, Arr> reverseDiff(Program &p, Arr z) {
    IRBuilder builder(p);
    assert(z.sh.ndim == 0);
    map<Arr, set<Arr>> deps = getDependences(p);

    cout << "=================================================\n";
    for (auto it : deps) {
        cout << it.first.name << " -> ";
        cout << "{ ";
        for (auto a : it.second) {
            cout << a.name << " ";
        }
        cout << "}";
        cout << "\n";
    }
    cout << "=================================================\n";

    map<Arr, Arr> dz_darr;
    Arr dz_dz("dz_dz");
    builder.copy(dz_dz.ix(), new ConstantInt(1));
    dz_darr[z] = dz_dz;

    queue<Arr> horizon;
    horizon.push(z);

    while (!horizon.empty()) {
        Arr o = horizon.front();
        horizon.pop();
        cout << "analyzing uses of: |" << o.name << "| \n";

        // array is not differentiable (not part of program)
        if (!p.is_array_assigned(o)) {
            continue;
        }

        // dz/do
        Assign oassign = p[o];
        Expr *oval = oassign.rhs;
        vector<Arr> oixs = oassign.lhs->ixs;

        assert(dz_darr.count(o));
        // array that hold derivative of ao wrt dz
        Arr dz_do = dz_darr.find(o)->second;

        // write out into every input
        // dz_di += dz_do x do_di [where o = f(i, ...) do_di = df_di]
        for (Arr i : deps[o]) {
            cout << "\tanalyzing |" << o.name << "| = f(" << i.name << ")\n";

            // needs to be initialized
            Arr dz_di;
            if (dz_darr.count(i)) {
                dz_di = dz_darr.find(i)->second;
            } else {
                dz_di = Arr("d" + z.name + "_d" + i.name, i.sh);
                dz_darr.insert(std::make_pair(i, dz_di));
                // TODO: write zero.
            }

            vector<Arr> iixs;
            for (int ix = 0; ix < i.sh.ndim; ++ix) {
                iixs.push_back(Arr(i.name + "_" + std::to_string(ix)));
            }

            // do/di
            Expr *do_di = oval->grad(i.name, iixs)->normalize();
            (void)(do_di);

            do_di = do_di->normalize();
            cout << "contracting along: ";
            for (Arr i : iixs) {
                cout << i.name << " ";
            }
            cout << "\n";
            cout << "  - dz/do:" << dz_do.to_str() << "\n";
            cout << "  - do/di:" << do_di->normalize()->to_str_with_shape()
                 << "\n";

            // write into dz_di
            builder.incr(
                dz_di.ix(iixs),
                Contract::contract(oixs, new Mul(dz_do.ix(oixs), do_di))
                    ->normalize());

            horizon.push(i);
        }
    }

    return dz_darr;
}

Expr *cell(Program &p, Arr I2H, Arr H2H, Arr i, Arr h, Arr I2Hi, Arr H2Hh,
           Arr ix) {
    IRBuilder builder(p);
    builder.copy(H2Hh.ix(ix), matvecmul(H2H, h, ix));
    builder.copy(I2Hi.ix(ix), matvecmul(I2H, i, ix));
    return new Tanh(new Add(I2Hi.ix(ix), H2Hh.ix(ix)));
}

void test_lstm() {
    // size of input/output arrays
    static const int IOSIZE = 5;
    static const int HIDDENSIZE = 10;
    static const int NINPUTS = 4;

    Arr inputs[NINPUTS];
    Arr I2Hi[NINPUTS];
    Arr hiddens[NINPUTS + 1];
    Arr H2Hh[NINPUTS];

    for (int i = 0; i < NINPUTS; ++i) {
        inputs[i] = Arr("i" + to_string(i), IOSIZE);
        I2Hi[i] = Arr("I2Hi" + to_string(i), HIDDENSIZE);
    }

    (void)(inputs);

    for (int i = 0; i < NINPUTS + 1; ++i) {
        hiddens[i] = Arr("h" + to_string(i), HIDDENSIZE);
    }

    for (int i = 0; i < NINPUTS; ++i) {
        H2Hh[i] = Arr("H2Hh" + to_string(i), HIDDENSIZE);
    }

    Arr ix("ix");
    Arr I2H("I2H", HIDDENSIZE, IOSIZE);
    Arr H2H("H2H", HIDDENSIZE, HIDDENSIZE);
    Arr H2O("H2O", IOSIZE, HIDDENSIZE);

    Program p;
    IRBuilder builder(p);

    Arr hprev = hiddens[0];
    for (int i = 0; i < NINPUTS; ++i) {
        Expr *c = cell(p, I2H, H2H, inputs[i], hprev, I2Hi[i], H2Hh[i], ix);
        cout << "program:\n" << p.to_str() << "\n";
        builder.copy(hiddens[i + 1].ix(ix), c);
        hprev = hiddens[i + 1];
    }

    Arr predict("p", IOSIZE);
    builder.copy(predict.ix(ix), matvecmul(H2O, hprev, ix));

    Arr output("o", IOSIZE);
    Arr loss("l");
    builder.copy(loss.ix(), l2(output, predict));

    cout << "*****LSTM*****:\n";
    cout << p.to_str();

    Arr i("i"), j("j");
    map<Arr, Arr> arr2der = reverseDiff(p, loss);

    // Arr *dH2H = new Arr("dH2H", HIDDENSIZE, HIDDENSIZE);
    // builder.incr(H2H.ix(i, j), der);

    cout << "*****LSTM(full)*****:\n";
    cout << p.to_str();

    //CodegenC cc;
    //cout << "******LSTM(codegen)******\n" << cc.program(p) << "\n";

    CodegenMLIR cmlir;
    cout << "******LSTM(codegen to MLIR)******\n" << cmlir.program(p) << "\n";
}

int main(int argc, char **argv) {
    test_expr_dot();
    test_expr_matvec();
    test_expr_tanh();
    test_program_dot();
    test_program_dot_batched();
    test_program_dot_indicrect();
    test_program_dot_batched_indirect();
    test_lstm();
    // return 0;

    if (argc != 2) {
        cout << "usage: <input-path>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");

    // construct vocab
    while (1) {
        string w = parse_word(f);
        cout << "found word: " << w << "\n";
        if (feof(f)) {
            break;
        }
        add_word_to_vocab(w);
    }
    rewind(f);
    while (1) {
        vector<int> sentence = parse_sentence(f);
        sentences.push_back(sentence);
        if (feof(f)) {
            break;
        }
    }
    fclose(f);

    for (auto it : word2ix) {
        cout << "|" << it.first << "|->" << it.second << "\n";
    }

    cout << "SENTENCES:\n";
    for (auto s : sentences) {
        for (auto w : s) {
            cout << w << " ";
        }
        cout << "\n";
    }
}
