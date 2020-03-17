//
// Created by bollu on 18/03/20.
//

#ifndef LANGUAGEMODELS_CODEGENMLIR_H
#define LANGUAGEMODELS_CODEGENMLIR_H

#include "lang.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;


using namespace std;

struct CodegenMLIR {

    string program(Program p)  { return ""; }
};
#endif //LANGUAGEMODELS_CODEGENMLIR_H
