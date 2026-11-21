#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
struct Plan {
    cublasLtHandle_t h;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a,b,c;
    cublasLtMatmulHeuristicResult_t alg[48];
    int count;
};
extern "C" Plan* init_plan(int m,int n,int k,void* bias,int gelu,int* count) {
    Plan* p=new Plan{};
    cublasLtCreate(&p->h);
    cublasLtMatmulDescCreate(&p->desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    cublasOperation_t trans=CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans));
    cublasLtEpilogue_t ep=gelu?CUBLASLT_EPILOGUE_GELU_BIAS:CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep));
    cublasLtMatmulDescSetAttribute(p->desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias));
    cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16BF,k,n,k);
    cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16BF,k,m,k);
    cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t size=32<<20;
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&size,sizeof(size));
    cublasLtMatmulAlgoGetHeuristic(p->h,p->desc,p->a,p->b,p->c,p->c,pref,48,p->alg,&p->count);
    *count=p->count;
    cublasLtMatmulPreferenceDestroy(pref);
    return p;
}
extern "C" int apply_plan(Plan* p,int i,void* x,void* w,void* y,void* work,cudaStream_t st) {
    float a=1.f,b=0.f;
    return cublasLtMatmul(p->h,p->desc,&a,w,p->a,x,p->b,&b,y,p->c,y,p->c,&p->alg[i].algo,work,32<<20,st);
}
