#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
extern "C" {
struct MP {long prec;int sign;long exp;unsigned long* d;};
void mpfr_init2(MP*,long);void mpfr_clear(MP*);int mpfr_set_d(MP*,double,int);int mpfr_exp(MP*,const MP*,int);double mpfr_get_d(const MP*,int);
}
struct I {double l,h;I(double a=0):l(a),h(a){} I(double a,double b):l(a),h(b){}};
static double dn(double x){return std::nextafter(x,-INFINITY);}static double up(double x){return std::nextafter(x,INFINITY);}
static I add(I a,I b){return I(dn(a.l+b.l),up(a.h+b.h));}static I neg(I a){return I(-a.h,-a.l);}static I sub(I a,I b){return add(a,neg(b));}
static I mul(I a,I b){double x[]={a.l*b.l,a.l*b.h,a.h*b.l,a.h*b.h};for(double t:x)if(std::isnan(t))return I(-INFINITY,INFINITY);return I(dn(*std::min_element(x,x+4)),up(*std::max_element(x,x+4)));}
static I divi(I a,I b){if(b.l<=0&&b.h>=0)return I(-INFINITY,INFINITY);return mul(a,I(dn(1/b.h),up(1/b.l)));}
struct Expo {MP x,y;Expo(){mpfr_init2(&x,160);mpfr_init2(&y,160);}~Expo(){mpfr_clear(&x);mpfr_clear(&y);}I exp(I a){mpfr_set_d(&x,a.l,0);mpfr_exp(&y,&x,3);double l=mpfr_get_d(&y,3);mpfr_set_d(&x,a.h,0);mpfr_exp(&y,&x,2);return I(l,mpfr_get_d(&y,2));}};
extern "C" int bounds(const double* rp,const double* beta,const double* lm,const double* center,const unsigned char* mask,int M,int P,int D,int r,int a,int policy,double* out){
 Expo ex;I gain(0);double extra=0;for(int q=0;q<M;q++)for(int p=0;p<P;p++)extra=std::max(extra,-beta[q*P+p]);I scale=ex.exp(I(-extra));
 for(int q=0;q<M;q++){
  auto at=[&](const double* x,int p,int d){return x[(q*P+p)*D+d];};
  double shift=-INFINITY;for(int p=0;p<P;p++)shift=std::max(shift,lm[q*P+p]+beta[q*P+p]);
  std::vector<I> exact(P),base(P),N(D),Dout(D),H(D),delta(D);I Z(0);
  for(int p=0;p<P;p++){I le=add(I(lm[q*P+p]),I(beta[q*P+p]));exact[p]=ex.exp(sub(le,I(shift)));base[p]=ex.exp(sub(I(lm[q*P+p]),I(shift)));Z=add(Z,exact[p]);for(int d=0;d<D;d++)N[d]=add(N[d],mul(exact[p],add(I(at(center,p,d)),I(at(rp,p,d)))));}
  for(int d=0;d<D;d++)Dout[d]=divi(N[d],Z);
  if(policy==1){I energy[2];int pages[2]={r,a};for(int k=0;k<2;k++){int p=pages[k];for(int d=0;d<D;d++){I v=add(I(at(center,p,d)),I(at(rp,p,d)));I net=sub(mul(divi(mul(exact[p],scale),Z),sub(v,Dout[d])),mul(divi(mul(base[p],scale),Z),sub(I(at(center,p,d)),Dout[d])));energy[k]=add(energy[k],mul(net,net));}}gain=add(gain,sub(energy[1],energy[0]));continue;}
  auto output=[&](bool after){double sh=-INFINITY;for(int p=0;p<P;p++){bool s=mask[p];if(after&&p==r)s=false;if(after&&p==a)s=true;sh=std::max(sh,lm[q*P+p]+(s?beta[q*P+p]:0));}I den(0);std::vector<I> num(D);for(int p=0;p<P;p++){bool s=mask[p];if(after&&p==r)s=false;if(after&&p==a)s=true;I le=s?add(I(lm[q*P+p]),I(beta[q*P+p])):I(lm[q*P+p]);I w=ex.exp(sub(le,I(sh)));den=add(den,w);for(int d=0;d<D;d++)num[d]=add(num[d],mul(w,s?add(I(at(center,p,d)),I(at(rp,p,d))):I(at(center,p,d))));}for(int d=0;d<D;d++)num[d]=divi(num[d],den);return num;};
  H=output(false);std::vector<I> Hnew=output(true);
  for(int d=0;d<D;d++){I e=sub(Dout[d],H[d]);I dd=sub(Hnew[d],H[d]);gain=add(gain,sub(mul(I(2),mul(e,dd)),mul(dd,dd)));}
 }
 gain=divi(gain,I(M));out[0]=gain.l;out[1]=gain.h;return 0;
}
