#include <import/math/poly.h>
#include <import/sys.h>
#include "math/poly/FitLapack.h"

int main(int argc, char * argv[] ) {

  const int num_trials = 300000;
  int test_size  = argc-1;
  int num_coeffs = argc-1;
 
  //
  // test problem:
  //
  // given x and f(x) we calculate a polynomial 
  // that describes f() 
  //
  // For this test f() will be a 3rd order polynomial,
  // but our test data will have a little random noise
  // added to it (~1%). We'll have test_size = 15 samples
  // to give our fitting function so it will overfit.
  // (but the important thing is comparing out
  //
  std::vector<double> x(test_size);
  std::vector<double> fx(test_size);

  std::vector<double> coefs(num_coeffs);
  for(int i = 1; i < argc; ++i) {
    coefs[i-1] = atof(argv[i]);
  }
  
  math::poly::OneD<double> actual(coefs);

  // setup X
  for(int i = 0; i < test_size; ++i) {
    x[i] = static_cast<double>(i + 10);
  }

  // setup f(X)
  for(int i = 0; i < test_size; ++i) {
    double sum = 0;
    for(size_t j = 0; j < coefs.size(); ++j) {
      sum += actual[j]*std::pow(x[i],static_cast<double>(j));
    }
    fx[i] = sum;
  }

  sys::CPUStopWatch csw;
  double ctime = 0.0;
  double ltime = 0.0;
  
  csw.start();
  math::poly::OneD<double> resultC;
  for(int i = 0; i < num_trials; ++i) {
    resultC = math::poly::fit<std::vector<double> >(x,fx,num_coeffs-1);
  }
  ctime = csw.stop();
  
  csw.clear();
  csw.start();
 
  math::poly::OneD<double> resultL;
  for(int i = 0; i < num_trials; ++i) {
    resultL = math::poly::fitDgels<std::vector<double> >(x,fx,num_coeffs-1);
  }
  ltime = csw.stop();

  std::cout << " Coda:   f(x) = " << resultC << "(" << ctime << " ms cpu)" << std::endl;
  std::cout << " Lapack: f(x) = " << resultL << "(" << ltime << " ms cpu)" << std::endl;
  std::cout << " Actual: f(x) = " << actual  << std::endl;





}
