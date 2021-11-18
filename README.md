<p align="center">
  <img src="https://i.imgur.com/IPqkXTN.png">
   <h1 align="center" style="border-bottom: none">kMath</h1>
   <h4 align="center">/kmæθ/</h4>
   <h6 align="center">A work-in-progress general-purpose C++20/23 header-only maths library that works in constant context</h6>
   </br>
</p>

## Abstract

The kMath Project aims to provide a simple implementation of mathematical concepts such as euclidean vectors, matrices, quaternions, euler angles, linear interpolation (lerp, slerp) that can be used in constant context.
This allows the compiler to better understand the code at compile-time and generate better assembly.

We are not using compiler-specific or platform specific extensions but rely on the compilers auto-vectorization, -loop-unrolling and inlining instead which is particularly useful in some environments such as embedded or kernel that may not have access to the C Standard Library, AVX/SSE2 or MMX/x87 FPU instructions.

## Documentation

You can find our documentation [here](http://typena.me/docs/structmath_1_1_vector.html), to generate it locally `cd` into git root directory and start `doxygen`.

## Optimizations

```cpp
// main.cpp
#include <https://raw.githubusercontent.com/auto-lambda/kMath/master/include/kmath/math.hpp>
#include <cstdio>

int main(int const argc, char const * const[]) {
  math::Vector const vector_2d {1.5, 1.5};      // (1.5, 1.5)
  math::Vector       vector_3d {1.5, 1.5, 1.5}; // (1.5, 1.5, 1.5)

  using ScalarType = decltype(vector_2d)::Scalar;
  // argc is 1 but the compiler doesn't know at compile-time
  auto const scalar = static_cast<ScalarType>(argc);
  
  auto const scalar_mul_vec =    scalar * vector_3d; //  1.50
  auto const vec_mul_scalar = vector_3d * scalar;    //  1.50
  auto const vec_mul_vec    = vector_2d * vector_3d; //  2.25
  vector_3d *= vector_3d;                            //  2.25
  auto const neg_vec3d = -vector_3d;                 // -2.25

  // math::ct_sqrt dynamically dispatches between
  // (a) compile-time implementation of sqrt if constant evaluated
  // (b) std::sqrt otherwise
  constexpr auto rational = math::ct_sqrt(5.0625); // 5.0625 = 2.25²
  
  std::printf("%.2lf\n%.2lf\n%.2lf\n%.2lf\n%.2lf\n%.2lf\n",
    scalar_mul_vec[0],  //  1.50
    vec_mul_scalar[0],  //  1.50
       vec_mul_vec[0],  //  2.25
         vector_3d[0],  //  2.25
         neg_vec3d[0],  // -2.25
            rational);  //  2.25

  // This gets optimized out!
  if (vector_2d[0] != static_cast<ScalarType>(1.5))
    puts("test failed");

  // you can use vec.raw() and unpack with structured bindings
  [[maybe_unused]] auto &[x,y,z] = vector_3d.raw();
}
```
Compiling `main.cpp` with `clang 13` and `-std=c++2b -O3 -Wall -Wpedantic -Wconversion -Werror -mavx2` produces the following output:
```cpp
.LCPI0_0:
  .quad   0x3ff8000000000000                       # double  1.50
.LCPI0_1:
  .quad   0xc002000000000000                       # double -2.25
.LCPI0_2:
  .quad   0x4002000000000000                       # double  2.25
main:                                              # @main
  push      rax
  vcvtsi2sd xmm0, xmm0, edi                        # scalar = static_cast<ScalarType>(argc)
  vmulsd    xmm0, xmm0, qword ptr [rip + .LCPI0_0] #   arg1 = scalar * vector_3d // 1.50
  vmovsd    xmm4,       qword ptr [rip + .LCPI0_1] #   arg5 = -2.25
  vmovsd    xmm2,       qword ptr [rip + .LCPI0_2] #   arg3 =  2.25
  mov       edi, offset .L.str                     #   arg0 = format string
  vmovapd   xmm1, xmm0                             #   arg2 = arg1 // 1.50
  vmovaps   xmm3, xmm2                             #   arg4 = arg3 // 2.25
  vmovaps   xmm5, xmm2                             #   arg6 = arg3 // 2.25
  mov       al, 6                                  # printf argument count
  call      printf                                 # call printf
  xor       eax, eax                               # return 0
  pop       rcx
  ret
.L.str:
  .asciz  "%.2lf\n%.2lf\n%.2lf\n%.2lf\n%.2lf\n%.2lf\n"
```
***[Try it live on the amazing Compiler Explorer by Matt Godbolt](https://godbolt.org/z/fEEoP3Tqv)*** *[x86-64 <sup>clang, gcc, icx</sup>, ARM64 <sup>gcc</sup>, RISC-V <sup>clang, gcc</sup>]*

## Progress
- [x] mathematical constants
- [x] vector
- [ ] matrix
- [ ] quaternions
- [ ] interpolation
