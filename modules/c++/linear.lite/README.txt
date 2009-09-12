The linear.lite library is still in a very early form.
The API is likely to undergo some changes, and the code is not stable.

Notes on implementation:

Sparse Matrix class doesnt work (compile) in MTL on windows.  Error is
in MTL

Sparse Matrix in both boost and MTL do not allow a caller access to a reference
to the element type directly, but through containers.  As a result, no
non-const accessor is allowed for get() and operator[].  The set() method
must be used instead

MTL and boost both fail to build, even without Sparse matrices on Solaris.  

