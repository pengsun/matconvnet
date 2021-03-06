# Installing and compiling the library

In order to install the library, follows these steps:

1.  Download and unpack the library source code into a directory of
    your choice. Call the path to this directory `<MatConvNet>`.
2.  [Compile](#compiling) the library.
3.  Start MATLAB and type:

        > run <MatConvNet>/matlab/vl_setupnn

    in order to add MatConvNet to MATLAB's search path.

At this point the library is ready to use. You can test it by using
the command:

    > vl_test_nnlayers

To test GPU support (if you have [compiled it](#gpu)) use instead:

> vl_test_nnlayers(true)

Note that the second tests runs slower than the CPU version; do not
worry, this is an artefact of the test procedure.

<a name='compiling'></a>
## Compiling

MatConvNet compiles under Linux, Mac, and Windows (with the exception
of the `vl_imreadjpeg` tool which is not yet supported under
Windows). This page discusses compiling MatConvNet using the MATLAB
function `vl_compilenn`. While this is the easiest method,
[the command line or an IDE can be used as well](install-alt.md).

<a name='cpu'></a>
### Compiling for CPU

If this is the first time you compile MatConvNet, consider trying
first the CPU version. In order to do this, use the
[`vl_compilenn`](mfiles/vl_compilenn) command supplied with the
library:

1.  Make sure that MATLAB is
    [configured to use your compiler](http://www.mathworks.com/help/matlab/matlab_external/changing-default-compiler.html).
2.  Open MATLAB and issue the commands:

        > cd <MatConvNet>
        > addpath matlab
        > vl_compilenn

At this point MatConvNet should start compiling. If all goes well, you
are ready to use the library. If not, you can try debugging the
problem by running the complation script again in verbose mode:

    > vl_compilenn('verbose', 1)

Increase the verbosity level to 2 to get even more information.

<a name='gpu'></a>
### Compiling the GPU support

To use the the GPU-accelerated version of the library, you will need a
NVIDA GPU card with compute capability 2.0 or greater and a copy of
the NVIDIA CUDA toolkit. The version of the CUDA toolkit should
ideally **match your MATLAB version**:

| MATLAB    | CUDA toolkit      |
|-----------|-------------------|
| R2013b    | 5.5               |
| R2014a    | 6.0               |
| R2014b    | 6.5               |

You can also use the `gpuDevice` MATLAB command to find out the
correct version of the CUDA toolkit. It is also possible (and
sometimes necessary) to use a more recent version of CUDA of the one
officially supported; this is [explained later](#nvcc).

Assuming that there is only a single copy of the CUDA toolkit
installed in your system and that it matches MATLAB's version, compile
the library with:

    > vl_compilenn('enableGpu', true)

If you have multiple versions of the CUDA toolkit, or if the script
cannot find the toolkit for any reason, specify the path to the CUDA
toolkit explicitly. For example, on a Mac this may look like:

    > vl_compilenn('enableGpu', true, 'cudaRoot', '/Developer/NVIDIA/CUDA-6.0')

Once more, you can use the `verbose` option to obtain more information
if needed.

<a name='nvcc'></a>
### Using an unsupported CUDA toolkit version

MatConvNet can be compiled to use a more recent version of the CUDA
toolkit than the one officially supported by MATLAB. While this may
cause unforeseen issues (although none is known so far), it is
necessary to use [recent libraries such as cuDNN](#cudnn).

Compiling with a newer version of CUDA requires using the
`cudaMethod,nvcc` option. For example, on a Mac this may look like:

    > vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'cudaMethod', 'nvcc')

Note that at this point MatConvNet MEX files are linked *against the
specified CUDA libraries* instead of the one distributed with
MATLAB. Hence, in order to use MatConvNet it is now necessary to allow
MATLAB accessing these libraries. On Linux and Mac, one way to do so
is to start MATLAB from the command line (terminal) specifying the
`LD_LIBRARY_PATH` option. For instance, on a Mac this may look like:

    $ LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib /Applications/MATLAB_R2014b.app/bin/matlab

On Windows, chances are that the CUDA libraries are already visible to
MATLAB so that nothing else needs to be done.

<a name='cudnn'></a>
### Compiling the cuDNN support

MatConvNet supports the NVIDIA <a
href='https://developer.nvidia.com/cuDNN'>cuDNN library</a> for deep
learning (and in particular their fast convolution code). In order to
use it, obtain the
[cuDNN Candidate Release 2](http://devblogs.nvidia.com/parallelforall/accelerate-machine-learning-cudnn-deep-neural-network-library). Note
that only Candidate Release 2 has been tested so far (Candidate
Release 1 will *not* work). Make sure that the CUDA toolkit matches
the one in cuDNN (e.g. 6.5). This often means that the CUDA toolkit
will *not* match the one used internally by MATLAB, such that the
[compilation method](#nvcc) discussed above must be used.

Unpack the cuDNN library binaries and header files in a place
`<Cudnn>` of you choice. In the rest of the instructions, it will be
assumed that this is a new directory called `local/` in the
`<MatConvNet>` root directory,
(i.e. `<Cudnn>`=`<MatConvNet>/local`). For example, the directory
structure on a Mac could look like:

     COPYING
     Makefile
     Makefile.mex
     ...
     examples/
     local/
       cudnn.h
       libcudnn.6.5.dylib
       libcudnn.dylib
       ...

Use `vl_compilenn` with the `cudnnEnable,true` option to compile the
library; do not forget to use `cudaMethod,nvcc` as, at it is likely,
the CUDA toolkit version is newer than MATLAB's CUDA toolkit. For
example, on Mac this may look like:

    > vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'cudaMethod', 'nvcc', ...
                   'enableCudnn', 'true', ...
                   'cudnnRoot', 'local/') ;

MatConvNet is now compiled with cuDNN support. When starting MATLAB,
however, do not forget to point it to the paths of both the CUDA and
cuDNN libraries. On a Mac terminal, this may look like:

    $ cd <MatConvNet>
    $ LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib:local /Applications/MATLAB_R2014b.app/bin/matlab

On Windows, copy the CUDNN DLL file `<Cudnn>/cudnn*dll` (or from
wherever you unpacked cuDNN) into the `<MatConvNet>/matlab/mex`
directroy.

### Compiling `vl_imreadjpeg`
<a name='jpeg'></a>

The `vl_imreadjpeg` function in the MatConvNet toolbox accelerates
reading large batches of JPEG images. In order to compile it, a copy
of LibJPEG and of the corresponding header files must be available to
the MEX compiler used by MATLAB.

On *Linux*, it usually suffices to install the LibJPEG developer
package (for example `libjpeg-dev` on Ubuntu Linux). Then both
`vl_compilenn()` and the Makefile should work out of the box.

On *Mac OS X*, LibJPEG can be obtained for example by using
[MacPorts](http://www.macports.org):

    > sudo port install jpeg

This makes the library available as `/opt/local/lib/libjpeg.dylib` and
the header file as `/opt/local/include/jpeglib.h`. If you compile the
library us using `vl_compilenn()`, you can pass the location of these
files as part of the `ImreadJpegFlags` option:

    > vl_compilenn('enableImreadJpeg', true, 'imreadJpegFlags', ...
        {'-I/opt/local/include','-L/opt/local/lib','-ljpeg'});

If LibJPEG is installed elsewhere, you would have to replace the paths
`/opt/local/include` and `/opt/local/lib` accordingly.

## Further examples

To compile all the features in MatConvNet on a Mac and MATLAB 2014b,
CUDA toolkit 6.5 and cuDNN Release Candidate 2, use:

    > vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'enableCudnn', true, 'cudnnRoot', 'local/', ...
                   'enableImreadJpeg', true,  ...
                   'imreadJpegCompileFlags', {'-I/opt/local/include'}, ...
                   'imreadJpegLinkFlags', {'-L/opt/local/lib','-ljpeg'}) ;

The equivalent command on Ubuntu Linux would look like:

    > vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
                   'cudaRoot', '/opt/local/cuda-6.5', ...
                   'enableCudnn', true, 'cudnnRoot', 'local/', ...
                   'enableImreadJpeg', true) ;
