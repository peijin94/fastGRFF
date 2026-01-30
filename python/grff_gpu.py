import pathlib
import cupy as cp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = (HERE / "..").resolve()
CUDA_SRC = (ROOT / "cuda" / "gpugrff.cu").resolve()
INCLUDE_DIR = (ROOT / "include").resolve()
CUDA_DIR = (ROOT / "cuda").resolve()


_module = None


def _build_module():
    code = CUDA_SRC.read_text(encoding="utf-8")
    return cp.RawModule(
        code=code,
        options=(
            "--std=c++17",
            f"-I{INCLUDE_DIR}",
            f"-I{CUDA_DIR}",
        ),
        name_expressions=("get_mw_kernel", "get_mw_slice_kernel"),
    )


def _ensure_heap(bytes_required=64 * 1024 * 1024):
    # Ensure device malloc has enough heap for per-call allocations.
    cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitMallocHeapSize, bytes_required)


def _get_module():
    global _module
    if _module is None:
        _ensure_heap()
        _module = _build_module()
    return _module


def get_mw(Lparms, Rparms, Parms, T_arr, DEM_arr, DDM_arr, RL):
    """
    GPU GET_MW (single pixel), matching GRFF array layout (Fortran-order).

    Inputs must be CuPy arrays, dtype float64 for doubles, int32 for Lparms.
    RL is written in-place with shape (OutSize, Nf), order='F'.
    """
    mod = _get_module()
    kernel = mod.get_function("get_mw_kernel")
    status = cp.zeros((1,), dtype=cp.int32)

    kernel(
        grid=(1,),
        block=(1,),
        args=(Lparms, Rparms, Parms, T_arr, DEM_arr, DDM_arr, RL, status),
    )
    return int(status.get()[0])


def get_mw_slice(Lparms_M, Rparms_M, Parms_M, T_arr, DEM_arr_M, DDM_arr_M, RL_M):
    """
    GPU GET_MW_SLICE (multi-pixel), matching GRFF array layout (Fortran-order).

    One CUDA block per pixel, one thread per block.
    """
    mod = _get_module()
    kernel = mod.get_function("get_mw_slice_kernel")
    Npix = int(Lparms_M[0].get())
    status = cp.zeros((Npix,), dtype=cp.int32)

    kernel(
        grid=(Npix,),
        block=(1,),
        args=(Lparms_M, Rparms_M, Parms_M, T_arr, DEM_arr_M, DDM_arr_M, RL_M, status),
    )
    return status.get()
