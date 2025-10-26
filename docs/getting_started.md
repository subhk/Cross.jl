# Getting Started

This page mirrors the onboarding style of Kore’s manual and focuses on getting a fresh clone of Cross.jl ready for experiments. Follow the steps in order on Linux, macOS, or WSL.

## 1. Prerequisites

- Julia 1.10 or newer (Cross.jl is developed against 1.10.x).
- A C/Fortran toolchain (for building dependencies such as MKL or FFTW if Julia requests them).
- Git (for cloning and pulling updates).

Optional but recommended:

- Python ≥ 3.10 if you plan to build the MkDocs documentation locally.
- VS Code with the Julia extension for a richer REPL and plot experience.

## 2. Clone the Repository

```bash
git clone https://github.com/<your-org>/Cross.jl.git
cd Cross.jl
```

If you intend to contribute documentation using GitHub Pages, keep the default branch checked out (usually `main`).

## 3. Instantiate the Julia Environment

Open Julia inside the project folder and instantiate the dependencies recorded in `Project.toml` and `Manifest.toml`:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

The first run downloads packages such as `LinearAlgebra`, `KrylovKit`, `FeastKit`, and `JLD2`. Subsequent sessions reuse the compiled artifacts.

## 4. Run the Test Suite

Before creating new problems, ensure your build passes the regression tests:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.test()
```

Expect the tests to take a few minutes; they assemble small eigenvalue problems to verify that matrix blocks and solver wrappers agree with stored benchmarks.

## 5. Verify an Example Script

For a basic smoke test, execute the linear stability demo:

```bash
julia --project=. example/linear_stability_demo.jl
```

You should see a summary of critical Rayleigh numbers and drift frequencies. If the script fails, consult the Troubleshooting section on the bottom of this page.

## 6. Recommended Julia Configuration

Add the following snippet to `~/.julia/config/startup.jl` to make REPL sessions friendlier:

```julia
atreplinit() do repl
    try
        @eval using Revise
    catch err
        @warn "Revise not available" err
    end
end
```

This enables Revise to pick up changes as you edit the source files—essential for iterative development.

## 7. Optional: Build the MkDocs Site Locally

The documentation in this repo uses MkDocs Material, just like Kore. To preview locally:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r docs/requirements.txt  # created below
mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser to see the rendered site with live reload.

## Troubleshooting Checklist

- **Package refuses to precompile:** run `Pkg.update()` and `Pkg.instantiate()` again; artifacts may be missing.
- **Out-of-memory errors:** reduce `lmax` or `Nr` in the examples; use sparse solver options (`nev`, `which` flags).
- **Solver hangs:** switch to a different backend (`:arpack`, `:krylovkit`, or `:feast`) using the options described in the Solver Reference page.
- **Documentation build fails:** ensure you are using Python 3.10+ and that the virtual environment is active before running `mkdocs`.
