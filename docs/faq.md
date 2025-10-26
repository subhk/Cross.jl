# FAQ & Troubleshooting

This page follows the quick-reference format of Kore’s FAQ. It collects recurring questions from early adopters and bundles ready-made fixes.

## Installation

**Q: Julia complains about incompatible versions of KrylovKit or FeastKit.**  
A: Run `Pkg.update()` within the project and ensure your Julia version is ≥ 1.10. If conflicts persist, delete `Manifest.toml` and re-run `Pkg.instantiate()` to resolve fresh dependencies.

**Q: `Pkg.instantiate()` hangs or takes excessively long.**  
A: Check your network connection and proxy settings. Some dependencies (e.g., MKL) are large; allow several minutes on first install.

## Running Examples

**Q: `example/linear_stability_demo.jl` throws `MethodError`.**  
A: Ensure you launch Julia with `--project=.` so the correct Cross.jl environment is active.

**Q: The eigenvalue search diverges or oscillates.**  
A: Provide a better `Ra_guess` and reduce `nev` to 2 or 4. Extreme Ekman numbers may require tighter tolerances.

## Performance

**Q: Memory usage spikes beyond expectations.**  
A: Lower `lmax` or `Nr`, switch `use_sparse_weighting` to `true` (default), and run on a machine with adequate RAM. Tri-global problems scale with `length(m_range) × lmax × Nr × 3`.

**Q: FEAST returns zero eigenvalues.**  
A: Adjust the search contour and the number of integration points. Ensure your contour encloses the target eigenvalues.

## Basic States

**Q: My custom base state produces NaNs.**  
A: Verify the dictionaries cover all `(ℓ, m)` pairs and that derivatives are consistent with the underlying grid. Start from `conduction_basic_state` and modify incrementally.

**Q: How do I load coefficients from external tools?**  
A: Convert your field to spectral coefficients via spherical harmonic transforms (e.g., `SHTns`, `SphericalHarmonics.jl`) and store them in the dictionaries expected by `BasicState` or `BasicState3D`.

## MHD Module

**Q: Enabling the MHD extension slows everything down.**  
A: The magnetic field doubles the number of variables. Use coarser grids while prototyping and preallocate arrays if you extend the code.

**Q: Magnetic boundary conditions confuse me.**  
A: `0` denotes insulating boundaries, `1` denotes conducting. Match them to your physical setup (e.g., Earth’s core uses insulating outer boundary).

## Documentation

**Q: MkDocs build fails with `ModuleNotFoundError`.**  
A: Install the required Python packages listed in `docs/requirements.txt` (created in this repo). Always activate the virtual environment before running `mkdocs serve`.

**Q: How do I publish on GitHub Pages?**  
A: Enable Pages for the repository, choose GitHub Actions, and use the sample workflow in `.github/workflows/docs.yml` (provided in this project).

## Still Stuck?

- Open an issue on GitHub with a minimal script reproducing the problem.
- Include Julia version, platform, and solver backend details.
- Provide logs by setting `ENV["CROSS_VERBOSE"] = "1"` for the failing run.
